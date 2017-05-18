import numpy as np
import pandas as pd
import operator
import re
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
import datetime
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
# train = pd.read_csv("train.csv",parse_dates=['timestamp'])
# macro = pd.read_csv('macro.csv',parse_dates=['timestamp'])
# test = pd.read_csv('test.csv',parse_dates=['timestamp'])
merged = pd.get_dummies(pd.read_csv('../merged.csv',parse_dates=['timestamp']))
# merged = pd.read_csv('../merged.csv',parse_dates=['timestamp'])


# merged = pd.merge(train,macro,on = 'timestamp')

#*** data cleanup

columnToRemove = re.compile("_all")

# all_columns_to_drop = merged.columns.map(lambda x: x if a.search(x) != None else None)
all_columns_to_drop = filter(lambda col: columnToRemove.search(col) != None, merged.columns)

merged = merged.drop(all_columns_to_drop, axis=1)

merged.loc[merged['state'] > 4, 'state'] = 4
merged.loc[merged['build_year'] == 20052009, 'build_year'] = 2007

output = merged['price_doc']
delCols = ['price_doc','timestamp']

merged = merged.drop(delCols, axis=1)

eval_size = 0.10
kf = KFold(len(output), round(1. / eval_size))
train_indices, valid_indices = next(iter(kf))

X_train, y_train = merged.iloc[train_indices], output.iloc[train_indices]
X_valid, y_valid = merged.iloc[valid_indices], output.iloc[valid_indices]

# print X_train

# text_columns = merged.select_dtypes(['string']).columns
# print text_columns
# exit()

# text_data_train = list(X_train.apply(lambda x: '%s %s %s %s %s %s' %(x['product_type'],x['sub_area'],x['ecology'],
#                                                          x['child_on_acc_pre_school'],
#                                                          x['modern_education_share'],x['old_education_build_share'],), axis=1))
# text_data_valid = list(X_valid.apply(lambda x: '%s %s %s %s %s %s' %(x['product_type'],x['sub_area'],x['ecology'],
#                                                          x['child_on_acc_pre_school'],x['modern_education_share'],
#                                                          x['old_education_build_share'],), axis=1))

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                      token_pattern=r'\W{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True,
                      sublinear_tf=1, stop_words='english')

# tfv_train = tfv.fit_transform(text_data_train)
# tfv_valid = tfv.fit_transform(text_data_valid)

tsvd = TruncatedSVD(n_components=320)
skb = SelectKBest(k=250)
xgbo = xgb.XGBRegressor(max_depth=7, learning_rate=0.01, n_estimators=500,
                        reg_alpha=0.1, reg_lambda=0.01,nthread=-1,gamma=0.05,subsample=0.6)

pipeline = Pipeline([("features", FeatureUnion([("tsvd", tsvd), ("skb", skb)])),
                     ("xgb", xgbo)])

# X_train.to_csv('xtrain01.csv', index=False)
a = datetime.datetime.now().replace(microsecond=0)
print "time now {0}".format(a)
model = pipeline.fit(X_train, y_train)
print ('test accuracy %.3f' % pipeline.score(X_valid, y_valid))
y_pred = pipeline.predict(X_valid)
b = datetime.datetime.now().replace(microsecond=0)

print "time now {0}".format(b)

print "total time taken {0}".format((b-a))

#0.388 score
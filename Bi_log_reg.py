# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 21:02:26 2020

@author: shara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
help(LogisticRegression)
banks = pd.read_csv("F:\Warun\\DS Assignments\\DS Assignments\\Binomial Log Reg\\bank-full.csv" , sep = ";")
banks.head()
type(banks)
del banks1
banks1 = banks.drop(columns = ["contact", "day","month" ,"pdays" ])
banks1.head() 
banks1.describe()
type(banks1['age'])
type(banks1['poutcome'])
banks1.dtypes
banks1.info()
for i in banks1.columns:
    if banks1[i].dtype == 'object' :
        banks1[i] = banks1[i].astype('category')
banks1.dtypes
banks1.job.value_counts()
banks1_t = banks1.describe().T
print(banks1_t)

category = ["job","marital","education","default","housing","loan","poutcome","y"]
for col in category:
    plt.figure(figsize = (11,6))
    sns.barplot(banks1[col].value_counts(),banks1[col].value_counts().index, data = banks1)
    plt.title(col)
    plt.tight_layout()

sns.boxplot(data = banks1, orient = "n", palette = "Set3")
banks1.boxplot(return_type = 'axes', figsize = (30,10))
column_list = []
iqr_list = []
out_low = []
out_up = []
tot_outlier = []


for i in banks1.describe().columns : 
    QTR1 = banks1[i].quantile(0.25)
    QTR3 = banks1[i].quantile(0.75)
    IQR = QTR3 - QTR1
    LTV = QTR1 - (1.5* IQR)
    UTV = QTR3 + (1.5 * IQR)
    current_column = i
    current_iqr = IQR
    bl_LTV = banks1[banks1[i] < LTV][i].count()
    ab_UTV = banks1[banks[i] > UTV][i].count()
    TOT_outliers = bl_LTV + ab_UTV
    column_list.append(current_column)
    iqr_list.append(current_iqr)
    out_low.append(bl_LTV)
    out_up.append(ab_UTV)
    tot_outlier.append(TOT_outliers)
    outlier_report = {"Column_name" : column_list, "IQR" : iqr_list, "Below_outliers" : out_low, "Above_outlier" : out_up, "Total_outliers" : tot_outlier}
    outlier_report = pd.DataFrame(outlier_report)
    print(outlier_report)
    
sns.boxplot(data = banks1.age , orient = "n", palette = "Set3")
sns.boxplot(data = banks1.duration , orient = "n", palette = "Set3")
sns.boxplot(data = banks1.campaign, orient = "n", palette = "Set3")
sns.boxplot(data = banks1.previous, orient = "n", palette = "Set3")
banks1.describe()
print(banks1.y.value_counts())
sns.pairplot(banks1)
banks1.corr()

bins = range(0,100,10)
bins1 = range(0,1000,100)

sns.distplot(banks1.age[banks1.y == "yes"] , bins = bins)
sns.distplot(banks1.age[banks1.y == "no"] , bins = bins)
sns.distplot(banks1.campaign[banks1.y == "yes"] , bins = bins)
sns.distplot(banks1.campaign[banks1.y == "no"], bins = bins )
sns.distplot(banks1.balance[banks1.y == "yes"], bins = bins1)
sns.distplot(banks1.balance[banks1.y == "no"], bins = bins1)
sns.distplot(banks1.duration[banks1.y == "yes"], bins = bins )
sns.distplot(banks1.duration[banks1.y == "no"] , bins = bins )
fig,ax2 = plt.subplots()
sns.countplot(banks1["campaign"], data = banks1, hue = 'y', ax = ax2)
fig,ax2 = plt.subplots()
sns.countplot(banks1["previous"], data = banks1, hue = 'y', ax = ax2)
fig,ax2 = plt.subplots()
sns.countplot(banks1["age"], data = banks1, hue = 'y', ax = ax2)
fig,ax2 = plt.subplots()
sns.countplot(banks1["balance"], data = banks1, hue = 'y', ax = ax2)
categorical_column = ['job', "marital", "education", "default", "housing", "loan","poutcome" ]
banks2 = pd.get_dummies(banks1, columns = categorical_column)
banks2.head()
banks2.shape
from sklearn.model_selection import train_test_split
x = banks2.drop('y', axis = 1)
y = banks2[['y']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20, random_state = 7)
x_train.shape
import sklearn
from sklearn import preprocessing
x_train_scaled = preprocessing.scale(x_train)
x_test_scaled = preprocessing.scale(x_test)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
seed = 10
# kfold = model_selection.KFold(n_splits = 10, random_state = seed)
# logreg = LogisticRegression(solver = 'lbfgs')
# logreg.fit(x_train,y_train)
# banks1['y'].dtype
# LR_Y_Pred = logreg.predict(x_test)
# LR_test_Score = logreg.score(x_test, y_test)
# test_accuracy = accuracy_score(y_test, LR_Y_Pred)
# cross_validation_result = model_selection.cross_val_score(logreg,x_train,y_train, cv = kfold, scoring = "accuracy")
sklearn.metrics.SCORERS.keys()
kfold2 = model_selection.KFold(n_splits = 10, shuffle = True)
# cross_validation_result2 = model_selection.cross_val_score(logreg,x_train,y_train, cv = kfold2, scoring = "accuracy")
# cls_report1 = classification_report(y_test, LR_Y_Pred, output_dict = True)
# pd.DataFrame(cls_report1)
# x_train_pred = logreg.predict(x_train)
# train_accuracy = accuracy_score(y_train, x_train_pred)
pip install -U imbalanced-learn
from imblearn.over_sampling import SMOTE
SMOTE = SMOTE(random_state = 0)
x_smote, y_smote = SMOTE.fit_resample(x, y)
y_smote.y.value()
y_smote.y.value_counts()
from imblearn.over_sampling import RandomOverSampler
over_sampler = RandomOverSampler(random_state = 0)
x_over,y_over = over_sampler.fit_resample(x,y)
y_over.y.value_counts()
x_s_train,x_s_test, y_s_train, y_s_test = train_test_split(x_smote,y_smote, test_size = 0.2,random_state = 7)
x_s_train_scaled = preprocessing.scale(x_s_train)
x_s_test_scaled = preprocessing.scale(x_s_test)
logreg = LogisticRegression(solver = 'lbfgs')
logreg.fit(x_s_train_scaled,y_s_train)
SMOTE_Y_Pred = logreg.predict(x_s_test_scaled)
SMOTE_test_Score = accuracy_score(y_s_test, SMOTE_Y_Pred)
kfold = model_selection.KFold(n_splits = 10, random_state = seed)
cross_validation_result = model_selection.cross_val_score(logreg,x_s_test_scaled,y_s_test, cv = kfold, scoring = "accuracy")
SMOTE_cls_report_test = classification_report(y_s_test, SMOTE_Y_Pred, output_dict = True)
pd.DataFrame(SMOTE_cls_report_test)
SMOTE_train_pred = logreg.predict(x_s_train_scaled)
SMOTE_train_Score = accuracy_score(y_s_train, SMOTE_train_pred)
cross_validation_result_train = model_selection.cross_val_score(logreg,x_s_train_scaled,y_s_train, cv = kfold2, scoring = "accuracy")
SMOTE_cls_report_train = classification_report(y_s_train, SMOTE_train_pred, output_dict = True)
pd.DataFrame(SMOTE_cls_report_train)

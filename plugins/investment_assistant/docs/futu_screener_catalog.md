# Futu Screener Catalog Snapshot

- Generated at: `2026-05-21T15:25:17.432515+00:00`
- Source: `manual_futu_screener_catalog_export`
- Futu SDK version: `10.05.6508`
- JSON snapshot: `/Users/roy_li/gits/hermes-agent/plugins/investment_assistant/data/futu_screener_catalog.json`
- Refresh command: `.venv/bin/python plugins/investment_assistant/scripts/export_futu_screener_catalog.py --markets US,HK,SH,SZ`

This file is the offline option catalog used by the investment assistant's Futu-assisted discovery tool. Refresh it manually when Futu App/OpenAPI choices change.

## Market `HK`

### App-Style Screener Categories

#### 行情 (`market_quote`)

Use market/plate to constrain the universe, then quote and accumulated fields to find liquid, tradable, strong or weak candidates.

- **交易所/市场**: `market_selector`
  - Supported values: `US, HK, SH, SZ`
  - LLM hint: Select the listing market. This is not a StockField.
- **所属行业/概念/板块**: `plate_selector`
  - LLM hint: Probe theme words as plate_keywords. If a useful plate_code is found, stock_filter can filter within that plate_code.
- **价格/52周位置/量比/委比/每手价格**: `stock_filter`
  - Fields: `CUR_PRICE, CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO, CUR_PRICE_TO_LOWEST52_WEEKS_RATIO, HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO, LOW_PRICE_TO_LOWEST52_WEEKS_RATIO, VOLUME_RATIO, BID_ASK_RATIO, LOT_PRICE`
  - LLM hint: Use these for tradability, momentum context, and avoiding illiquid tails.
- **涨跌幅/振幅/成交量/成交额/换手率**: `stock_filter`
  - Fields: `CHANGE_RATE, AMPLITUDE, VOLUME, TURNOVER, TURNOVER_RATE`
  - LLM hint: Use days plus sort/filter bounds to discover strong momentum or high liquidity names, e.g. TURNOVER DESCEND or CHANGE_RATE over 20/60 days.

#### 估值 (`valuation`)

Use valuation as a secondary screen after thematic relevance; do not let cheapness replace theme fit.

- **市值/流通市值/股本**: `stock_filter`
  - Fields: `MARKET_VAL, FLOAT_MARKET_VAL, TOTAL_SHARE, FLOAT_SHARE`
  - LLM hint: Use market cap to separate mega-cap anchors from small high-beta satellites.
- **市盈率（静态/TTM）/市净率/市销率/市现率**: `stock_filter`
  - Fields: `PE_ANNUAL, PE_TTM, PB_RATE, PS_TTM, PCF_TTM`
  - LLM hint: Useful for valuation risk flags and relative comparison, not theme discovery alone.
- **估值分位/行业估值分位**: `derived_or_future_adapter`
  - Alternate source: `derive from peer universe snapshots/financials; not exposed as current get_stock_filter StockField`
  - LLM hint: Do not pass valuation percentile as stock_filter_specs. Mark it as desired downstream enrichment if needed.

#### 分红 (`dividend`)

Dividend data is enrichment, not a get_stock_filter field in this SDK.

- **TTM 分红/股息率**: `non_stock_filter`
  - Alternate source: `get_market_snapshot fields dividend_ttm/dividend_ratio_ttm`
  - LLM hint: Do not create stock_filter_specs for dividends; request quote snapshot enrichment later.

#### 技术 (`technical`)

Use technical filters to discover trend candidates or confirm that a theme is active; avoid using them as the only reason a name belongs to a theme.

- **指标解读**: `derived_interpretation`
  - Alternate source: `derive from pattern/custom_indicator StockFields and K-line enrichment`
  - LLM hint: The app-style interpretation layer is not a single OpenAPI enum. Use the specific MA/EMA/KDJ/RSI/MACD/BOLL fields below.
- **MA/EMA 均线形态**: `stock_filter`
  - Fields: `MA_ALIGNMENT_LONG, MA_ALIGNMENT_SHORT, EMA_ALIGNMENT_LONG, EMA_ALIGNMENT_SHORT`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use for trend/momentum probes, e.g. MA_ALIGNMENT_LONG on K_DAY.
- **RSI/KDJ/MACD/BOLL 形态**: `stock_filter`
  - Fields: `RSI_GOLD_CROSS_LOW, RSI_DEATH_CROSS_HIGH, RSI_TOP_DIVERGENCE, RSI_BOTTOM_DIVERGENCE, KDJ_GOLD_CROSS_LOW, KDJ_DEATH_CROSS_HIGH, KDJ_TOP_DIVERGENCE, KDJ_BOTTOM_DIVERGENCE, MACD_GOLD_CROSS_LOW, MACD_DEATH_CROSS_HIGH, MACD_TOP_DIVERGENCE, MACD_BOTTOM_DIVERGENCE, BOLL_BREAK_UPPER, BOLL_BREAK_LOWER, BOLL_CROSS_MIDDLE_UP, BOLL_CROSS_MIDDLE_DOWN`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use for oscillator and Bollinger-band signals when discovering active candidates.
- **MA/EMA/KDJ/RSI/MACD/BOLL 自定义指标比较**: `stock_filter`
  - Fields: `PRICE, MA5, MA10, MA20, MA30, MA60, MA120, MA250, RSI, EMA5, EMA10, EMA20, EMA30, EMA60, EMA120, EMA250, VALUE, MA, EMA, KDJ_K, KDJ_D, KDJ_J, MACD_DIFF, MACD_DEA, MACD, BOLL_UPPER, BOLL_MIDDLER, BOLL_LOWER`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use only when a simple pattern field is insufficient.

#### 财务 (`financial`)

Use fundamentals to filter quality and growth after the theme map is drafted.

- **利润/收入/利润率/ROE/ROIC/现金流/资产负债/增长率/EPS**: `stock_filter`
  - Fields: `NET_PROFIT, NET_PROFIX_GROWTH, SUM_OF_BUSINESS, SUM_OF_BUSINESS_GROWTH, NET_PROFIT_RATE, GROSS_PROFIT_RATE, DEBT_ASSET_RATE, RETURN_ON_EQUITY_RATE, ROIC, ROA_TTM, EBIT_TTM, EBITDA, OPERATING_MARGIN_TTM, EBIT_MARGIN, EBITDA_MARGIN, FINANCIAL_COST_RATE, OPERATING_PROFIT_TTM, SHAREHOLDER_NET_PROFIT_TTM, NET_PROFIT_CASH_COVER_TTM, CURRENT_RATIO, QUICK_RATIO, CURRENT_ASSET_RATIO, CURRENT_DEBT_RATIO, EQUITY_MULTIPLIER, PROPERTY_RATIO, CASH_AND_CASH_EQUIVALENTS, TOTAL_ASSET_TURNOVER, FIXED_ASSET_TURNOVER, INVENTORY_TURNOVER, OPERATING_CASH_FLOW_TTM, ACCOUNTS_RECEIVABLE, EBIT_GROWTH_RATE, OPERATING_PROFIT_GROWTH_RATE, TOTAL_ASSETS_GROWTH_RATE, PROFIT_TO_SHAREHOLDERS_GROWTH_RATE, PROFIT_BEFORE_TAX_GROWTH_RATE, EPS_GROWTH_RATE, ROE_GROWTH_RATE, ROIC_GROWTH_RATE, NOCF_GROWTH_RATE, NOCF_PER_SHARE_GROWTH_RATE, OPERATING_REVENUE_CASH_COVER, OPERATING_PROFIT_TO_TOTAL_PROFIT, BASIC_EPS, DILUTED_EPS, NOCF_PER_SHARE`
  - LLM hint: Useful probes: revenue growth, gross margin, ROE/ROIC, operating cash flow, debt ratio. Exact numbers still need downstream SEC/fundamental validation.

#### 分析 (`analysis`)

Analyst ratings/revisions are not exposed by get_stock_filter here.

- **评级/目标价/一致预期/盈利修正**: `external_or_future_adapter`
  - Alternate source: `analyst_estimate_revision_adapter`
  - LLM hint: Do not invent these facts. Mark as needed evidence if useful.

#### 期权 (`options`)

Options are not stock_filter fields. Use option-chain/surface enrichment after candidate discovery.

- **期权活跃度/IV/到期日/Put-Call context**: `option_chain_enrichment`
  - Alternate source: `get_option_expiration_date + get_option_chain + option snapshots`
  - LLM hint: Use later to assess tradability and income-strategy readiness, not as stock_filter_specs.

### OpenAPI Stock Filter Fields

#### `simple` (21)

`STOCK_CODE`, `STOCK_NAME`, `CUR_PRICE`, `CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO`, `CUR_PRICE_TO_LOWEST52_WEEKS_RATIO`, `HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO`, `LOW_PRICE_TO_LOWEST52_WEEKS_RATIO`, `VOLUME_RATIO`, `BID_ASK_RATIO`, `LOT_PRICE`, `MARKET_VAL`, `PE_ANNUAL`, `PE_TTM`, `PB_RATE`, `CHANGE_RATE_5MIN`, `CHANGE_RATE_BEGIN_YEAR`, `PS_TTM`, `PCF_TTM`, `TOTAL_SHARE`, `FLOAT_SHARE`, `FLOAT_MARKET_VAL`

#### `accumulate` (5)

`CHANGE_RATE`, `AMPLITUDE`, `VOLUME`, `TURNOVER`, `TURNOVER_RATE`

#### `financial` (46)

`NET_PROFIT`, `NET_PROFIX_GROWTH`, `SUM_OF_BUSINESS`, `SUM_OF_BUSINESS_GROWTH`, `NET_PROFIT_RATE`, `GROSS_PROFIT_RATE`, `DEBT_ASSET_RATE`, `RETURN_ON_EQUITY_RATE`, `ROIC`, `ROA_TTM`, `EBIT_TTM`, `EBITDA`, `OPERATING_MARGIN_TTM`, `EBIT_MARGIN`, `EBITDA_MARGIN`, `FINANCIAL_COST_RATE`, `OPERATING_PROFIT_TTM`, `SHAREHOLDER_NET_PROFIT_TTM`, `NET_PROFIT_CASH_COVER_TTM`, `CURRENT_RATIO`, `QUICK_RATIO`, `CURRENT_ASSET_RATIO`, `CURRENT_DEBT_RATIO`, `EQUITY_MULTIPLIER`, `PROPERTY_RATIO`, `CASH_AND_CASH_EQUIVALENTS`, `TOTAL_ASSET_TURNOVER`, `FIXED_ASSET_TURNOVER`, `INVENTORY_TURNOVER`, `OPERATING_CASH_FLOW_TTM`, `ACCOUNTS_RECEIVABLE`, `EBIT_GROWTH_RATE`, `OPERATING_PROFIT_GROWTH_RATE`, `TOTAL_ASSETS_GROWTH_RATE`, `PROFIT_TO_SHAREHOLDERS_GROWTH_RATE`, `PROFIT_BEFORE_TAX_GROWTH_RATE`, `EPS_GROWTH_RATE`, `ROE_GROWTH_RATE`, `ROIC_GROWTH_RATE`, `NOCF_GROWTH_RATE`, `NOCF_PER_SHARE_GROWTH_RATE`, `OPERATING_REVENUE_CASH_COVER`, `OPERATING_PROFIT_TO_TOTAL_PROFIT`, `BASIC_EPS`, `DILUTED_EPS`, `NOCF_PER_SHARE`

#### `pattern` (20)

`MA_ALIGNMENT_LONG`, `MA_ALIGNMENT_SHORT`, `EMA_ALIGNMENT_LONG`, `EMA_ALIGNMENT_SHORT`, `RSI_GOLD_CROSS_LOW`, `RSI_DEATH_CROSS_HIGH`, `RSI_TOP_DIVERGENCE`, `RSI_BOTTOM_DIVERGENCE`, `KDJ_GOLD_CROSS_LOW`, `KDJ_DEATH_CROSS_HIGH`, `KDJ_TOP_DIVERGENCE`, `KDJ_BOTTOM_DIVERGENCE`, `MACD_GOLD_CROSS_LOW`, `MACD_DEATH_CROSS_HIGH`, `MACD_TOP_DIVERGENCE`, `MACD_BOTTOM_DIVERGENCE`, `BOLL_BREAK_UPPER`, `BOLL_BREAK_LOWER`, `BOLL_CROSS_MIDDLE_UP`, `BOLL_CROSS_MIDDLE_DOWN`

#### `custom_indicator` (28)

`PRICE`, `MA5`, `MA10`, `MA20`, `MA30`, `MA60`, `MA120`, `MA250`, `RSI`, `EMA5`, `EMA10`, `EMA20`, `EMA30`, `EMA60`, `EMA120`, `EMA250`, `VALUE`, `MA`, `EMA`, `KDJ_K`, `KDJ_D`, `KDJ_J`, `MACD_DIFF`, `MACD_DEA`, `MACD`, `BOLL_UPPER`, `BOLL_MIDDLER`, `BOLL_LOWER`

- `sort_dir`: `ASCEND`, `DESCEND`
- `financial_quarter`: `ANNUAL`, `FIRST_QUARTER`, `INTERIM`, `THIRD_QUARTER`, `MOST_RECENT_QUARTER`
- `supported_pattern_ktype`: `K_60M`, `K_DAY`, `K_WEEK`, `K_MON`
- `relative_position`: `MORE`, `LESS`, `CROSS_UP`, `CROSS_DOWN`

### Futu Plate Choices

#### `ALL` (295)

| code | name | plate_type | plate_id |
|---|---|---|---|
| HK.GANGGUTONG | 港股通(沪) | ALL | GangGuTong |
| HK.LIST1001 | 乳制品 | ALL | LIST1001 |
| HK.LIST1002 | 采购及供应链管理 | ALL | LIST1002 |
| HK.LIST1003 | 保险 | ALL | LIST1003 |
| HK.LIST1004 | 信贷 | ALL | LIST1004 |
| HK.LIST1005 | 公共运输 | ALL | LIST1005 |
| HK.LIST1006 | 其他金属及矿物 | ALL | LIST1006 |
| HK.LIST1007 | 其他金融 | ALL | LIST1007 |
| HK.LIST1008 | 农产品 | ALL | LIST1008 |
| HK.LIST1009 | 出版 | ALL | LIST1009 |
| HK.LIST1010 | 包装食品 | ALL | LIST1010 |
| HK.LIST1011 | 化肥及农用化合物 | ALL | LIST1011 |
| HK.LIST1012 | 医疗设备及用品 | ALL | LIST1012 |
| HK.LIST1013 | 半导体 | ALL | LIST1013 |
| HK.LIST1014 | 卫星及无线通讯 | ALL | LIST1014 |
| HK.LIST1015 | 印刷及包装 | ALL | LIST1015 |
| HK.LIST1016 | 非传统/可再生能源 | ALL | LIST1016 |
| HK.LIST1017 | 商业用车及货车 | ALL | LIST1017 |
| HK.LIST1019 | 地产发展商 | ALL | LIST1019 |
| HK.LIST1020 | 地产投资 | ALL | LIST1020 |
| HK.LIST1021 | 家具 | ALL | LIST1021 |
| HK.LIST1022 | 家庭电器 | ALL | LIST1022 |
| HK.LIST1025 | 工业零件及器材 | ALL | LIST1025 |
| HK.LIST1026 | 广告及宣传 | ALL | LIST1026 |
| HK.LIST1027 | 广播 | ALL | LIST1027 |
| HK.LIST1028 | 建筑材料 | ALL | LIST1028 |
| HK.LIST1029 | 影视娱乐 | ALL | LIST1029 |
| HK.LIST1030 | 投资及资产管理 | ALL | LIST1030 |
| HK.LIST1031 | 其他支援服务 | ALL | LIST1031 |
| HK.LIST1032 | 消闲及文娱设施 | ALL | LIST1032 |
| HK.LIST1033 | 新能源物料 | ALL | LIST1033 |
| HK.LIST1034 | 旅游及观光 | ALL | LIST1034 |
| HK.LIST1035 | 纺织品及布料 | ALL | LIST1035 |
| HK.LIST1037 | 林业及木材 | ALL | LIST1037 |
| HK.LIST1039 | 水务 | ALL | LIST1039 |
| HK.LIST1040 | 汽车 | ALL | LIST1040 |
| HK.LIST1041 | 汽车零件 | ALL | LIST1041 |
| HK.LIST1042 | 油气生产商 | ALL | LIST1042 |
| HK.LIST1043 | 油气设备与服务 | ALL | LIST1043 |
| HK.LIST1044 | 煤炭 | ALL | LIST1044 |
| HK.LIST1045 | 燃气供应 | ALL | LIST1045 |
| HK.LIST1046 | 特殊化工用品 | ALL | LIST1046 |
| HK.LIST1047 | 玩具及消闲用品 | ALL | LIST1047 |
| HK.LIST1049 | 珠宝钟表 | ALL | LIST1049 |
| HK.LIST1050 | 生物技术 | ALL | LIST1050 |
| HK.LIST1051 | 常规电力 | ALL | LIST1051 |
| HK.LIST1052 | 消费电子产品 | ALL | LIST1052 |
| HK.LIST1053 | 电脑及周边器材 | ALL | LIST1053 |
| HK.LIST1054 | 电讯服务 | ALL | LIST1054 |
| HK.LIST1055 | 消费性电讯设备 | ALL | LIST1055 |
| HK.LIST1056 | 多元化零售商 | ALL | LIST1056 |
| HK.LIST1059 | 纸及纸制品 | ALL | LIST1059 |
| HK.LIST1061 | 综合企业 | ALL | LIST1061 |
| HK.LIST1062 | 个人护理 | ALL | LIST1062 |
| HK.LIST1063 | 航空航天与国防 | ALL | LIST1063 |
| HK.LIST1064 | 航空服务 | ALL | LIST1064 |
| HK.LIST1065 | 航空货运及物流 | ALL | LIST1065 |
| HK.LIST1066 | 航运及港口 | ALL | LIST1066 |
| HK.LIST1067 | 药品 | ALL | LIST1067 |
| HK.LIST1068 | 证券及经纪 | ALL | LIST1068 |
| HK.LIST1069 | 赌场及博彩 | ALL | LIST1069 |
| HK.LIST1070 | 超市及便利店 | ALL | LIST1070 |
| HK.LIST1071 | 酒店及度假村 | ALL | LIST1071 |
| HK.LIST1072 | 酒精饮料 | ALL | LIST1072 |
| HK.LIST1073 | 重型基建 | ALL | LIST1073 |
| HK.LIST1074 | 重型机械 | ALL | LIST1074 |
| HK.LIST1075 | 钢铁 | ALL | LIST1075 |
| HK.LIST1076 | 铁路及公路 | ALL | LIST1076 |
| HK.LIST1077 | 铜 | ALL | LIST1077 |
| HK.LIST1078 | 铝 | ALL | LIST1078 |
| HK.LIST1079 | 银行 | ALL | LIST1079 |
| HK.LIST1080 | 非酒精饮料 | ALL | LIST1080 |
| HK.LIST1082 | 食品添加剂 | ALL | LIST1082 |
| HK.LIST1083 | 餐饮 | ALL | LIST1083 |
| HK.LIST1084 | 黄金及贵金属 | ALL | LIST1084 |
| HK.LIST1086 | 医疗及医学美容服务 | ALL | LIST1086 |
| HK.LIST1087 | 房地产信托 | ALL | LIST1087 |
| HK.LIST1089 | 地产代理 | ALL | LIST1089 |
| HK.LIST1090 | 物业服务及管理 | ALL | LIST1090 |
| HK.LIST1091 | 教育 | ALL | LIST1091 |
| HK.LIST1095 | 楼宇建造 | ALL | LIST1095 |
| HK.LIST1100 | 应用软件 | ALL | LIST1100 |
| HK.LIST1106 | 红筹股 | ALL | LIST1106 |
| HK.LIST1107 | 蓝筹股 | ALL | LIST1107 |
| HK.LIST1110 | 阿里概念股 | ALL | LIST1110 |
| HK.LIST1113 | 彩票股 | ALL | LIST1113 |
| HK.LIST1114 | 博彩股 | ALL | LIST1114 |
| HK.LIST1116 | 医疗保健 | ALL | LIST1116 |
| HK.LIST1123 | 军工股 | ALL | LIST1123 |
| HK.LIST1126 | 房地产基金 | ALL | LIST1126 |
| HK.LIST1130 | 保险股 | ALL | LIST1130 |
| HK.LIST1136 | 页岩气 | ALL | LIST1136 |
| HK.LIST1149 | 国内零售股 | ALL | LIST1149 |
| HK.LIST1152 | 二胎 | ALL | LIST1152 |
| HK.LIST1161 | 双十一 | ALL | LIST1161 |
| HK.LIST1166 | 手机产业链 | ALL | LIST1166 |
| HK.LIST1167 | LED | ALL | LIST1167 |
| HK.LIST1168 | MSCI中国大陆小型股 | ALL | LIST1168 |
| HK.LIST1169 | MSCI中国香港小型股 | ALL | LIST1169 |
| HK.LIST1175 | 一带一路 | ALL | LIST1175 |
| HK.LIST1176 | 5G概念 | ALL | LIST1176 |
| HK.LIST1178 | 粤港澳大湾区 | ALL | LIST1178 |
| HK.LIST1180 | 特斯拉概念股 | ALL | LIST1180 |
| HK.LIST1181 | 啤酒 | ALL | LIST1181 |
| HK.LIST1185 | 体育用品 | ALL | LIST1185 |
| HK.LIST1186 | 稀土概念 | ALL | LIST1186 |
| HK.LIST1190 | 腾讯概念 | ALL | LIST1190 |
| HK.LIST1191 | 云办公 | ALL | LIST1191 |
| HK.LIST1192 | SaaS概念 | ALL | LIST1192 |
| HK.LIST1193 | 在线教育 | ALL | LIST1193 |
| HK.LIST1196 | 汽车经销商 | ALL | LIST1196 |
| HK.LIST1200 | 核电 | ALL | LIST1200 |
| HK.LIST1202 | 化妆品 | ALL | LIST1202 |
| HK.LIST1205 | 石油股 | ALL | LIST1205 |
| HK.LIST1206 | 电讯设备 | ALL | LIST1206 |
| HK.LIST1207 | 电力股 | ALL | LIST1207 |
| HK.LIST1208 | 手游股 | ALL | LIST1208 |
| HK.LIST1209 | 婴童用品股 | ALL | LIST1209 |
| HK.LIST1210 | 百货业股 | ALL | LIST1210 |
| HK.LIST1213 | 港口运输股 | ALL | LIST1213 |
| HK.LIST1214 | 电信股 | ALL | LIST1214 |
| HK.LIST1215 | 环保 | ALL | LIST1215 |
| HK.LIST1216 | 煤炭股 | ALL | LIST1216 |
| HK.LIST1217 | 综合车企股 | ALL | LIST1217 |
| HK.LIST1218 | 电池 | ALL | LIST1218 |
| HK.LIST1219 | 物流 | ALL | LIST1219 |
| HK.LIST1220 | 内地物业管理股 | ALL | LIST1220 |
| HK.LIST1221 | 农业股 | ALL | LIST1221 |
| HK.LIST1222 | 黄金股 | ALL | LIST1222 |
| HK.LIST1223 | 奢侈品品牌股 | ALL | LIST1223 |
| HK.LIST1224 | 电力设备股 | ALL | LIST1224 |
| HK.LIST1226 | 重型机械股 | ALL | LIST1226 |
| HK.LIST1227 | 食品股 | ALL | LIST1227 |
| HK.LIST1230 | 纸业股 | ALL | LIST1230 |
| HK.LIST1231 | 水务股 | ALL | LIST1231 |
| HK.LIST1232 | 奶制品股 | ALL | LIST1232 |
| HK.LIST1233 | 光伏太阳能股 | ALL | LIST1233 |
| HK.LIST1234 | 内房股 | ALL | LIST1234 |
| HK.LIST1235 | 内地教育股 | ALL | LIST1235 |
| HK.LIST1236 | 家电股 | ALL | LIST1236 |
| HK.LIST1237 | 风电股 | ALL | LIST1237 |
| HK.LIST1239 | 内银股 | ALL | LIST1239 |
| HK.LIST1240 | 航空股 | ALL | LIST1240 |
| HK.LIST1241 | 石油与天然气 | ALL | LIST1241 |
| HK.LIST1242 | 建材水泥股 | ALL | LIST1242 |
| HK.LIST1243 | 中资券商股 | ALL | LIST1243 |
| HK.LIST1244 | 高铁基建股 | ALL | LIST1244 |
| HK.LIST1245 | 燃气股 | ALL | LIST1245 |
| HK.LIST1246 | 公路及铁路股 | ALL | LIST1246 |
| HK.LIST1251 | 互联网医疗 | ALL | LIST1251 |
| HK.LIST1252 | 香港本地银行股 | ALL | LIST1252 |
| HK.LIST1254 | 生物医药B类股 | ALL | LIST1254 |
| HK.LIST1261 | 医药外包概念 | ALL | LIST1261 |
| HK.LIST1263 | 香港本地消费股 | ALL | LIST1263 |
| HK.LIST1266 | 抖音概念股 | ALL | LIST1266 |
| HK.LIST1267 | 烟草及电子烟股 | ALL | LIST1267 |
| HK.LIST1268 | 其他服饰配件 | ALL | LIST1268 |
| HK.LIST1269 | 汽车零售商 | ALL | LIST1269 |
| HK.LIST1270 | 服装零售商 | ALL | LIST1270 |
| HK.LIST1271 | 环保工程 | ALL | LIST1271 |
| HK.LIST1272 | 禽畜饲料 | ALL | LIST1272 |
| HK.LIST1273 | 禽畜肉类 | ALL | LIST1273 |
| HK.LIST1274 | 电子零件 | ALL | LIST1274 |
| HK.LIST1275 | 鞋类 | ALL | LIST1275 |
| HK.LIST1276 | 其他零售商 | ALL | LIST1276 |
| HK.LIST1277 | 服装 | ALL | LIST1277 |
| HK.LIST1278 | 家居装修零售商 | ALL | LIST1278 |
| HK.LIST1279 | 影视股 | ALL | LIST1279 |
| HK.LIST1284 | 中医药 | ALL | LIST1284 |
| HK.LIST1287 | 蚂蚁金服概念 | ALL | LIST1287 |
| HK.LIST1288 | 昨日强势股 | ALL | LIST1288 |
| HK.LIST1289 | 云计算 | ALL | LIST1289 |
| HK.LIST1290 | 次新股 | ALL | LIST1290 |
| HK.LIST1304 | 回港中概股 | ALL | LIST1304 |
| HK.LIST1305 | 北水核心资产 | ALL | LIST1305 |
| HK.LIST1306 | 短视频概念股 | ALL | LIST1306 |
| HK.LIST1311 | 房地产投资信托 | ALL | LIST1311 |
| HK.LIST1312 | 有色金属 | ALL | LIST1312 |
| HK.LIST1313 | 医美概念股 | ALL | LIST1313 |
| HK.LIST1314 | 碳中和概念股 | ALL | LIST1314 |
| HK.LIST1319 | 民办高教 | ALL | LIST1319 |
| HK.LIST1320 | K12教育 | ALL | LIST1320 |
| HK.LIST1321 | 三胎概念 | ALL | LIST1321 |
| HK.LIST1326 | 中医药概念 | ALL | LIST1326 |
| HK.LIST1328 | 元宇宙概念 | ALL | LIST1328 |
| HK.LIST1329 | 绿电概念 | ALL | LIST1329 |
| HK.LIST1331 | 小米概念 | ALL | LIST1331 |
| HK.LIST1334 | 锂电池 | ALL | LIST1334 |
| HK.LIST1335 | 养老概念 | ALL | LIST1335 |
| HK.LIST1336 | 职业教育 | ALL | LIST1336 |
| HK.LIST1342 | 氢能源概念股 | ALL | LIST1342 |
| HK.LIST1344 | 高股息概念 | ALL | LIST1344 |
| HK.LIST1348 | 中特估-国企 | ALL | LIST1348 |
| HK.LIST1351 | 虚拟现实 | ALL | LIST1351 |
| HK.LIST1353 | 室温超导概念 | ALL | LIST1353 |
| HK.LIST1354 | 能源储存装置 | ALL | LIST1354 |
| HK.LIST1355 | 公路运输 | ALL | LIST1355 |
| HK.LIST1356 | 烟草 | ALL | LIST1356 |
| HK.LIST1357 | 药品分销 | ALL | LIST1357 |
| HK.LIST1358 | 核能 | ALL | LIST1358 |
| HK.LIST1359 | 游戏软件 | ALL | LIST1359 |
| HK.LIST1360 | 半导体设备与材料 | ALL | LIST1360 |
| HK.LIST1362 | 以巴冲突 | ALL | LIST1362 |
| HK.LIST1922 | 港股通(深) | ALL | LIST1922 |
| HK.LIST1983 | 香港股票ADR | ALL | LIST1983 |
| HK.LIST1984 | 港股新经济 | ALL | LIST1984 |
| HK.LIST1991 | OLED概念 | ALL | LIST1991 |
| HK.LIST1992 | 工业大麻 | ALL | LIST1992 |
| HK.LIST1994 | 香港零售股 | ALL | LIST1994 |
| HK.LIST1996 | 猪肉概念 | ALL | LIST1996 |
| HK.LIST1998 | 节假日概念股 | ALL | LIST1998 |
| HK.LIST1999 | 殡葬概念 | ALL | LIST1999 |
| HK.LIST20074 | 加密货币概念股 | ALL | LIST20074 |
| HK.LIST20815 | 商品 | ALL | LIST20815 |
| HK.LIST20816 | 股票 | ALL | LIST20816 |
| HK.LIST20817 | 定息产品 | ALL | LIST20817 |
| HK.LIST20818 | 货币市场 | ALL | LIST20818 |
| HK.LIST20820 | 股息ETF | ALL | LIST20820 |
| HK.LIST20822 | 黄金ETF | ALL | LIST20822 |
| HK.LIST20823 | 政府债券ETF | ALL | LIST20823 |
| HK.LIST20827 | 铁矿石ETF | ALL | LIST20827 |
| HK.LIST20833 | 原油ETF | ALL | LIST20833 |
| HK.LIST20834 | 消费品及服务 | ALL | LIST20834 |
| HK.LIST20835 | 能源 | ALL | LIST20835 |
| HK.LIST20837 | 医疗保健 | ALL | LIST20837 |
| HK.LIST20838 | 物料 | ALL | LIST20838 |
| HK.LIST20839 | 房地产 | ALL | LIST20839 |
| HK.LIST20840 | 科技 | ALL | LIST20840 |
| HK.LIST20841 | 运输 | ALL | LIST20841 |
| HK.LIST20844 | 东盟 | ALL | LIST20844 |
| HK.LIST20845 | 亚洲(除日本) | ALL | LIST20845 |
| HK.LIST20846 | 亚太区(除日本) | ALL | LIST20846 |
| HK.LIST20847 | 中国 | ALL | LIST20847 |
| HK.LIST20848 | 欧洲 | ALL | LIST20848 |
| HK.LIST20849 | 环球 | ALL | LIST20849 |
| HK.LIST20850 | 全球新兴市场 | ALL | LIST20850 |
| HK.LIST20851 | 大中华 | ALL | LIST20851 |
| HK.LIST20852 | 中国香港 | ALL | LIST20852 |
| HK.LIST20853 | 印度 | ALL | LIST20853 |
| HK.LIST20854 | 日本 | ALL | LIST20854 |
| HK.LIST20855 | 韩国 | ALL | LIST20855 |
| HK.LIST20856 | 沙特阿拉伯 | ALL | LIST20856 |
| HK.LIST20857 | 中国台湾 | ALL | LIST20857 |
| HK.LIST20858 | 美国 | ALL | LIST20858 |
| HK.LIST20859 | 越南 | ALL | LIST20859 |
| HK.LIST20881 | 亚太区 | ALL | LIST20881 |
| HK.LIST21046 | 加密货币期货ETF | ALL | LIST21046 |
| HK.LIST21047 | 人工智能ETF | ALL | LIST21047 |
| HK.LIST21048 | 消费ETF | ALL | LIST21048 |
| HK.LIST21049 | 游戏娱乐ETF | ALL | LIST21049 |
| HK.LIST21050 | ESG和清洁能源ETF | ALL | LIST21050 |
| HK.LIST21051 | 地产ETF | ALL | LIST21051 |
| HK.LIST21052 | 互联网ETF | ALL | LIST21052 |
| HK.LIST21053 | 半导体ETF | ALL | LIST21053 |
| HK.LIST21054 | 科技创新ETF | ALL | LIST21054 |
| HK.LIST21055 | 汽车ETF | ALL | LIST21055 |
| HK.LIST21062 | 追踪日本ETF | ALL | LIST21062 |
| HK.LIST22873 | 加密货币现货ETF | ALL | LIST22873 |
| HK.LIST22886 | 明星科网股 | ALL | LIST22886 |
| HK.LIST22910 | 新能源车企 | ALL | LIST22910 |
| HK.LIST22911 | 生物医药 | ALL | LIST22911 |
| HK.LIST22912 | 芯片股 | ALL | LIST22912 |
| HK.LIST22913 | 博彩股 | ALL | LIST22913 |
| HK.LIST22928 | 美国降息利好概念 | ALL | LIST22928 |
| HK.LIST23360 | 互动媒体及服务 | ALL | LIST23360 |
| HK.LIST23361 | 线上零售商 | ALL | LIST23361 |
| HK.LIST23362 | 支付服务 | ALL | LIST23362 |
| HK.LIST23363 | 数码解决方案服务 | ALL | LIST23363 |
| HK.LIST23364 | 互联网服务及基础设施 | ALL | LIST23364 |
| HK.LIST23404 | A股ETF | ALL | LIST23404 |
| HK.LIST23408 | 高股息ETF | ALL | LIST23408 |
| HK.LIST23586 | 人工智能 | ALL | LIST23586 |
| HK.LIST23589 | 智能驾驶概念股 | ALL | LIST23589 |
| HK.LIST23593 | 机器人概念股 | ALL | LIST23593 |
| HK.LIST23598 | AI医疗概念股 | ALL | LIST23598 |
| HK.LIST23674 | 稳定币概念 | ALL | LIST23674 |
| HK.LIST23675 | 创新药概念 | ALL | LIST23675 |
| HK.LIST23676 | 新消费概念 | ALL | LIST23676 |
| HK.LIST23699 | 脑机接口概念 | ALL | LIST23699 |
| HK.LIST23846 | 轨道与列车设备 | ALL | LIST23846 |
| HK.LIST23847 | 摩托车及其他 | ALL | LIST23847 |
| HK.LIST23848 | 家居消耗品 | ALL | LIST23848 |
| HK.LIST23849 | 膳食补充品 | ALL | LIST23849 |
| HK.LIST23850 | 护肤与化妆品 | ALL | LIST23850 |
| HK.LIST23851 | 电讯网路基建设施 | ALL | LIST23851 |
| HK.LIST24024 | 光通信 | ALL | LIST24024 |
| HK.LIST24031 | 商业航天 | ALL | LIST24031 |
| HK.LIST24032 | 茶饮股 | ALL | LIST24032 |
| HK.LIST24037 | AI应用 | ALL | LIST24037 |
| HK.LIST24039 | 铜矿股 | ALL | LIST24039 |
| HK.LIST24055 | 存储概念 | ALL | LIST24055 |
| HK.LIST24056 | 追踪韩国ETF | ALL | LIST24056 |
| HK.LIST24062 | OpenClaw概念股 | ALL | LIST24062 |
| HK.LIST24125 | PCB概念 | ALL | LIST24125 |
| HK.LIST24126 | AI次新股 | ALL | LIST24126 |

#### `CONCEPT` (117)

| code | name | plate_type | plate_id |
|---|---|---|---|
| HK.LIST1110 | 阿里概念股 | CONCEPT | LIST1110 |
| HK.LIST1161 | 双十一 | CONCEPT | LIST1161 |
| HK.LIST1166 | 手机产业链 | CONCEPT | LIST1166 |
| HK.LIST1175 | 一带一路 | CONCEPT | LIST1175 |
| HK.LIST1176 | 5G概念 | CONCEPT | LIST1176 |
| HK.LIST1178 | 粤港澳大湾区 | CONCEPT | LIST1178 |
| HK.LIST1180 | 特斯拉概念股 | CONCEPT | LIST1180 |
| HK.LIST1181 | 啤酒 | CONCEPT | LIST1181 |
| HK.LIST1185 | 体育用品 | CONCEPT | LIST1185 |
| HK.LIST1186 | 稀土概念 | CONCEPT | LIST1186 |
| HK.LIST1190 | 腾讯概念 | CONCEPT | LIST1190 |
| HK.LIST1191 | 云办公 | CONCEPT | LIST1191 |
| HK.LIST1192 | SaaS概念 | CONCEPT | LIST1192 |
| HK.LIST1193 | 在线教育 | CONCEPT | LIST1193 |
| HK.LIST1196 | 汽车经销商 | CONCEPT | LIST1196 |
| HK.LIST1200 | 核电 | CONCEPT | LIST1200 |
| HK.LIST1202 | 化妆品 | CONCEPT | LIST1202 |
| HK.LIST1205 | 石油股 | CONCEPT | LIST1205 |
| HK.LIST1206 | 电讯设备 | CONCEPT | LIST1206 |
| HK.LIST1207 | 电力股 | CONCEPT | LIST1207 |
| HK.LIST1208 | 手游股 | CONCEPT | LIST1208 |
| HK.LIST1209 | 婴童用品股 | CONCEPT | LIST1209 |
| HK.LIST1210 | 百货业股 | CONCEPT | LIST1210 |
| HK.LIST1213 | 港口运输股 | CONCEPT | LIST1213 |
| HK.LIST1214 | 电信股 | CONCEPT | LIST1214 |
| HK.LIST1215 | 环保 | CONCEPT | LIST1215 |
| HK.LIST1216 | 煤炭股 | CONCEPT | LIST1216 |
| HK.LIST1217 | 综合车企股 | CONCEPT | LIST1217 |
| HK.LIST1218 | 电池 | CONCEPT | LIST1218 |
| HK.LIST1219 | 物流 | CONCEPT | LIST1219 |
| HK.LIST1220 | 内地物业管理股 | CONCEPT | LIST1220 |
| HK.LIST1221 | 农业股 | CONCEPT | LIST1221 |
| HK.LIST1222 | 黄金股 | CONCEPT | LIST1222 |
| HK.LIST1223 | 奢侈品品牌股 | CONCEPT | LIST1223 |
| HK.LIST1224 | 电力设备股 | CONCEPT | LIST1224 |
| HK.LIST1226 | 重型机械股 | CONCEPT | LIST1226 |
| HK.LIST1227 | 食品股 | CONCEPT | LIST1227 |
| HK.LIST1230 | 纸业股 | CONCEPT | LIST1230 |
| HK.LIST1231 | 水务股 | CONCEPT | LIST1231 |
| HK.LIST1232 | 奶制品股 | CONCEPT | LIST1232 |
| HK.LIST1233 | 光伏太阳能股 | CONCEPT | LIST1233 |
| HK.LIST1234 | 内房股 | CONCEPT | LIST1234 |
| HK.LIST1235 | 内地教育股 | CONCEPT | LIST1235 |
| HK.LIST1236 | 家电股 | CONCEPT | LIST1236 |
| HK.LIST1237 | 风电股 | CONCEPT | LIST1237 |
| HK.LIST1239 | 内银股 | CONCEPT | LIST1239 |
| HK.LIST1240 | 航空股 | CONCEPT | LIST1240 |
| HK.LIST1241 | 石油与天然气 | CONCEPT | LIST1241 |
| HK.LIST1242 | 建材水泥股 | CONCEPT | LIST1242 |
| HK.LIST1243 | 中资券商股 | CONCEPT | LIST1243 |
| HK.LIST1244 | 高铁基建股 | CONCEPT | LIST1244 |
| HK.LIST1245 | 燃气股 | CONCEPT | LIST1245 |
| HK.LIST1246 | 公路及铁路股 | CONCEPT | LIST1246 |
| HK.LIST1251 | 互联网医疗 | CONCEPT | LIST1251 |
| HK.LIST1252 | 香港本地银行股 | CONCEPT | LIST1252 |
| HK.LIST1254 | 生物医药B类股 | CONCEPT | LIST1254 |
| HK.LIST1261 | 医药外包概念 | CONCEPT | LIST1261 |
| HK.LIST1263 | 香港本地消费股 | CONCEPT | LIST1263 |
| HK.LIST1266 | 抖音概念股 | CONCEPT | LIST1266 |
| HK.LIST1267 | 烟草及电子烟股 | CONCEPT | LIST1267 |
| HK.LIST1279 | 影视股 | CONCEPT | LIST1279 |
| HK.LIST1287 | 蚂蚁金服概念 | CONCEPT | LIST1287 |
| HK.LIST1288 | 昨日强势股 | CONCEPT | LIST1288 |
| HK.LIST1289 | 云计算 | CONCEPT | LIST1289 |
| HK.LIST1290 | 次新股 | CONCEPT | LIST1290 |
| HK.LIST1304 | 回港中概股 | CONCEPT | LIST1304 |
| HK.LIST1305 | 北水核心资产 | CONCEPT | LIST1305 |
| HK.LIST1306 | 短视频概念股 | CONCEPT | LIST1306 |
| HK.LIST1312 | 有色金属 | CONCEPT | LIST1312 |
| HK.LIST1313 | 医美概念股 | CONCEPT | LIST1313 |
| HK.LIST1314 | 碳中和概念股 | CONCEPT | LIST1314 |
| HK.LIST1319 | 民办高教 | CONCEPT | LIST1319 |
| HK.LIST1320 | K12教育 | CONCEPT | LIST1320 |
| HK.LIST1321 | 三胎概念 | CONCEPT | LIST1321 |
| HK.LIST1326 | 中医药概念 | CONCEPT | LIST1326 |
| HK.LIST1328 | 元宇宙概念 | CONCEPT | LIST1328 |
| HK.LIST1329 | 绿电概念 | CONCEPT | LIST1329 |
| HK.LIST1331 | 小米概念 | CONCEPT | LIST1331 |
| HK.LIST1334 | 锂电池 | CONCEPT | LIST1334 |
| HK.LIST1335 | 养老概念 | CONCEPT | LIST1335 |
| HK.LIST1336 | 职业教育 | CONCEPT | LIST1336 |
| HK.LIST1342 | 氢能源概念股 | CONCEPT | LIST1342 |
| HK.LIST1344 | 高股息概念 | CONCEPT | LIST1344 |
| HK.LIST1348 | 中特估-国企 | CONCEPT | LIST1348 |
| HK.LIST1351 | 虚拟现实 | CONCEPT | LIST1351 |
| HK.LIST1353 | 室温超导概念 | CONCEPT | LIST1353 |
| HK.LIST1362 | 以巴冲突 | CONCEPT | LIST1362 |
| HK.LIST1991 | OLED概念 | CONCEPT | LIST1991 |
| HK.LIST1992 | 工业大麻 | CONCEPT | LIST1992 |
| HK.LIST1994 | 香港零售股 | CONCEPT | LIST1994 |
| HK.LIST1996 | 猪肉概念 | CONCEPT | LIST1996 |
| HK.LIST1998 | 节假日概念股 | CONCEPT | LIST1998 |
| HK.LIST1999 | 殡葬概念 | CONCEPT | LIST1999 |
| HK.LIST20074 | 加密货币概念股 | CONCEPT | LIST20074 |
| HK.LIST22886 | 明星科网股 | CONCEPT | LIST22886 |
| HK.LIST22910 | 新能源车企 | CONCEPT | LIST22910 |
| HK.LIST22911 | 生物医药 | CONCEPT | LIST22911 |
| HK.LIST22912 | 芯片股 | CONCEPT | LIST22912 |
| HK.LIST22913 | 博彩股 | CONCEPT | LIST22913 |
| HK.LIST22928 | 美国降息利好概念 | CONCEPT | LIST22928 |
| HK.LIST23586 | 人工智能 | CONCEPT | LIST23586 |
| HK.LIST23589 | 智能驾驶概念股 | CONCEPT | LIST23589 |
| HK.LIST23593 | 机器人概念股 | CONCEPT | LIST23593 |
| HK.LIST23598 | AI医疗概念股 | CONCEPT | LIST23598 |
| HK.LIST23674 | 稳定币概念 | CONCEPT | LIST23674 |
| HK.LIST23675 | 创新药概念 | CONCEPT | LIST23675 |
| HK.LIST23676 | 新消费概念 | CONCEPT | LIST23676 |
| HK.LIST23699 | 脑机接口概念 | CONCEPT | LIST23699 |
| HK.LIST24024 | 光通信 | CONCEPT | LIST24024 |
| HK.LIST24031 | 商业航天 | CONCEPT | LIST24031 |
| HK.LIST24032 | 茶饮股 | CONCEPT | LIST24032 |
| HK.LIST24037 | AI应用 | CONCEPT | LIST24037 |
| HK.LIST24039 | 铜矿股 | CONCEPT | LIST24039 |
| HK.LIST24055 | 存储概念 | CONCEPT | LIST24055 |
| HK.LIST24062 | OpenClaw概念股 | CONCEPT | LIST24062 |
| HK.LIST24125 | PCB概念 | CONCEPT | LIST24125 |
| HK.LIST24126 | AI次新股 | CONCEPT | LIST24126 |

#### `INDUSTRY` (111)

| code | name | plate_type | plate_id |
|---|---|---|---|
| HK.LIST1001 | 乳制品 | INDUSTRY | LIST1001 |
| HK.LIST1002 | 采购及供应链管理 | INDUSTRY | LIST1002 |
| HK.LIST1003 | 保险 | INDUSTRY | LIST1003 |
| HK.LIST1004 | 信贷 | INDUSTRY | LIST1004 |
| HK.LIST1005 | 公共运输 | INDUSTRY | LIST1005 |
| HK.LIST1006 | 其他金属及矿物 | INDUSTRY | LIST1006 |
| HK.LIST1007 | 其他金融 | INDUSTRY | LIST1007 |
| HK.LIST1008 | 农产品 | INDUSTRY | LIST1008 |
| HK.LIST1009 | 出版 | INDUSTRY | LIST1009 |
| HK.LIST1010 | 包装食品 | INDUSTRY | LIST1010 |
| HK.LIST1011 | 化肥及农用化合物 | INDUSTRY | LIST1011 |
| HK.LIST1012 | 医疗设备及用品 | INDUSTRY | LIST1012 |
| HK.LIST1013 | 半导体 | INDUSTRY | LIST1013 |
| HK.LIST1014 | 卫星及无线通讯 | INDUSTRY | LIST1014 |
| HK.LIST1015 | 印刷及包装 | INDUSTRY | LIST1015 |
| HK.LIST1016 | 非传统/可再生能源 | INDUSTRY | LIST1016 |
| HK.LIST1017 | 商业用车及货车 | INDUSTRY | LIST1017 |
| HK.LIST1019 | 地产发展商 | INDUSTRY | LIST1019 |
| HK.LIST1020 | 地产投资 | INDUSTRY | LIST1020 |
| HK.LIST1021 | 家具 | INDUSTRY | LIST1021 |
| HK.LIST1022 | 家庭电器 | INDUSTRY | LIST1022 |
| HK.LIST1025 | 工业零件及器材 | INDUSTRY | LIST1025 |
| HK.LIST1026 | 广告及宣传 | INDUSTRY | LIST1026 |
| HK.LIST1027 | 广播 | INDUSTRY | LIST1027 |
| HK.LIST1028 | 建筑材料 | INDUSTRY | LIST1028 |
| HK.LIST1029 | 影视娱乐 | INDUSTRY | LIST1029 |
| HK.LIST1030 | 投资及资产管理 | INDUSTRY | LIST1030 |
| HK.LIST1031 | 其他支援服务 | INDUSTRY | LIST1031 |
| HK.LIST1032 | 消闲及文娱设施 | INDUSTRY | LIST1032 |
| HK.LIST1033 | 新能源物料 | INDUSTRY | LIST1033 |
| HK.LIST1034 | 旅游及观光 | INDUSTRY | LIST1034 |
| HK.LIST1035 | 纺织品及布料 | INDUSTRY | LIST1035 |
| HK.LIST1037 | 林业及木材 | INDUSTRY | LIST1037 |
| HK.LIST1039 | 水务 | INDUSTRY | LIST1039 |
| HK.LIST1040 | 汽车 | INDUSTRY | LIST1040 |
| HK.LIST1041 | 汽车零件 | INDUSTRY | LIST1041 |
| HK.LIST1042 | 油气生产商 | INDUSTRY | LIST1042 |
| HK.LIST1043 | 油气设备与服务 | INDUSTRY | LIST1043 |
| HK.LIST1044 | 煤炭 | INDUSTRY | LIST1044 |
| HK.LIST1045 | 燃气供应 | INDUSTRY | LIST1045 |
| HK.LIST1046 | 特殊化工用品 | INDUSTRY | LIST1046 |
| HK.LIST1047 | 玩具及消闲用品 | INDUSTRY | LIST1047 |
| HK.LIST1049 | 珠宝钟表 | INDUSTRY | LIST1049 |
| HK.LIST1050 | 生物技术 | INDUSTRY | LIST1050 |
| HK.LIST1051 | 常规电力 | INDUSTRY | LIST1051 |
| HK.LIST1052 | 消费电子产品 | INDUSTRY | LIST1052 |
| HK.LIST1053 | 电脑及周边器材 | INDUSTRY | LIST1053 |
| HK.LIST1054 | 电讯服务 | INDUSTRY | LIST1054 |
| HK.LIST1055 | 消费性电讯设备 | INDUSTRY | LIST1055 |
| HK.LIST1056 | 多元化零售商 | INDUSTRY | LIST1056 |
| HK.LIST1059 | 纸及纸制品 | INDUSTRY | LIST1059 |
| HK.LIST1061 | 综合企业 | INDUSTRY | LIST1061 |
| HK.LIST1062 | 个人护理 | INDUSTRY | LIST1062 |
| HK.LIST1063 | 航空航天与国防 | INDUSTRY | LIST1063 |
| HK.LIST1064 | 航空服务 | INDUSTRY | LIST1064 |
| HK.LIST1065 | 航空货运及物流 | INDUSTRY | LIST1065 |
| HK.LIST1066 | 航运及港口 | INDUSTRY | LIST1066 |
| HK.LIST1067 | 药品 | INDUSTRY | LIST1067 |
| HK.LIST1068 | 证券及经纪 | INDUSTRY | LIST1068 |
| HK.LIST1069 | 赌场及博彩 | INDUSTRY | LIST1069 |
| HK.LIST1070 | 超市及便利店 | INDUSTRY | LIST1070 |
| HK.LIST1071 | 酒店及度假村 | INDUSTRY | LIST1071 |
| HK.LIST1072 | 酒精饮料 | INDUSTRY | LIST1072 |
| HK.LIST1073 | 重型基建 | INDUSTRY | LIST1073 |
| HK.LIST1074 | 重型机械 | INDUSTRY | LIST1074 |
| HK.LIST1075 | 钢铁 | INDUSTRY | LIST1075 |
| HK.LIST1076 | 铁路及公路 | INDUSTRY | LIST1076 |
| HK.LIST1077 | 铜 | INDUSTRY | LIST1077 |
| HK.LIST1078 | 铝 | INDUSTRY | LIST1078 |
| HK.LIST1079 | 银行 | INDUSTRY | LIST1079 |
| HK.LIST1080 | 非酒精饮料 | INDUSTRY | LIST1080 |
| HK.LIST1082 | 食品添加剂 | INDUSTRY | LIST1082 |
| HK.LIST1083 | 餐饮 | INDUSTRY | LIST1083 |
| HK.LIST1084 | 黄金及贵金属 | INDUSTRY | LIST1084 |
| HK.LIST1086 | 医疗及医学美容服务 | INDUSTRY | LIST1086 |
| HK.LIST1089 | 地产代理 | INDUSTRY | LIST1089 |
| HK.LIST1090 | 物业服务及管理 | INDUSTRY | LIST1090 |
| HK.LIST1091 | 教育 | INDUSTRY | LIST1091 |
| HK.LIST1095 | 楼宇建造 | INDUSTRY | LIST1095 |
| HK.LIST1100 | 应用软件 | INDUSTRY | LIST1100 |
| HK.LIST1268 | 其他服饰配件 | INDUSTRY | LIST1268 |
| HK.LIST1269 | 汽车零售商 | INDUSTRY | LIST1269 |
| HK.LIST1270 | 服装零售商 | INDUSTRY | LIST1270 |
| HK.LIST1271 | 环保工程 | INDUSTRY | LIST1271 |
| HK.LIST1272 | 禽畜饲料 | INDUSTRY | LIST1272 |
| HK.LIST1273 | 禽畜肉类 | INDUSTRY | LIST1273 |
| HK.LIST1274 | 电子零件 | INDUSTRY | LIST1274 |
| HK.LIST1275 | 鞋类 | INDUSTRY | LIST1275 |
| HK.LIST1276 | 其他零售商 | INDUSTRY | LIST1276 |
| HK.LIST1277 | 服装 | INDUSTRY | LIST1277 |
| HK.LIST1278 | 家居装修零售商 | INDUSTRY | LIST1278 |
| HK.LIST1284 | 中医药 | INDUSTRY | LIST1284 |
| HK.LIST1311 | 房地产投资信托 | INDUSTRY | LIST1311 |
| HK.LIST1354 | 能源储存装置 | INDUSTRY | LIST1354 |
| HK.LIST1355 | 公路运输 | INDUSTRY | LIST1355 |
| HK.LIST1356 | 烟草 | INDUSTRY | LIST1356 |
| HK.LIST1357 | 药品分销 | INDUSTRY | LIST1357 |
| HK.LIST1358 | 核能 | INDUSTRY | LIST1358 |
| HK.LIST1359 | 游戏软件 | INDUSTRY | LIST1359 |
| HK.LIST1360 | 半导体设备与材料 | INDUSTRY | LIST1360 |
| HK.LIST23360 | 互动媒体及服务 | INDUSTRY | LIST23360 |
| HK.LIST23361 | 线上零售商 | INDUSTRY | LIST23361 |
| HK.LIST23362 | 支付服务 | INDUSTRY | LIST23362 |
| HK.LIST23363 | 数码解决方案服务 | INDUSTRY | LIST23363 |
| HK.LIST23364 | 互联网服务及基础设施 | INDUSTRY | LIST23364 |
| HK.LIST23846 | 轨道与列车设备 | INDUSTRY | LIST23846 |
| HK.LIST23847 | 摩托车及其他 | INDUSTRY | LIST23847 |
| HK.LIST23848 | 家居消耗品 | INDUSTRY | LIST23848 |
| HK.LIST23849 | 膳食补充品 | INDUSTRY | LIST23849 |
| HK.LIST23850 | 护肤与化妆品 | INDUSTRY | LIST23850 |
| HK.LIST23851 | 电讯网路基建设施 | INDUSTRY | LIST23851 |

#### `REGION` (0)

| code | name | plate_type | plate_id |
|---|---|---|---|

## Market `SH`

### App-Style Screener Categories

#### 行情 (`market_quote`)

Use market/plate to constrain the universe, then quote and accumulated fields to find liquid, tradable, strong or weak candidates.

- **交易所/市场**: `market_selector`
  - Supported values: `US, HK, SH, SZ`
  - LLM hint: Select the listing market. This is not a StockField.
- **所属行业/概念/板块**: `plate_selector`
  - LLM hint: Probe theme words as plate_keywords. If a useful plate_code is found, stock_filter can filter within that plate_code.
- **价格/52周位置/量比/委比/每手价格**: `stock_filter`
  - Fields: `CUR_PRICE, CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO, CUR_PRICE_TO_LOWEST52_WEEKS_RATIO, HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO, LOW_PRICE_TO_LOWEST52_WEEKS_RATIO, VOLUME_RATIO, BID_ASK_RATIO, LOT_PRICE`
  - LLM hint: Use these for tradability, momentum context, and avoiding illiquid tails.
- **涨跌幅/振幅/成交量/成交额/换手率**: `stock_filter`
  - Fields: `CHANGE_RATE, AMPLITUDE, VOLUME, TURNOVER, TURNOVER_RATE`
  - LLM hint: Use days plus sort/filter bounds to discover strong momentum or high liquidity names, e.g. TURNOVER DESCEND or CHANGE_RATE over 20/60 days.

#### 估值 (`valuation`)

Use valuation as a secondary screen after thematic relevance; do not let cheapness replace theme fit.

- **市值/流通市值/股本**: `stock_filter`
  - Fields: `MARKET_VAL, FLOAT_MARKET_VAL, TOTAL_SHARE, FLOAT_SHARE`
  - LLM hint: Use market cap to separate mega-cap anchors from small high-beta satellites.
- **市盈率（静态/TTM）/市净率/市销率/市现率**: `stock_filter`
  - Fields: `PE_ANNUAL, PE_TTM, PB_RATE, PS_TTM, PCF_TTM`
  - LLM hint: Useful for valuation risk flags and relative comparison, not theme discovery alone.
- **估值分位/行业估值分位**: `derived_or_future_adapter`
  - Alternate source: `derive from peer universe snapshots/financials; not exposed as current get_stock_filter StockField`
  - LLM hint: Do not pass valuation percentile as stock_filter_specs. Mark it as desired downstream enrichment if needed.

#### 分红 (`dividend`)

Dividend data is enrichment, not a get_stock_filter field in this SDK.

- **TTM 分红/股息率**: `non_stock_filter`
  - Alternate source: `get_market_snapshot fields dividend_ttm/dividend_ratio_ttm`
  - LLM hint: Do not create stock_filter_specs for dividends; request quote snapshot enrichment later.

#### 技术 (`technical`)

Use technical filters to discover trend candidates or confirm that a theme is active; avoid using them as the only reason a name belongs to a theme.

- **指标解读**: `derived_interpretation`
  - Alternate source: `derive from pattern/custom_indicator StockFields and K-line enrichment`
  - LLM hint: The app-style interpretation layer is not a single OpenAPI enum. Use the specific MA/EMA/KDJ/RSI/MACD/BOLL fields below.
- **MA/EMA 均线形态**: `stock_filter`
  - Fields: `MA_ALIGNMENT_LONG, MA_ALIGNMENT_SHORT, EMA_ALIGNMENT_LONG, EMA_ALIGNMENT_SHORT`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use for trend/momentum probes, e.g. MA_ALIGNMENT_LONG on K_DAY.
- **RSI/KDJ/MACD/BOLL 形态**: `stock_filter`
  - Fields: `RSI_GOLD_CROSS_LOW, RSI_DEATH_CROSS_HIGH, RSI_TOP_DIVERGENCE, RSI_BOTTOM_DIVERGENCE, KDJ_GOLD_CROSS_LOW, KDJ_DEATH_CROSS_HIGH, KDJ_TOP_DIVERGENCE, KDJ_BOTTOM_DIVERGENCE, MACD_GOLD_CROSS_LOW, MACD_DEATH_CROSS_HIGH, MACD_TOP_DIVERGENCE, MACD_BOTTOM_DIVERGENCE, BOLL_BREAK_UPPER, BOLL_BREAK_LOWER, BOLL_CROSS_MIDDLE_UP, BOLL_CROSS_MIDDLE_DOWN`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use for oscillator and Bollinger-band signals when discovering active candidates.
- **MA/EMA/KDJ/RSI/MACD/BOLL 自定义指标比较**: `stock_filter`
  - Fields: `PRICE, MA5, MA10, MA20, MA30, MA60, MA120, MA250, RSI, EMA5, EMA10, EMA20, EMA30, EMA60, EMA120, EMA250, VALUE, MA, EMA, KDJ_K, KDJ_D, KDJ_J, MACD_DIFF, MACD_DEA, MACD, BOLL_UPPER, BOLL_MIDDLER, BOLL_LOWER`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use only when a simple pattern field is insufficient.

#### 财务 (`financial`)

Use fundamentals to filter quality and growth after the theme map is drafted.

- **利润/收入/利润率/ROE/ROIC/现金流/资产负债/增长率/EPS**: `stock_filter`
  - Fields: `NET_PROFIT, NET_PROFIX_GROWTH, SUM_OF_BUSINESS, SUM_OF_BUSINESS_GROWTH, NET_PROFIT_RATE, GROSS_PROFIT_RATE, DEBT_ASSET_RATE, RETURN_ON_EQUITY_RATE, ROIC, ROA_TTM, EBIT_TTM, EBITDA, OPERATING_MARGIN_TTM, EBIT_MARGIN, EBITDA_MARGIN, FINANCIAL_COST_RATE, OPERATING_PROFIT_TTM, SHAREHOLDER_NET_PROFIT_TTM, NET_PROFIT_CASH_COVER_TTM, CURRENT_RATIO, QUICK_RATIO, CURRENT_ASSET_RATIO, CURRENT_DEBT_RATIO, EQUITY_MULTIPLIER, PROPERTY_RATIO, CASH_AND_CASH_EQUIVALENTS, TOTAL_ASSET_TURNOVER, FIXED_ASSET_TURNOVER, INVENTORY_TURNOVER, OPERATING_CASH_FLOW_TTM, ACCOUNTS_RECEIVABLE, EBIT_GROWTH_RATE, OPERATING_PROFIT_GROWTH_RATE, TOTAL_ASSETS_GROWTH_RATE, PROFIT_TO_SHAREHOLDERS_GROWTH_RATE, PROFIT_BEFORE_TAX_GROWTH_RATE, EPS_GROWTH_RATE, ROE_GROWTH_RATE, ROIC_GROWTH_RATE, NOCF_GROWTH_RATE, NOCF_PER_SHARE_GROWTH_RATE, OPERATING_REVENUE_CASH_COVER, OPERATING_PROFIT_TO_TOTAL_PROFIT, BASIC_EPS, DILUTED_EPS, NOCF_PER_SHARE`
  - LLM hint: Useful probes: revenue growth, gross margin, ROE/ROIC, operating cash flow, debt ratio. Exact numbers still need downstream SEC/fundamental validation.

#### 分析 (`analysis`)

Analyst ratings/revisions are not exposed by get_stock_filter here.

- **评级/目标价/一致预期/盈利修正**: `external_or_future_adapter`
  - Alternate source: `analyst_estimate_revision_adapter`
  - LLM hint: Do not invent these facts. Mark as needed evidence if useful.

#### 期权 (`options`)

Options are not stock_filter fields. Use option-chain/surface enrichment after candidate discovery.

- **期权活跃度/IV/到期日/Put-Call context**: `option_chain_enrichment`
  - Alternate source: `get_option_expiration_date + get_option_chain + option snapshots`
  - LLM hint: Use later to assess tradability and income-strategy readiness, not as stock_filter_specs.

### OpenAPI Stock Filter Fields

#### `simple` (21)

`STOCK_CODE`, `STOCK_NAME`, `CUR_PRICE`, `CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO`, `CUR_PRICE_TO_LOWEST52_WEEKS_RATIO`, `HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO`, `LOW_PRICE_TO_LOWEST52_WEEKS_RATIO`, `VOLUME_RATIO`, `BID_ASK_RATIO`, `LOT_PRICE`, `MARKET_VAL`, `PE_ANNUAL`, `PE_TTM`, `PB_RATE`, `CHANGE_RATE_5MIN`, `CHANGE_RATE_BEGIN_YEAR`, `PS_TTM`, `PCF_TTM`, `TOTAL_SHARE`, `FLOAT_SHARE`, `FLOAT_MARKET_VAL`

#### `accumulate` (5)

`CHANGE_RATE`, `AMPLITUDE`, `VOLUME`, `TURNOVER`, `TURNOVER_RATE`

#### `financial` (46)

`NET_PROFIT`, `NET_PROFIX_GROWTH`, `SUM_OF_BUSINESS`, `SUM_OF_BUSINESS_GROWTH`, `NET_PROFIT_RATE`, `GROSS_PROFIT_RATE`, `DEBT_ASSET_RATE`, `RETURN_ON_EQUITY_RATE`, `ROIC`, `ROA_TTM`, `EBIT_TTM`, `EBITDA`, `OPERATING_MARGIN_TTM`, `EBIT_MARGIN`, `EBITDA_MARGIN`, `FINANCIAL_COST_RATE`, `OPERATING_PROFIT_TTM`, `SHAREHOLDER_NET_PROFIT_TTM`, `NET_PROFIT_CASH_COVER_TTM`, `CURRENT_RATIO`, `QUICK_RATIO`, `CURRENT_ASSET_RATIO`, `CURRENT_DEBT_RATIO`, `EQUITY_MULTIPLIER`, `PROPERTY_RATIO`, `CASH_AND_CASH_EQUIVALENTS`, `TOTAL_ASSET_TURNOVER`, `FIXED_ASSET_TURNOVER`, `INVENTORY_TURNOVER`, `OPERATING_CASH_FLOW_TTM`, `ACCOUNTS_RECEIVABLE`, `EBIT_GROWTH_RATE`, `OPERATING_PROFIT_GROWTH_RATE`, `TOTAL_ASSETS_GROWTH_RATE`, `PROFIT_TO_SHAREHOLDERS_GROWTH_RATE`, `PROFIT_BEFORE_TAX_GROWTH_RATE`, `EPS_GROWTH_RATE`, `ROE_GROWTH_RATE`, `ROIC_GROWTH_RATE`, `NOCF_GROWTH_RATE`, `NOCF_PER_SHARE_GROWTH_RATE`, `OPERATING_REVENUE_CASH_COVER`, `OPERATING_PROFIT_TO_TOTAL_PROFIT`, `BASIC_EPS`, `DILUTED_EPS`, `NOCF_PER_SHARE`

#### `pattern` (20)

`MA_ALIGNMENT_LONG`, `MA_ALIGNMENT_SHORT`, `EMA_ALIGNMENT_LONG`, `EMA_ALIGNMENT_SHORT`, `RSI_GOLD_CROSS_LOW`, `RSI_DEATH_CROSS_HIGH`, `RSI_TOP_DIVERGENCE`, `RSI_BOTTOM_DIVERGENCE`, `KDJ_GOLD_CROSS_LOW`, `KDJ_DEATH_CROSS_HIGH`, `KDJ_TOP_DIVERGENCE`, `KDJ_BOTTOM_DIVERGENCE`, `MACD_GOLD_CROSS_LOW`, `MACD_DEATH_CROSS_HIGH`, `MACD_TOP_DIVERGENCE`, `MACD_BOTTOM_DIVERGENCE`, `BOLL_BREAK_UPPER`, `BOLL_BREAK_LOWER`, `BOLL_CROSS_MIDDLE_UP`, `BOLL_CROSS_MIDDLE_DOWN`

#### `custom_indicator` (28)

`PRICE`, `MA5`, `MA10`, `MA20`, `MA30`, `MA60`, `MA120`, `MA250`, `RSI`, `EMA5`, `EMA10`, `EMA20`, `EMA30`, `EMA60`, `EMA120`, `EMA250`, `VALUE`, `MA`, `EMA`, `KDJ_K`, `KDJ_D`, `KDJ_J`, `MACD_DIFF`, `MACD_DEA`, `MACD`, `BOLL_UPPER`, `BOLL_MIDDLER`, `BOLL_LOWER`

- `sort_dir`: `ASCEND`, `DESCEND`
- `financial_quarter`: `ANNUAL`, `FIRST_QUARTER`, `INTERIM`, `THIRD_QUARTER`, `MOST_RECENT_QUARTER`
- `supported_pattern_ktype`: `K_60M`, `K_DAY`, `K_WEEK`, `K_MON`
- `relative_position`: `MORE`, `LESS`, `CROSS_UP`, `CROSS_DOWN`

### Futu Plate Choices

#### `ALL` (639)

| code | name | plate_type | plate_id |
|---|---|---|---|
| SH.LIST0001 | 白色家电 | ALL | LIST0001 |
| SH.LIST0002 | 半导体 | ALL | LIST0002 |
| SH.LIST0003 | 包装印刷 | ALL | LIST0003 |
| SH.LIST0004 | 物流 | ALL | LIST0004 |
| SH.LIST0005 | 航海装备Ⅱ | ALL | LIST0005 |
| SH.LIST0006 | 电力 | ALL | LIST0006 |
| SH.LIST0007 | 玻璃玻纤 | ALL | LIST0007 |
| SH.LIST0008 | 多元金融 | ALL | LIST0008 |
| SH.LIST0011 | 养殖业 | ALL | LIST0011 |
| SH.LIST0013 | 纺织制造 | ALL | LIST0013 |
| SH.LIST0014 | 地面兵装Ⅱ | ALL | LIST0014 |
| SH.LIST0017 | 电机Ⅱ | ALL | LIST0017 |
| SH.LIST0020 | 其他电源设备Ⅱ | ALL | LIST0020 |
| SH.LIST0022 | 消费电子 | ALL | LIST0022 |
| SH.LIST0025 | 动物保健Ⅱ | ALL | LIST0025 |
| SH.LIST0026 | 房地产开发 | ALL | LIST0026 |
| SH.LIST0027 | 化学纤维 | ALL | LIST0027 |
| SH.LIST0028 | 房屋建设Ⅱ | ALL | LIST0028 |
| SH.LIST0031 | 工业金属 | ALL | LIST0031 |
| SH.LIST0032 | 光学光电子 | ALL | LIST0032 |
| SH.LIST0033 | 航空装备Ⅱ | ALL | LIST0033 |
| SH.LIST0035 | 航天装备Ⅱ | ALL | LIST0035 |
| SH.LIST0039 | 化学原料 | ALL | LIST0039 |
| SH.LIST0040 | 贸易Ⅱ | ALL | LIST0040 |
| SH.LIST0041 | 化学制品 | ALL | LIST0041 |
| SH.LIST0042 | 化学制药 | ALL | LIST0042 |
| SH.LIST0044 | 贵金属 | ALL | LIST0044 |
| SH.LIST0045 | 基础建设 | ALL | LIST0045 |
| SH.LIST0047 | 证券Ⅱ | ALL | LIST0047 |
| SH.LIST0048 | 汽车零部件 | ALL | LIST0048 |
| SH.LIST0049 | 计算机设备 | ALL | LIST0049 |
| SH.LIST0051 | 家居用品 | ALL | LIST0051 |
| SH.LIST0052 | 金属新材料 | ALL | LIST0052 |
| SH.LIST0053 | 饲料 | ALL | LIST0053 |
| SH.LIST0055 | 食品加工 | ALL | LIST0055 |
| SH.LIST0057 | 水泥 | ALL | LIST0057 |
| SH.LIST0058 | 塑料 | ALL | LIST0058 |
| SH.LIST0059 | 生物制品 | ALL | LIST0059 |
| SH.LIST0061 | 通信设备 | ALL | LIST0061 |
| SH.LIST0063 | 林业Ⅱ | ALL | LIST0063 |
| SH.LIST0065 | 橡胶 | ALL | LIST0065 |
| SH.LIST0066 | 农产品加工 | ALL | LIST0066 |
| SH.LIST0068 | 医疗服务 | ALL | LIST0068 |
| SH.LIST0071 | 医药商业 | ALL | LIST0071 |
| SH.LIST0073 | 综合Ⅱ | ALL | LIST0073 |
| SH.LIST0074 | 中药Ⅱ | ALL | LIST0074 |
| SH.LIST0075 | 专用设备 | ALL | LIST0075 |
| SH.LIST0076 | 造纸 | ALL | LIST0076 |
| SH.LIST0077 | 保险Ⅱ | ALL | LIST0077 |
| SH.LIST0078 | 其他电子Ⅱ | ALL | LIST0078 |
| SH.LIST0079 | 装修建材 | ALL | LIST0079 |
| SH.LIST0080 | 摩托车及其他 | ALL | LIST0080 |
| SH.LIST0082 | 汽车服务 | ALL | LIST0082 |
| SH.LIST0083 | 燃气Ⅱ | ALL | LIST0083 |
| SH.LIST0086 | 黑色家电 | ALL | LIST0086 |
| SH.LIST0088 | 通信服务 | ALL | LIST0088 |
| SH.LIST0089 | 通用设备 | ALL | LIST0089 |
| SH.LIST0090 | 小金属 | ALL | LIST0090 |
| SH.LIST0091 | 一般零售 | ALL | LIST0091 |
| SH.LIST0092 | 医疗器械 | ALL | LIST0092 |
| SH.LIST0095 | 渔业 | ALL | LIST0095 |
| SH.LIST0096 | 元件 | ALL | LIST0096 |
| SH.LIST0099 | 轨交设备Ⅱ | ALL | LIST0099 |
| SH.LIST0103 | 博彩概念 | ALL | LIST0103 |
| SH.LIST0104 | 保障房 | ALL | LIST0104 |
| SH.LIST0109 | 储能概念 | ALL | LIST0109 |
| SH.LIST0111 | 成渝特区 | ALL | LIST0111 |
| SH.LIST0113 | 财政扶持 | ALL | LIST0113 |
| SH.LIST0115 | 参股金融 | ALL | LIST0115 |
| SH.LIST0118 | 低价 | ALL | LIST0118 |
| SH.LIST0119 | 国资整合 | ALL | LIST0119 |
| SH.LIST0120 | 地理信息 | ALL | LIST0120 |
| SH.LIST0123 | 低市净率 | ALL | LIST0123 |
| SH.LIST0124 | 大订单 | ALL | LIST0124 |
| SH.LIST0125 | 定向增发 | ALL | LIST0125 |
| SH.LIST0128 | 大盘 | ALL | LIST0128 |
| SH.LIST0129 | 电子商务 | ALL | LIST0129 |
| SH.LIST0131 | 风险警示板 | ALL | LIST0131 |
| SH.LIST0132 | 风能 | ALL | LIST0132 |
| SH.LIST0133 | 高价 | ALL | LIST0133 |
| SH.LIST0138 | 公募增发 | ALL | LIST0138 |
| SH.LIST0140 | 国际板 | ALL | LIST0140 |
| SH.LIST0142 | 股权激励 | ALL | LIST0142 |
| SH.LIST0143 | 国家安全 | ALL | LIST0143 |
| SH.LIST0144 | 股权投资 | ALL | LIST0144 |
| SH.LIST0145 | 航天军工 | ALL | LIST0145 |
| SH.LIST0149 | 沪台通 | ALL | LIST0149 |
| SH.LIST0150 | 海洋工程 | ALL | LIST0150 |
| SH.LIST0151 | 海西概念 | ALL | LIST0151 |
| SH.LIST0153 | 核能 | ALL | LIST0153 |
| SH.LIST0154 | H股 | ALL | LIST0154 |
| SH.LIST0156 | 黄金股 | ALL | LIST0156 |
| SH.LIST0158 | IPO受益 | ALL | LIST0158 |
| SH.LIST0160 | 基金增仓 | ALL | LIST0160 |
| SH.LIST0161 | 金融IC卡 | ALL | LIST0161 |
| SH.LIST0163 | 基金新增 | ALL | LIST0163 |
| SH.LIST0169 | 抗流感 | ALL | LIST0169 |
| SH.LIST0177 | 明星基金 | ALL | LIST0177 |
| SH.LIST0178 | 农村金融 | ALL | LIST0178 |
| SH.LIST0179 | 内陆自贸区 | ALL | LIST0179 |
| SH.LIST0180 | 农业机械 | ALL | LIST0180 |
| SH.LIST0181 | 农垦概念 | ALL | LIST0181 |
| SH.LIST0184 | 券商重仓 | ALL | LIST0184 |
| SH.LIST0185 | 去IOE概念 | ALL | LIST0185 |
| SH.LIST0186 | QFII持股 | ALL | LIST0186 |
| SH.LIST0190 | 深证100 | ALL | LIST0190 |
| SH.LIST0191 | 数字电视 | ALL | LIST0191 |
| SH.LIST0192 | 生物育种 | ALL | LIST0192 |
| SH.LIST0193 | 社保重仓 | ALL | LIST0193 |
| SH.LIST0194 | 生猪概念 | ALL | LIST0194 |
| SH.LIST0195 | 深成500 | ALL | LIST0195 |
| SH.LIST0198 | 三沙概念 | ALL | LIST0198 |
| SH.LIST0199 | 涉矿概念 | ALL | LIST0199 |
| SH.LIST0200 | 送转 | ALL | LIST0200 |
| SH.LIST0201 | 水利 | ALL | LIST0201 |
| SH.LIST0202 | 三网融合 | ALL | LIST0202 |
| SH.LIST0206 | 上证180 | ALL | LIST0206 |
| SH.LIST0210 | 水域改革 | ALL | LIST0210 |
| SH.LIST0211 | 上证380 | ALL | LIST0211 |
| SH.LIST0212 | 手机游戏 | ALL | LIST0212 |
| SH.LIST0216 | TMT概念 | ALL | LIST0216 |
| SH.LIST0217 | 透析 | ALL | LIST0217 |
| SH.LIST0218 | 特种药 | ALL | LIST0218 |
| SH.LIST0221 | 特色化工 | ALL | LIST0221 |
| SH.LIST0223 | 土地改革 | ALL | LIST0223 |
| SH.LIST0228 | 王亚伟概念 | ALL | LIST0228 |
| SH.LIST0231 | 小盘 | ALL | LIST0231 |
| SH.LIST0232 | 循环经济 | ALL | LIST0232 |
| SH.LIST0233 | 新三板 | ALL | LIST0233 |
| SH.LIST0234 | 新股未分红 | ALL | LIST0234 |
| SH.LIST0235 | 新股改革 | ALL | LIST0235 |
| SH.LIST0238 | 新媒体 | ALL | LIST0238 |
| SH.LIST0241 | 西部开发 | ALL | LIST0241 |
| SH.LIST0246 | 移动软件 | ALL | LIST0246 |
| SH.LIST0249 | 婴童概念 | ALL | LIST0249 |
| SH.LIST0250 | 预盈预增 | ALL | LIST0250 |
| SH.LIST0251 | 预亏预减 | ALL | LIST0251 |
| SH.LIST0254 | 整体上市 | ALL | LIST0254 |
| SH.LIST0255 | 中俄自贸区 | ALL | LIST0255 |
| SH.LIST0257 | 中盘 | ALL | LIST0257 |
| SH.LIST0258 | 再融资 | ALL | LIST0258 |
| SH.LIST0259 | 智能手机 | ALL | LIST0259 |
| SH.LIST0260 | 智能汽车 | ALL | LIST0260 |
| SH.LIST0262 | 重组并购 | ALL | LIST0262 |
| SH.LIST0263 | 中证500 | ALL | LIST0263 |
| SH.LIST0266 | 智能制造 | ALL | LIST0266 |
| SH.LIST0268 | 注资承诺 | ALL | LIST0268 |
| SH.LIST0269 | 舟山自贸区 | ALL | LIST0269 |
| SH.LIST0271 | 中证100 | ALL | LIST0271 |
| SH.LIST0272 | 中日韩自贸 | ALL | LIST0272 |
| SH.LIST0273 | 增持回购 | ALL | LIST0273 |
| SH.LIST0279 | 3D影视 | ALL | LIST0279 |
| SH.LIST0285 | 双11 | ALL | LIST0285 |
| SH.LIST0301 | 汽车电子概念 | ALL | LIST0301 |
| SH.LIST0302 | 高送转 | ALL | LIST0302 |
| SH.LIST0303 | 上海自贸区 | ALL | LIST0303 |
| SH.LIST0304 | 电力物联网 | ALL | LIST0304 |
| SH.LIST0305 | 养鸡 | ALL | LIST0305 |
| SH.LIST0306 | 医疗器械概念 | ALL | LIST0306 |
| SH.LIST0307 | 燃料电池概念 | ALL | LIST0307 |
| SH.LIST0308 | 乳业 | ALL | LIST0308 |
| SH.LIST0309 | 黄金概念 | ALL | LIST0309 |
| SH.LIST0310 | 智能音箱 | ALL | LIST0310 |
| SH.LIST0311 | 微信小程序 | ALL | LIST0311 |
| SH.LIST0312 | 新疆 | ALL | LIST0312 |
| SH.LIST0313 | 福建 | ALL | LIST0313 |
| SH.LIST0314 | 健康中国 | ALL | LIST0314 |
| SH.LIST0315 | 雄安新区 | ALL | LIST0315 |
| SH.LIST0316 | 边缘计算 | ALL | LIST0316 |
| SH.LIST0317 | 太阳能概念 | ALL | LIST0317 |
| SH.LIST0318 | 养老概念 | ALL | LIST0318 |
| SH.LIST0319 | 短视频 | ALL | LIST0319 |
| SH.LIST0320 | 国产软件 | ALL | LIST0320 |
| SH.LIST0321 | 煤化工概念 | ALL | LIST0321 |
| SH.LIST0322 | 天然气 | ALL | LIST0322 |
| SH.LIST0323 | 网约车 | ALL | LIST0323 |
| SH.LIST0324 | 海工装备 | ALL | LIST0324 |
| SH.LIST0325 | 阿里概念 | ALL | LIST0325 |
| SH.LIST0326 | 长三角一体化 | ALL | LIST0326 |
| SH.LIST0327 | 大数据 | ALL | LIST0327 |
| SH.LIST0329 | 铁路基建 | ALL | LIST0329 |
| SH.LIST0330 | 冷链物流 | ALL | LIST0330 |
| SH.LIST0331 | 军民融合 | ALL | LIST0331 |
| SH.LIST0332 | 沪伦通 | ALL | LIST0332 |
| SH.LIST0333 | OLED | ALL | LIST0333 |
| SH.LIST0334 | 手游概念 | ALL | LIST0334 |
| SH.LIST0335 | 人脸识别 | ALL | LIST0335 |
| SH.LIST0336 | 物联网 | ALL | LIST0336 |
| SH.LIST0337 | 知识产权保护 | ALL | LIST0337 |
| SH.LIST0338 | 融资融券 | ALL | LIST0338 |
| SH.LIST0339 | 农机 | ALL | LIST0339 |
| SH.LIST0340 | 河南 | ALL | LIST0340 |
| SH.LIST0341 | 贵州 | ALL | LIST0341 |
| SH.LIST0342 | 高铁 | ALL | LIST0342 |
| SH.LIST0343 | 超级真菌 | ALL | LIST0343 |
| SH.LIST0344 | 国企改革 | ALL | LIST0344 |
| SH.LIST0345 | 智能医疗 | ALL | LIST0345 |
| SH.LIST0346 | 乡村振兴 | ALL | LIST0346 |
| SH.LIST0347 | 移动支付 | ALL | LIST0347 |
| SH.LIST0348 | 债转股 | ALL | LIST0348 |
| SH.LIST0349 | 中字头 | ALL | LIST0349 |
| SH.LIST0350 | 啤酒概念 | ALL | LIST0350 |
| SH.LIST0351 | 上海国资改革 | ALL | LIST0351 |
| SH.LIST0352 | 智能穿戴 | ALL | LIST0352 |
| SH.LIST0353 | 山西 | ALL | LIST0353 |
| SH.LIST0354 | 钛白粉概念 | ALL | LIST0354 |
| SH.LIST0355 | 天津自贸区 | ALL | LIST0355 |
| SH.LIST0356 | LED概念 | ALL | LIST0356 |
| SH.LIST0358 | 锂电池概念 | ALL | LIST0358 |
| SH.LIST0359 | 垃圾分类 | ALL | LIST0359 |
| SH.LIST0360 | 杭州湾大湾区 | ALL | LIST0360 |
| SH.LIST0361 | 地热能 | ALL | LIST0361 |
| SH.LIST0362 | 蚂蚁金服概念 | ALL | LIST0362 |
| SH.LIST0363 | 云计算 | ALL | LIST0363 |
| SH.LIST0365 | 猪肉 | ALL | LIST0365 |
| SH.LIST0366 | 白马股 | ALL | LIST0366 |
| SH.LIST0368 | 网络安全 | ALL | LIST0368 |
| SH.LIST0369 | 民营医院 | ALL | LIST0369 |
| SH.LIST0370 | 数字孪生 | ALL | LIST0370 |
| SH.LIST0371 | 数字中国 | ALL | LIST0371 |
| SH.LIST0372 | 透明工厂 | ALL | LIST0372 |
| SH.LIST0373 | 钴概念 | ALL | LIST0373 |
| SH.LIST0374 | 特色小镇 | ALL | LIST0374 |
| SH.LIST0376 | 体育产业 | ALL | LIST0376 |
| SH.LIST0379 | 江西 | ALL | LIST0379 |
| SH.LIST0380 | 京津冀一体化 | ALL | LIST0380 |
| SH.LIST0381 | 可燃冰 | ALL | LIST0381 |
| SH.LIST0383 | 宁夏 | ALL | LIST0383 |
| SH.LIST0384 | 江苏 | ALL | LIST0384 |
| SH.LIST0385 | 稀土永磁 | ALL | LIST0385 |
| SH.LIST0386 | 人造肉 | ALL | LIST0386 |
| SH.LIST0387 | 石墨电极 | ALL | LIST0387 |
| SH.LIST0388 | 辽宁 | ALL | LIST0388 |
| SH.LIST0390 | O2O概念 | ALL | LIST0390 |
| SH.LIST0391 | 海南 | ALL | LIST0391 |
| SH.LIST0392 | 小米概念 | ALL | LIST0392 |
| SH.LIST0393 | 机器人概念 | ALL | LIST0393 |
| SH.LIST0394 | 家用电器概念 | ALL | LIST0394 |
| SH.LIST0395 | 页岩气 | ALL | LIST0395 |
| SH.LIST0397 | 青海 | ALL | LIST0397 |
| SH.LIST0398 | 宁德时代概念 | ALL | LIST0398 |
| SH.LIST0399 | 跨境电商概念 | ALL | LIST0399 |
| SH.LIST0400 | 种植业 | ALL | LIST0400 |
| SH.LIST0401 | 专业工程 | ALL | LIST0401 |
| SH.LIST0402 | 专业连锁Ⅱ | ALL | LIST0402 |
| SH.LIST0403 | 装修装饰Ⅱ | ALL | LIST0403 |
| SH.LIST0411 | 航母概念 | ALL | LIST0411 |
| SH.LIST0412 | PM2.5 | ALL | LIST0412 |
| SH.LIST0413 | 装配式建筑 | ALL | LIST0413 |
| SH.LIST0414 | 单抗概念 | ALL | LIST0414 |
| SH.LIST0415 | 互联网金融 | ALL | LIST0415 |
| SH.LIST0416 | 快递物流 | ALL | LIST0416 |
| SH.LIST0417 | 特高压 | ALL | LIST0417 |
| SH.LIST0418 | 光伏概念 | ALL | LIST0418 |
| SH.LIST0419 | 智能家居 | ALL | LIST0419 |
| SH.LIST0420 | 北斗导航 | ALL | LIST0420 |
| SH.LIST0421 | 全息技术 | ALL | LIST0421 |
| SH.LIST0422 | 参股新三板 | ALL | LIST0422 |
| SH.LIST0423 | 云南 | ALL | LIST0423 |
| SH.LIST0424 | 湖北 | ALL | LIST0424 |
| SH.LIST0426 | 特斯拉 | ALL | LIST0426 |
| SH.LIST0427 | 地下管网 | ALL | LIST0427 |
| SH.LIST0428 | 电力改革 | ALL | LIST0428 |
| SH.LIST0429 | 智慧城市 | ALL | LIST0429 |
| SH.LIST0430 | 3D打印 | ALL | LIST0430 |
| SH.LIST0432 | 波罗的海干散货指数(BDI) | ALL | LIST0432 |
| SH.LIST0433 | 券商 | ALL | LIST0433 |
| SH.LIST0434 | 玻璃概念 | ALL | LIST0434 |
| SH.LIST0436 | 河北 | ALL | LIST0436 |
| SH.LIST0437 | 燃料乙醇 | ALL | LIST0437 |
| SH.LIST0439 | 沪股通 | ALL | LIST0439 |
| SH.LIST0440 | MSCI概念 | ALL | LIST0440 |
| SH.LIST0441 | 超高清视频 | ALL | LIST0441 |
| SH.LIST0442 | 西安自贸区 | ALL | LIST0442 |
| SH.LIST0443 | 3D玻璃 | ALL | LIST0443 |
| SH.LIST0444 | 房地产开发概念 | ALL | LIST0444 |
| SH.LIST0445 | 棉花 | ALL | LIST0445 |
| SH.LIST0446 | 节能照明 | ALL | LIST0446 |
| SH.LIST0448 | 柔性屏 | ALL | LIST0448 |
| SH.LIST0449 | 参股民营银行 | ALL | LIST0449 |
| SH.LIST0450 | 东盟自贸区 | ALL | LIST0450 |
| SH.LIST0451 | 石墨烯 | ALL | LIST0451 |
| SH.LIST0452 | 无人银行 | ALL | LIST0452 |
| SH.LIST0453 | 生物质能 | ALL | LIST0453 |
| SH.LIST0454 | 无人机 | ALL | LIST0454 |
| SH.LIST0455 | 振兴东北 | ALL | LIST0455 |
| SH.LIST0456 | 分散染料 | ALL | LIST0456 |
| SH.LIST0457 | 马彩概念 | ALL | LIST0457 |
| SH.LIST0458 | 华为海思概念 | ALL | LIST0458 |
| SH.LIST0459 | 进口博览会 | ALL | LIST0459 |
| SH.LIST0460 | 网红直播 | ALL | LIST0460 |
| SH.LIST0461 | 量子通信 | ALL | LIST0461 |
| SH.LIST0462 | 杭州亚运会 | ALL | LIST0462 |
| SH.LIST0463 | PCB概念 | ALL | LIST0463 |
| SH.LIST0464 | 靶材 | ALL | LIST0464 |
| SH.LIST0465 | 山东 | ALL | LIST0465 |
| SH.LIST0466 | 在线教育 | ALL | LIST0466 |
| SH.LIST0467 | 语音技术 | ALL | LIST0467 |
| SH.LIST0468 | 一带一路 | ALL | LIST0468 |
| SH.LIST0470 | 彩票概念 | ALL | LIST0470 |
| SH.LIST0471 | 广西 | ALL | LIST0471 |
| SH.LIST0472 | 增强现实(AR) | ALL | LIST0472 |
| SH.LIST0473 | 节能环保 | ALL | LIST0473 |
| SH.LIST0474 | 白酒概念 | ALL | LIST0474 |
| SH.LIST0475 | 生物疫苗 | ALL | LIST0475 |
| SH.LIST0476 | 基因测序 | ALL | LIST0476 |
| SH.LIST0477 | 浙江 | ALL | LIST0477 |
| SH.LIST0478 | 医药电商 | ALL | LIST0478 |
| SH.LIST0479 | 稀缺资源 | ALL | LIST0479 |
| SH.LIST0480 | 新材料 | ALL | LIST0480 |
| SH.LIST0481 | 职业教育 | ALL | LIST0481 |
| SH.LIST0482 | 吉林 | ALL | LIST0482 |
| SH.LIST0483 | 百度概念 | ALL | LIST0483 |
| SH.LIST0484 | 新零售 | ALL | LIST0484 |
| SH.LIST0485 | 污水处理 | ALL | LIST0485 |
| SH.LIST0487 | 重庆 | ALL | LIST0487 |
| SH.LIST0488 | 烟草 | ALL | LIST0488 |
| SH.LIST0489 | 草甘膦 | ALL | LIST0489 |
| SH.LIST0490 | 甘肃 | ALL | LIST0490 |
| SH.LIST0491 | 电子发票 | ALL | LIST0491 |
| SH.LIST0492 | 食品安全 | ALL | LIST0492 |
| SH.LIST0493 | 超导概念 | ALL | LIST0493 |
| SH.LIST0494 | 文化传媒概念 | ALL | LIST0494 |
| SH.LIST0496 | 西藏 | ALL | LIST0496 |
| SH.LIST0497 | 生态农业 | ALL | LIST0497 |
| SH.LIST0498 | 房屋租赁 | ALL | LIST0498 |
| SH.LIST0499 | 水利建设 | ALL | LIST0499 |
| SH.LIST0500 | 军工 | ALL | LIST0500 |
| SH.LIST0532 | 固废处理 | ALL | LIST0532 |
| SH.LIST0533 | 智能电网 | ALL | LIST0533 |
| SH.LIST0534 | 5G概念 | ALL | LIST0534 |
| SH.LIST0535 | 人工智能 | ALL | LIST0535 |
| SH.LIST0537 | 养老金持股 | ALL | LIST0537 |
| SH.LIST0538 | 湖南 | ALL | LIST0538 |
| SH.LIST0540 | 水泥概念 | ALL | LIST0540 |
| SH.LIST0541 | PPP模式 | ALL | LIST0541 |
| SH.LIST0542 | 无人驾驶 | ALL | LIST0542 |
| SH.LIST0543 | 股权转让 | ALL | LIST0543 |
| SH.LIST0544 | 粤港澳自贸区 | ALL | LIST0544 |
| SH.LIST0545 | 充电桩 | ALL | LIST0545 |
| SH.LIST0546 | 金融改革 | ALL | LIST0546 |
| SH.LIST0547 | 天津 | ALL | LIST0547 |
| SH.LIST0548 | 参股券商 | ALL | LIST0548 |
| SH.LIST0549 | 超级电容 | ALL | LIST0549 |
| SH.LIST0550 | 腾讯概念 | ALL | LIST0550 |
| SH.LIST0551 | 深股通 | ALL | LIST0551 |
| SH.LIST0552 | AH股 | ALL | LIST0552 |
| SH.LIST0553 | 摘帽 | ALL | LIST0553 |
| SH.LIST0554 | 国产操作系统 | ALL | LIST0554 |
| SH.LIST0555 | 福建自贸区 | ALL | LIST0555 |
| SH.LIST0557 | 油改概念 | ALL | LIST0557 |
| SH.LIST0558 | 核电概念 | ALL | LIST0558 |
| SH.LIST0559 | 新能源车 | ALL | LIST0559 |
| SH.LIST0560 | 苹果概念 | ALL | LIST0560 |
| SH.LIST0561 | 无人零售 | ALL | LIST0561 |
| SH.LIST0562 | 供应链金融 | ALL | LIST0562 |
| SH.LIST0563 | 举牌概念 | ALL | LIST0563 |
| SH.LIST0564 | 超级品牌 | ALL | LIST0564 |
| SH.LIST0565 | 工业互联网 | ALL | LIST0565 |
| SH.LIST0566 | 蓝宝石 | ALL | LIST0566 |
| SH.LIST0567 | 维生素 | ALL | LIST0567 |
| SH.LIST0568 | 上海 | ALL | LIST0568 |
| SH.LIST0569 | 白糖 | ALL | LIST0569 |
| SH.LIST0570 | 新疆振兴 | ALL | LIST0570 |
| SH.LIST0571 | 智能交通 | ALL | LIST0571 |
| SH.LIST0573 | 青蒿素 | ALL | LIST0573 |
| SH.LIST0575 | 足球概念 | ALL | LIST0575 |
| SH.LIST0576 | 富士康概念 | ALL | LIST0576 |
| SH.LIST0577 | 安徽 | ALL | LIST0577 |
| SH.LIST0578 | 镍概念 | ALL | LIST0578 |
| SH.LIST0579 | 生物医药 | ALL | LIST0579 |
| SH.LIST0580 | 高校 | ALL | LIST0580 |
| SH.LIST0581 | 电子竞技 | ALL | LIST0581 |
| SH.LIST0582 | 广东 | ALL | LIST0582 |
| SH.LIST0583 | 无线充电 | ALL | LIST0583 |
| SH.LIST0584 | 尾气治理 | ALL | LIST0584 |
| SH.LIST0585 | 禽流感 | ALL | LIST0585 |
| SH.LIST0586 | 集成电路概念 | ALL | LIST0586 |
| SH.LIST0587 | 特钢概念 | ALL | LIST0587 |
| SH.LIST0588 | 万达私有化 | ALL | LIST0588 |
| SH.LIST0590 | 创投 | ALL | LIST0590 |
| SH.LIST0591 | 疫苗检测溯源 | ALL | LIST0591 |
| SH.LIST0592 | 黑龙江 | ALL | LIST0592 |
| SH.LIST0593 | 电商概念 | ALL | LIST0593 |
| SH.LIST0594 | 芯片概念 | ALL | LIST0594 |
| SH.LIST0595 | 车联网(车路协同) | ALL | LIST0595 |
| SH.LIST0596 | 四川 | ALL | LIST0596 |
| SH.LIST0597 | 油气设备服务 | ALL | LIST0597 |
| SH.LIST0598 | 陕西 | ALL | LIST0598 |
| SH.LIST0599 | 物流电商平台 | ALL | LIST0599 |
| SH.LIST0601 | 信托概念 | ALL | LIST0601 |
| SH.LIST0602 | 工业4.0 | ALL | LIST0602 |
| SH.LIST0603 | 风电概念 | ALL | LIST0603 |
| SH.LIST0604 | 安防概念 | ALL | LIST0604 |
| SH.LIST0605 | 大飞机 | ALL | LIST0605 |
| SH.LIST0608 | 网络游戏 | ALL | LIST0608 |
| SH.LIST0609 | 土地流转 | ALL | LIST0609 |
| SH.LIST0611 | 期货概念 | ALL | LIST0611 |
| SH.LIST0613 | 华为概念 | ALL | LIST0613 |
| SH.LIST0614 | 海绵城市 | ALL | LIST0614 |
| SH.LIST0615 | 通用航空 | ALL | LIST0615 |
| SH.LIST0616 | 氢能源 | ALL | LIST0616 |
| SH.LIST0617 | 智慧停车 | ALL | LIST0617 |
| SH.LIST0618 | 北京 | ALL | LIST0618 |
| SH.LIST0619 | IPV6 | ALL | LIST0619 |
| SH.LIST0620 | 虚拟现实(VR) | ALL | LIST0620 |
| SH.LIST0621 | 转融券标的 | ALL | LIST0621 |
| SH.LIST0622 | 农村电商 | ALL | LIST0622 |
| SH.LIST0623 | 深圳国资改革 | ALL | LIST0623 |
| SH.LIST0624 | 智能电视 | ALL | LIST0624 |
| SH.LIST0625 | 内蒙古 | ALL | LIST0625 |
| SH.LIST0626 | 触摸屏概念 | ALL | LIST0626 |
| SH.LIST0627 | AB股 | ALL | LIST0627 |
| SH.LIST0628 | 工业大麻 | ALL | LIST0628 |
| SH.LIST0629 | 证金持股 | ALL | LIST0629 |
| SH.LIST0632 | 服装家纺 | ALL | LIST0632 |
| SH.LIST0634 | ETC | ALL | LIST0634 |
| SH.LIST0635 | 磷化工 | ALL | LIST0635 |
| SH.LIST0636 | 光刻机 | ALL | LIST0636 |
| SH.LIST0637 | 上证50 | ALL | LIST0637 |
| SH.LIST0639 | 深圳特区 | ALL | LIST0639 |
| SH.LIST0640 | 分拆上市预期 | ALL | LIST0640 |
| SH.LIST0641 | 数字货币 | ALL | LIST0641 |
| SH.LIST0642 | VPN | ALL | LIST0642 |
| SH.LIST0643 | 医疗美容概念 | ALL | LIST0643 |
| SH.LIST0645 | 中药概念 | ALL | LIST0645 |
| SH.LIST0646 | 基建 | ALL | LIST0646 |
| SH.LIST0647 | 无线耳机 | ALL | LIST0647 |
| SH.LIST0648 | 含可转债 | ALL | LIST0648 |
| SH.LIST0649 | 眼科医疗 | ALL | LIST0649 |
| SH.LIST0650 | 广电系 | ALL | LIST0650 |
| SH.LIST0651 | 影视传媒 | ALL | LIST0651 |
| SH.LIST0652 | 食品饮料概念 | ALL | LIST0652 |
| SH.LIST0653 | WIFI6 | ALL | LIST0653 |
| SH.LIST0655 | 农业种植 | ALL | LIST0655 |
| SH.LIST0656 | 数据中心 | ALL | LIST0656 |
| SH.LIST0657 | 被动元件概念 | ALL | LIST0657 |
| SH.LIST0659 | 氮化镓 | ALL | LIST0659 |
| SH.LIST0660 | 水产品 | ALL | LIST0660 |
| SH.LIST0661 | 远程办公 | ALL | LIST0661 |
| SH.LIST0662 | 传感器 | ALL | LIST0662 |
| SH.LIST0663 | 医疗信息化 | ALL | LIST0663 |
| SH.LIST0664 | 华为HMS | ALL | LIST0664 |
| SH.LIST0665 | 口罩 | ALL | LIST0665 |
| SH.LIST0666 | 呼吸机 | ALL | LIST0666 |
| SH.LIST0667 | 消毒液 | ALL | LIST0667 |
| SH.LIST0668 | 3D摄像头 | ALL | LIST0668 |
| SH.LIST0669 | HJT/HIT电池 | ALL | LIST0669 |
| SH.LIST0670 | 体外诊断概念 | ALL | LIST0670 |
| SH.LIST0671 | 汽车零部件概念 | ALL | LIST0671 |
| SH.LIST0672 | 家居概念 | ALL | LIST0672 |
| SH.LIST0673 | 旅游概念 | ALL | LIST0673 |
| SH.LIST0674 | MiniLED | ALL | LIST0674 |
| SH.LIST0675 | 转基因 | ALL | LIST0675 |
| SH.LIST0676 | 胎压监测 | ALL | LIST0676 |
| SH.LIST0677 | 航运概念 | ALL | LIST0677 |
| SH.LIST0678 | C2M概念 | ALL | LIST0678 |
| SH.LIST0679 | RCS富媒体通信 | ALL | LIST0679 |
| SH.LIST0680 | 煤炭概念 | ALL | LIST0680 |
| SH.LIST0681 | 纺织服装概念 | ALL | LIST0681 |
| SH.LIST0682 | 创新药 | ALL | LIST0682 |
| SH.LIST0683 | 今日头条概念 | ALL | LIST0683 |
| SH.LIST0684 | 钢铁概念 | ALL | LIST0684 |
| SH.LIST0685 | 有色金属概念 | ALL | LIST0685 |
| SH.LIST0686 | 病毒检测 | ALL | LIST0686 |
| SH.LIST0687 | 病毒防治 | ALL | LIST0687 |
| SH.LIST0688 | REITs | ALL | LIST0688 |
| SH.LIST0689 | 化妆品概念 | ALL | LIST0689 |
| SH.LIST0690 | 港口概念 | ALL | LIST0690 |
| SH.LIST0691 | 造纸概念 | ALL | LIST0691 |
| SH.LIST0692 | 西部大开发 | ALL | LIST0692 |
| SH.LIST0693 | 血液制品概念 | ALL | LIST0693 |
| SH.LIST0694 | 电梯概念 | ALL | LIST0694 |
| SH.LIST0695 | 工程机械概念 | ALL | LIST0695 |
| SH.LIST0696 | 包装印刷概念 | ALL | LIST0696 |
| SH.LIST0697 | 轮胎概念 | ALL | LIST0697 |
| SH.LIST0698 | 机器视觉 | ALL | LIST0698 |
| SH.LIST0699 | 免税店概念 | ALL | LIST0699 |
| SH.LIST0700 | 精装修 | ALL | LIST0700 |
| SH.LIST0701 | 碳基半导体 | ALL | LIST0701 |
| SH.LIST0702 | 地摊经济 | ALL | LIST0702 |
| SH.LIST0703 | 盲盒 | ALL | LIST0703 |
| SH.LIST0704 | 汽车整车概念 | ALL | LIST0704 |
| SH.LIST0705 | EDA设计软件 | ALL | LIST0705 |
| SH.LIST0706 | 爱奇艺概念 | ALL | LIST0706 |
| SH.LIST0707 | 保险概念 | ALL | LIST0707 |
| SH.LIST0708 | 银行概念 | ALL | LIST0708 |
| SH.LIST0709 | 中芯国际概念 | ALL | LIST0709 |
| SH.LIST0711 | 长寿药NMN | ALL | LIST0711 |
| SH.LIST0712 | 国家大基金持股 | ALL | LIST0712 |
| SH.LIST0713 | 代糖(甜味剂) | ALL | LIST0713 |
| SH.LIST0714 | 可降解塑料 | ALL | LIST0714 |
| SH.LIST0715 | 脑科学(脑机接口) | ALL | LIST0715 |
| SH.LIST0716 | 碳化硅 | ALL | LIST0716 |
| SH.LIST0717 | 金刚石 | ALL | LIST0717 |
| SH.LIST0718 | 第三代半导体 | ALL | LIST0718 |
| SH.LIST0720 | 快充概念 | ALL | LIST0720 |
| SH.LIST0721 | 蔚来汽车概念 | ALL | LIST0721 |
| SH.LIST0723 | 快手概念 | ALL | LIST0723 |
| SH.LIST0724 | 拼多多概念 | ALL | LIST0724 |
| SH.LIST0768 | 铜概念 | ALL | LIST0768 |
| SH.LIST0769 | 钛 | ALL | LIST0769 |
| SH.LIST0770 | 铝概念 | ALL | LIST0770 |
| SH.LIST0771 | RCEP概念 | ALL | LIST0771 |
| SH.LIST0772 | 黄酒概念 | ALL | LIST0772 |
| SH.LIST0773 | 社区团购 | ALL | LIST0773 |
| SH.LIST0774 | 碳交易 | ALL | LIST0774 |
| SH.LIST0775 | 固态电池 | ALL | LIST0775 |
| SH.LIST0776 | 钨概念 | ALL | LIST0776 |
| SH.LIST0777 | 白银概念 | ALL | LIST0777 |
| SH.LIST0778 | 钼概念 | ALL | LIST0778 |
| SH.LIST0779 | 粘胶短纤 | ALL | LIST0779 |
| SH.LIST0780 | PVC | ALL | LIST0780 |
| SH.LIST0781 | 甲醇 | ALL | LIST0781 |
| SH.LIST0782 | PTA | ALL | LIST0782 |
| SH.LIST0783 | 碳中和 | ALL | LIST0783 |
| SH.LIST0784 | BIPV概念(光伏建筑一体化) | ALL | LIST0784 |
| SH.LIST0785 | NFT文交所 | ALL | LIST0785 |
| SH.LIST0786 | 激光雷达 | ALL | LIST0786 |
| SH.LIST0787 | 辅助生殖 | ALL | LIST0787 |
| SH.LIST0790 | 盐湖提锂 | ALL | LIST0790 |
| SH.LIST0791 | 预制菜 | ALL | LIST0791 |
| SH.LIST0792 | 钠离子电池 | ALL | LIST0792 |
| SH.LIST0793 | 三胎概念 | ALL | LIST0793 |
| SH.LIST0794 | 宠物经济 | ALL | LIST0794 |
| SH.LIST0795 | 华为鸿蒙 | ALL | LIST0795 |
| SH.LIST0796 | 储能概念 | ALL | LIST0796 |
| SH.LIST0797 | 葡萄酒概念 | ALL | LIST0797 |
| SH.LIST0798 | 有机硅类 | ALL | LIST0798 |
| SH.LIST0799 | 专精特新 | ALL | LIST0799 |
| SH.LIST0809 | 工业母机 | ALL | LIST0809 |
| SH.LIST0814 | 境外 | ALL | LIST0814 |
| SH.LIST0815 | 硅锰 | ALL | LIST0815 |
| SH.LIST0816 | 北交所概念 | ALL | LIST0816 |
| SH.LIST0817 | 参股三板精选层 | ALL | LIST0817 |
| SH.LIST0818 | 元宇宙 | ALL | LIST0818 |
| SH.LIST0897 | 双黄连概念 | ALL | LIST0897 |
| SH.LIST0914 | 农业综合Ⅱ | ALL | LIST0914 |
| SH.LIST0915 | 农化制品 | ALL | LIST0915 |
| SH.LIST0916 | 非金属材料Ⅱ | ALL | LIST0916 |
| SH.LIST0917 | 冶钢原料 | ALL | LIST0917 |
| SH.LIST0918 | 普钢 | ALL | LIST0918 |
| SH.LIST0919 | 特钢Ⅱ | ALL | LIST0919 |
| SH.LIST0920 | 能源金属 | ALL | LIST0920 |
| SZ.LIST0922 | 深股通 | ALL | LIST0922 |
| SH.LIST0923 | 电子化学品Ⅱ | ALL | LIST0923 |
| SH.LIST0924 | 乘用车 | ALL | LIST0924 |
| SH.LIST0925 | 商用车 | ALL | LIST0925 |
| SH.LIST0926 | 小家电 | ALL | LIST0926 |
| SH.LIST0927 | 厨卫电器 | ALL | LIST0927 |
| SH.LIST0928 | 照明设备Ⅱ | ALL | LIST0928 |
| SH.LIST0929 | 家电零部件Ⅱ | ALL | LIST0929 |
| SH.LIST0930 | 其他家电Ⅱ | ALL | LIST0930 |
| SH.LIST0931 | 白酒Ⅱ | ALL | LIST0931 |
| SH.LIST0932 | 非白酒 | ALL | LIST0932 |
| SH.LIST0933 | 饮料乳品 | ALL | LIST0933 |
| SH.LIST0934 | 休闲食品 | ALL | LIST0934 |
| SH.LIST0935 | 调味发酵品Ⅱ | ALL | LIST0935 |
| SH.LIST0936 | 饰品 | ALL | LIST0936 |
| SH.LIST0937 | 文娱用品 | ALL | LIST0937 |
| SH.LIST0938 | 铁路公路 | ALL | LIST0938 |
| SH.LIST0939 | 航空机场 | ALL | LIST0939 |
| SH.LIST0940 | 航运港口 | ALL | LIST0940 |
| SH.LIST0941 | 房地产服务 | ALL | LIST0941 |
| SH.LIST0942 | 互联网电商 | ALL | LIST0942 |
| SH.LIST0943 | 旅游零售Ⅱ | ALL | LIST0943 |
| SH.LIST0944 | 体育Ⅱ | ALL | LIST0944 |
| SH.LIST0945 | 专业服务 | ALL | LIST0945 |
| SH.LIST0946 | 酒店餐饮 | ALL | LIST0946 |
| SH.LIST0947 | 教育 | ALL | LIST0947 |
| SH.LIST0948 | 国有大型银行Ⅱ | ALL | LIST0948 |
| SH.LIST0949 | 股份制银行Ⅱ | ALL | LIST0949 |
| SH.LIST0950 | 城商行Ⅱ | ALL | LIST0950 |
| SH.LIST0951 | 农商行Ⅱ | ALL | LIST0951 |
| SH.LIST0952 | 工程咨询服务Ⅱ | ALL | LIST0952 |
| SH.LIST0953 | 光伏设备 | ALL | LIST0953 |
| SH.LIST0954 | 电池 | ALL | LIST0954 |
| SH.LIST0955 | 电网设备 | ALL | LIST0955 |
| SH.LIST0956 | 工程机械 | ALL | LIST0956 |
| SH.LIST0957 | 自动化设备 | ALL | LIST0957 |
| SH.LIST0958 | 军工电子Ⅱ | ALL | LIST0958 |
| SH.LIST0959 | IT服务Ⅱ | ALL | LIST0959 |
| SH.LIST0960 | 软件开发 | ALL | LIST0960 |
| SH.LIST0961 | 游戏Ⅱ | ALL | LIST0961 |
| SH.LIST0962 | 广告营销 | ALL | LIST0962 |
| SH.LIST0963 | 影视院线 | ALL | LIST0963 |
| SH.LIST0964 | 数字媒体 | ALL | LIST0964 |
| SH.LIST0965 | 出版 | ALL | LIST0965 |
| SH.LIST0966 | 电视广播Ⅱ | ALL | LIST0966 |
| SH.LIST0967 | 煤炭开采 | ALL | LIST0967 |
| SH.LIST0968 | 焦炭Ⅱ | ALL | LIST0968 |
| SH.LIST0969 | 油气开采Ⅱ | ALL | LIST0969 |
| SH.LIST0970 | 油服工程 | ALL | LIST0970 |
| SH.LIST0971 | 炼化及贸易 | ALL | LIST0971 |
| SH.LIST0972 | 环境治理 | ALL | LIST0972 |
| SH.LIST0973 | 个护用品 | ALL | LIST0973 |
| SH.LIST0974 | 化妆品 | ALL | LIST0974 |
| SH.LIST0975 | 医疗美容 | ALL | LIST0975 |
| SH.LIST0976 | 旅游及景区 | ALL | LIST0976 |
| SH.LIST0977 | 风电设备 | ALL | LIST0977 |
| SH.LIST0978 | 环保设备Ⅱ | ALL | LIST0978 |
| SH.LIST0982 | 沪股新经济 | ALL | LIST0982 |
| SH.LIST0998 | 联想概念 | ALL | LIST0998 |
| SH.LIST22997 | 锗镓概念 | ALL | LIST22997 |
| SH.LIST22998 | 骨科材料 | ALL | LIST22998 |
| SH.LIST22999 | 应急产业 | ALL | LIST22999 |
| SH.LIST23000 | 自主可控 | ALL | LIST23000 |
| SH.LIST23001 | 东数西算/算力 | ALL | LIST23001 |
| SH.LIST23035 | 在线旅游 | ALL | LIST23035 |
| SH.LIST23036 | 数据安全 | ALL | LIST23036 |
| SH.LIST23037 | 抖音概念 | ALL | LIST23037 |
| SH.LIST23038 | 云游戏 | ALL | LIST23038 |
| SH.LIST23039 | 虚拟电厂 | ALL | LIST23039 |
| SH.LIST23040 | 卫星互联网 | ALL | LIST23040 |
| SH.LIST23041 | Web3.0 | ALL | LIST23041 |
| SH.LIST23042 | AIGC | ALL | LIST23042 |
| SH.LIST23043 | 教育信息化 | ALL | LIST23043 |
| SH.LIST23044 | 华为算力 | ALL | LIST23044 |
| SH.LIST23045 | 短剧/互动游戏 | ALL | LIST23045 |
| SH.LIST23046 | 元梦之星 | ALL | LIST23046 |
| SH.LIST23047 | 6G | ALL | LIST23047 |
| SH.LIST23048 | 毫米波 | ALL | LIST23048 |
| SH.LIST23049 | F5G | ALL | LIST23049 |
| SH.LIST23050 | 5.5G | ALL | LIST23050 |
| SH.LIST23051 | 通感一体化 | ALL | LIST23051 |
| SH.LIST23130 | AI算力芯片 | ALL | LIST23130 |
| SH.LIST23161 | 人民币升值受益 | ALL | LIST23161 |
| SH.LIST23180 | 可控核聚变 | ALL | LIST23180 |
| SH.LIST23344 | 高股息100 | ALL | LIST23344 |
| SH.LIST23368 | 超硬材料 | ALL | LIST23368 |
| SH.LIST23369 | 高端合金 | ALL | LIST23369 |
| SH.LIST23370 | 电信运营 | ALL | LIST23370 |
| SH.LIST23455 | AI应用 | ALL | LIST23455 |
| SH.LIST23480 | 证券IT | ALL | LIST23480 |
| SH.LIST23481 | AI编程 | ALL | LIST23481 |
| SH.LIST23597 | 腾讯云概念 | ALL | LIST23597 |
| SH.LIST23704 | RWA | ALL | LIST23704 |
| SH.LIST23706 | 稳定币概念 | ALL | LIST23706 |
| SH.LIST23916 | 防护服 | ALL | LIST23916 |

#### `CONCEPT` (1)

| code | name | plate_type | plate_id |
|---|---|---|---|
|  | Futu get_plate_list(SH, CONCEPT) failed: 获取板块列表频率太高，请求失败，每30秒最多10次。 | CONCEPT |  |

#### `INDUSTRY` (131)

| code | name | plate_type | plate_id |
|---|---|---|---|
| SH.LIST0001 | 白色家电 | INDUSTRY | LIST0001 |
| SH.LIST0002 | 半导体 | INDUSTRY | LIST0002 |
| SH.LIST0003 | 包装印刷 | INDUSTRY | LIST0003 |
| SH.LIST0004 | 物流 | INDUSTRY | LIST0004 |
| SH.LIST0005 | 航海装备Ⅱ | INDUSTRY | LIST0005 |
| SH.LIST0006 | 电力 | INDUSTRY | LIST0006 |
| SH.LIST0007 | 玻璃玻纤 | INDUSTRY | LIST0007 |
| SH.LIST0008 | 多元金融 | INDUSTRY | LIST0008 |
| SH.LIST0011 | 养殖业 | INDUSTRY | LIST0011 |
| SH.LIST0013 | 纺织制造 | INDUSTRY | LIST0013 |
| SH.LIST0014 | 地面兵装Ⅱ | INDUSTRY | LIST0014 |
| SH.LIST0017 | 电机Ⅱ | INDUSTRY | LIST0017 |
| SH.LIST0020 | 其他电源设备Ⅱ | INDUSTRY | LIST0020 |
| SH.LIST0022 | 消费电子 | INDUSTRY | LIST0022 |
| SH.LIST0025 | 动物保健Ⅱ | INDUSTRY | LIST0025 |
| SH.LIST0026 | 房地产开发 | INDUSTRY | LIST0026 |
| SH.LIST0027 | 化学纤维 | INDUSTRY | LIST0027 |
| SH.LIST0028 | 房屋建设Ⅱ | INDUSTRY | LIST0028 |
| SH.LIST0031 | 工业金属 | INDUSTRY | LIST0031 |
| SH.LIST0032 | 光学光电子 | INDUSTRY | LIST0032 |
| SH.LIST0033 | 航空装备Ⅱ | INDUSTRY | LIST0033 |
| SH.LIST0035 | 航天装备Ⅱ | INDUSTRY | LIST0035 |
| SH.LIST0039 | 化学原料 | INDUSTRY | LIST0039 |
| SH.LIST0040 | 贸易Ⅱ | INDUSTRY | LIST0040 |
| SH.LIST0041 | 化学制品 | INDUSTRY | LIST0041 |
| SH.LIST0042 | 化学制药 | INDUSTRY | LIST0042 |
| SH.LIST0044 | 贵金属 | INDUSTRY | LIST0044 |
| SH.LIST0045 | 基础建设 | INDUSTRY | LIST0045 |
| SH.LIST0047 | 证券Ⅱ | INDUSTRY | LIST0047 |
| SH.LIST0048 | 汽车零部件 | INDUSTRY | LIST0048 |
| SH.LIST0049 | 计算机设备 | INDUSTRY | LIST0049 |
| SH.LIST0051 | 家居用品 | INDUSTRY | LIST0051 |
| SH.LIST0052 | 金属新材料 | INDUSTRY | LIST0052 |
| SH.LIST0053 | 饲料 | INDUSTRY | LIST0053 |
| SH.LIST0055 | 食品加工 | INDUSTRY | LIST0055 |
| SH.LIST0057 | 水泥 | INDUSTRY | LIST0057 |
| SH.LIST0058 | 塑料 | INDUSTRY | LIST0058 |
| SH.LIST0059 | 生物制品 | INDUSTRY | LIST0059 |
| SH.LIST0061 | 通信设备 | INDUSTRY | LIST0061 |
| SH.LIST0063 | 林业Ⅱ | INDUSTRY | LIST0063 |
| SH.LIST0065 | 橡胶 | INDUSTRY | LIST0065 |
| SH.LIST0066 | 农产品加工 | INDUSTRY | LIST0066 |
| SH.LIST0068 | 医疗服务 | INDUSTRY | LIST0068 |
| SH.LIST0071 | 医药商业 | INDUSTRY | LIST0071 |
| SH.LIST0073 | 综合Ⅱ | INDUSTRY | LIST0073 |
| SH.LIST0074 | 中药Ⅱ | INDUSTRY | LIST0074 |
| SH.LIST0075 | 专用设备 | INDUSTRY | LIST0075 |
| SH.LIST0076 | 造纸 | INDUSTRY | LIST0076 |
| SH.LIST0077 | 保险Ⅱ | INDUSTRY | LIST0077 |
| SH.LIST0078 | 其他电子Ⅱ | INDUSTRY | LIST0078 |
| SH.LIST0079 | 装修建材 | INDUSTRY | LIST0079 |
| SH.LIST0080 | 摩托车及其他 | INDUSTRY | LIST0080 |
| SH.LIST0082 | 汽车服务 | INDUSTRY | LIST0082 |
| SH.LIST0083 | 燃气Ⅱ | INDUSTRY | LIST0083 |
| SH.LIST0086 | 黑色家电 | INDUSTRY | LIST0086 |
| SH.LIST0088 | 通信服务 | INDUSTRY | LIST0088 |
| SH.LIST0089 | 通用设备 | INDUSTRY | LIST0089 |
| SH.LIST0090 | 小金属 | INDUSTRY | LIST0090 |
| SH.LIST0091 | 一般零售 | INDUSTRY | LIST0091 |
| SH.LIST0092 | 医疗器械 | INDUSTRY | LIST0092 |
| SH.LIST0095 | 渔业 | INDUSTRY | LIST0095 |
| SH.LIST0096 | 元件 | INDUSTRY | LIST0096 |
| SH.LIST0099 | 轨交设备Ⅱ | INDUSTRY | LIST0099 |
| SH.LIST0400 | 种植业 | INDUSTRY | LIST0400 |
| SH.LIST0401 | 专业工程 | INDUSTRY | LIST0401 |
| SH.LIST0402 | 专业连锁Ⅱ | INDUSTRY | LIST0402 |
| SH.LIST0403 | 装修装饰Ⅱ | INDUSTRY | LIST0403 |
| SH.LIST0632 | 服装家纺 | INDUSTRY | LIST0632 |
| SH.LIST0914 | 农业综合Ⅱ | INDUSTRY | LIST0914 |
| SH.LIST0915 | 农化制品 | INDUSTRY | LIST0915 |
| SH.LIST0916 | 非金属材料Ⅱ | INDUSTRY | LIST0916 |
| SH.LIST0917 | 冶钢原料 | INDUSTRY | LIST0917 |
| SH.LIST0918 | 普钢 | INDUSTRY | LIST0918 |
| SH.LIST0919 | 特钢Ⅱ | INDUSTRY | LIST0919 |
| SH.LIST0920 | 能源金属 | INDUSTRY | LIST0920 |
| SH.LIST0923 | 电子化学品Ⅱ | INDUSTRY | LIST0923 |
| SH.LIST0924 | 乘用车 | INDUSTRY | LIST0924 |
| SH.LIST0925 | 商用车 | INDUSTRY | LIST0925 |
| SH.LIST0926 | 小家电 | INDUSTRY | LIST0926 |
| SH.LIST0927 | 厨卫电器 | INDUSTRY | LIST0927 |
| SH.LIST0928 | 照明设备Ⅱ | INDUSTRY | LIST0928 |
| SH.LIST0929 | 家电零部件Ⅱ | INDUSTRY | LIST0929 |
| SH.LIST0930 | 其他家电Ⅱ | INDUSTRY | LIST0930 |
| SH.LIST0931 | 白酒Ⅱ | INDUSTRY | LIST0931 |
| SH.LIST0932 | 非白酒 | INDUSTRY | LIST0932 |
| SH.LIST0933 | 饮料乳品 | INDUSTRY | LIST0933 |
| SH.LIST0934 | 休闲食品 | INDUSTRY | LIST0934 |
| SH.LIST0935 | 调味发酵品Ⅱ | INDUSTRY | LIST0935 |
| SH.LIST0936 | 饰品 | INDUSTRY | LIST0936 |
| SH.LIST0937 | 文娱用品 | INDUSTRY | LIST0937 |
| SH.LIST0938 | 铁路公路 | INDUSTRY | LIST0938 |
| SH.LIST0939 | 航空机场 | INDUSTRY | LIST0939 |
| SH.LIST0940 | 航运港口 | INDUSTRY | LIST0940 |
| SH.LIST0941 | 房地产服务 | INDUSTRY | LIST0941 |
| SH.LIST0942 | 互联网电商 | INDUSTRY | LIST0942 |
| SH.LIST0943 | 旅游零售Ⅱ | INDUSTRY | LIST0943 |
| SH.LIST0944 | 体育Ⅱ | INDUSTRY | LIST0944 |
| SH.LIST0945 | 专业服务 | INDUSTRY | LIST0945 |
| SH.LIST0946 | 酒店餐饮 | INDUSTRY | LIST0946 |
| SH.LIST0947 | 教育 | INDUSTRY | LIST0947 |
| SH.LIST0948 | 国有大型银行Ⅱ | INDUSTRY | LIST0948 |
| SH.LIST0949 | 股份制银行Ⅱ | INDUSTRY | LIST0949 |
| SH.LIST0950 | 城商行Ⅱ | INDUSTRY | LIST0950 |
| SH.LIST0951 | 农商行Ⅱ | INDUSTRY | LIST0951 |
| SH.LIST0952 | 工程咨询服务Ⅱ | INDUSTRY | LIST0952 |
| SH.LIST0953 | 光伏设备 | INDUSTRY | LIST0953 |
| SH.LIST0954 | 电池 | INDUSTRY | LIST0954 |
| SH.LIST0955 | 电网设备 | INDUSTRY | LIST0955 |
| SH.LIST0956 | 工程机械 | INDUSTRY | LIST0956 |
| SH.LIST0957 | 自动化设备 | INDUSTRY | LIST0957 |
| SH.LIST0958 | 军工电子Ⅱ | INDUSTRY | LIST0958 |
| SH.LIST0959 | IT服务Ⅱ | INDUSTRY | LIST0959 |
| SH.LIST0960 | 软件开发 | INDUSTRY | LIST0960 |
| SH.LIST0961 | 游戏Ⅱ | INDUSTRY | LIST0961 |
| SH.LIST0962 | 广告营销 | INDUSTRY | LIST0962 |
| SH.LIST0963 | 影视院线 | INDUSTRY | LIST0963 |
| SH.LIST0964 | 数字媒体 | INDUSTRY | LIST0964 |
| SH.LIST0965 | 出版 | INDUSTRY | LIST0965 |
| SH.LIST0966 | 电视广播Ⅱ | INDUSTRY | LIST0966 |
| SH.LIST0967 | 煤炭开采 | INDUSTRY | LIST0967 |
| SH.LIST0968 | 焦炭Ⅱ | INDUSTRY | LIST0968 |
| SH.LIST0969 | 油气开采Ⅱ | INDUSTRY | LIST0969 |
| SH.LIST0970 | 油服工程 | INDUSTRY | LIST0970 |
| SH.LIST0971 | 炼化及贸易 | INDUSTRY | LIST0971 |
| SH.LIST0972 | 环境治理 | INDUSTRY | LIST0972 |
| SH.LIST0973 | 个护用品 | INDUSTRY | LIST0973 |
| SH.LIST0974 | 化妆品 | INDUSTRY | LIST0974 |
| SH.LIST0975 | 医疗美容 | INDUSTRY | LIST0975 |
| SH.LIST0976 | 旅游及景区 | INDUSTRY | LIST0976 |
| SH.LIST0977 | 风电设备 | INDUSTRY | LIST0977 |
| SH.LIST0978 | 环保设备Ⅱ | INDUSTRY | LIST0978 |

#### `REGION` (1)

| code | name | plate_type | plate_id |
|---|---|---|---|
|  | Futu get_plate_list(SH, REGION) failed: 获取板块列表频率太高，请求失败，每30秒最多10次。 | REGION |  |

## Market `SZ`

### App-Style Screener Categories

#### 行情 (`market_quote`)

Use market/plate to constrain the universe, then quote and accumulated fields to find liquid, tradable, strong or weak candidates.

- **交易所/市场**: `market_selector`
  - Supported values: `US, HK, SH, SZ`
  - LLM hint: Select the listing market. This is not a StockField.
- **所属行业/概念/板块**: `plate_selector`
  - LLM hint: Probe theme words as plate_keywords. If a useful plate_code is found, stock_filter can filter within that plate_code.
- **价格/52周位置/量比/委比/每手价格**: `stock_filter`
  - Fields: `CUR_PRICE, CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO, CUR_PRICE_TO_LOWEST52_WEEKS_RATIO, HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO, LOW_PRICE_TO_LOWEST52_WEEKS_RATIO, VOLUME_RATIO, BID_ASK_RATIO, LOT_PRICE`
  - LLM hint: Use these for tradability, momentum context, and avoiding illiquid tails.
- **涨跌幅/振幅/成交量/成交额/换手率**: `stock_filter`
  - Fields: `CHANGE_RATE, AMPLITUDE, VOLUME, TURNOVER, TURNOVER_RATE`
  - LLM hint: Use days plus sort/filter bounds to discover strong momentum or high liquidity names, e.g. TURNOVER DESCEND or CHANGE_RATE over 20/60 days.

#### 估值 (`valuation`)

Use valuation as a secondary screen after thematic relevance; do not let cheapness replace theme fit.

- **市值/流通市值/股本**: `stock_filter`
  - Fields: `MARKET_VAL, FLOAT_MARKET_VAL, TOTAL_SHARE, FLOAT_SHARE`
  - LLM hint: Use market cap to separate mega-cap anchors from small high-beta satellites.
- **市盈率（静态/TTM）/市净率/市销率/市现率**: `stock_filter`
  - Fields: `PE_ANNUAL, PE_TTM, PB_RATE, PS_TTM, PCF_TTM`
  - LLM hint: Useful for valuation risk flags and relative comparison, not theme discovery alone.
- **估值分位/行业估值分位**: `derived_or_future_adapter`
  - Alternate source: `derive from peer universe snapshots/financials; not exposed as current get_stock_filter StockField`
  - LLM hint: Do not pass valuation percentile as stock_filter_specs. Mark it as desired downstream enrichment if needed.

#### 分红 (`dividend`)

Dividend data is enrichment, not a get_stock_filter field in this SDK.

- **TTM 分红/股息率**: `non_stock_filter`
  - Alternate source: `get_market_snapshot fields dividend_ttm/dividend_ratio_ttm`
  - LLM hint: Do not create stock_filter_specs for dividends; request quote snapshot enrichment later.

#### 技术 (`technical`)

Use technical filters to discover trend candidates or confirm that a theme is active; avoid using them as the only reason a name belongs to a theme.

- **指标解读**: `derived_interpretation`
  - Alternate source: `derive from pattern/custom_indicator StockFields and K-line enrichment`
  - LLM hint: The app-style interpretation layer is not a single OpenAPI enum. Use the specific MA/EMA/KDJ/RSI/MACD/BOLL fields below.
- **MA/EMA 均线形态**: `stock_filter`
  - Fields: `MA_ALIGNMENT_LONG, MA_ALIGNMENT_SHORT, EMA_ALIGNMENT_LONG, EMA_ALIGNMENT_SHORT`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use for trend/momentum probes, e.g. MA_ALIGNMENT_LONG on K_DAY.
- **RSI/KDJ/MACD/BOLL 形态**: `stock_filter`
  - Fields: `RSI_GOLD_CROSS_LOW, RSI_DEATH_CROSS_HIGH, RSI_TOP_DIVERGENCE, RSI_BOTTOM_DIVERGENCE, KDJ_GOLD_CROSS_LOW, KDJ_DEATH_CROSS_HIGH, KDJ_TOP_DIVERGENCE, KDJ_BOTTOM_DIVERGENCE, MACD_GOLD_CROSS_LOW, MACD_DEATH_CROSS_HIGH, MACD_TOP_DIVERGENCE, MACD_BOTTOM_DIVERGENCE, BOLL_BREAK_UPPER, BOLL_BREAK_LOWER, BOLL_CROSS_MIDDLE_UP, BOLL_CROSS_MIDDLE_DOWN`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use for oscillator and Bollinger-band signals when discovering active candidates.
- **MA/EMA/KDJ/RSI/MACD/BOLL 自定义指标比较**: `stock_filter`
  - Fields: `PRICE, MA5, MA10, MA20, MA30, MA60, MA120, MA250, RSI, EMA5, EMA10, EMA20, EMA30, EMA60, EMA120, EMA250, VALUE, MA, EMA, KDJ_K, KDJ_D, KDJ_J, MACD_DIFF, MACD_DEA, MACD, BOLL_UPPER, BOLL_MIDDLER, BOLL_LOWER`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use only when a simple pattern field is insufficient.

#### 财务 (`financial`)

Use fundamentals to filter quality and growth after the theme map is drafted.

- **利润/收入/利润率/ROE/ROIC/现金流/资产负债/增长率/EPS**: `stock_filter`
  - Fields: `NET_PROFIT, NET_PROFIX_GROWTH, SUM_OF_BUSINESS, SUM_OF_BUSINESS_GROWTH, NET_PROFIT_RATE, GROSS_PROFIT_RATE, DEBT_ASSET_RATE, RETURN_ON_EQUITY_RATE, ROIC, ROA_TTM, EBIT_TTM, EBITDA, OPERATING_MARGIN_TTM, EBIT_MARGIN, EBITDA_MARGIN, FINANCIAL_COST_RATE, OPERATING_PROFIT_TTM, SHAREHOLDER_NET_PROFIT_TTM, NET_PROFIT_CASH_COVER_TTM, CURRENT_RATIO, QUICK_RATIO, CURRENT_ASSET_RATIO, CURRENT_DEBT_RATIO, EQUITY_MULTIPLIER, PROPERTY_RATIO, CASH_AND_CASH_EQUIVALENTS, TOTAL_ASSET_TURNOVER, FIXED_ASSET_TURNOVER, INVENTORY_TURNOVER, OPERATING_CASH_FLOW_TTM, ACCOUNTS_RECEIVABLE, EBIT_GROWTH_RATE, OPERATING_PROFIT_GROWTH_RATE, TOTAL_ASSETS_GROWTH_RATE, PROFIT_TO_SHAREHOLDERS_GROWTH_RATE, PROFIT_BEFORE_TAX_GROWTH_RATE, EPS_GROWTH_RATE, ROE_GROWTH_RATE, ROIC_GROWTH_RATE, NOCF_GROWTH_RATE, NOCF_PER_SHARE_GROWTH_RATE, OPERATING_REVENUE_CASH_COVER, OPERATING_PROFIT_TO_TOTAL_PROFIT, BASIC_EPS, DILUTED_EPS, NOCF_PER_SHARE`
  - LLM hint: Useful probes: revenue growth, gross margin, ROE/ROIC, operating cash flow, debt ratio. Exact numbers still need downstream SEC/fundamental validation.

#### 分析 (`analysis`)

Analyst ratings/revisions are not exposed by get_stock_filter here.

- **评级/目标价/一致预期/盈利修正**: `external_or_future_adapter`
  - Alternate source: `analyst_estimate_revision_adapter`
  - LLM hint: Do not invent these facts. Mark as needed evidence if useful.

#### 期权 (`options`)

Options are not stock_filter fields. Use option-chain/surface enrichment after candidate discovery.

- **期权活跃度/IV/到期日/Put-Call context**: `option_chain_enrichment`
  - Alternate source: `get_option_expiration_date + get_option_chain + option snapshots`
  - LLM hint: Use later to assess tradability and income-strategy readiness, not as stock_filter_specs.

### OpenAPI Stock Filter Fields

#### `simple` (21)

`STOCK_CODE`, `STOCK_NAME`, `CUR_PRICE`, `CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO`, `CUR_PRICE_TO_LOWEST52_WEEKS_RATIO`, `HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO`, `LOW_PRICE_TO_LOWEST52_WEEKS_RATIO`, `VOLUME_RATIO`, `BID_ASK_RATIO`, `LOT_PRICE`, `MARKET_VAL`, `PE_ANNUAL`, `PE_TTM`, `PB_RATE`, `CHANGE_RATE_5MIN`, `CHANGE_RATE_BEGIN_YEAR`, `PS_TTM`, `PCF_TTM`, `TOTAL_SHARE`, `FLOAT_SHARE`, `FLOAT_MARKET_VAL`

#### `accumulate` (5)

`CHANGE_RATE`, `AMPLITUDE`, `VOLUME`, `TURNOVER`, `TURNOVER_RATE`

#### `financial` (46)

`NET_PROFIT`, `NET_PROFIX_GROWTH`, `SUM_OF_BUSINESS`, `SUM_OF_BUSINESS_GROWTH`, `NET_PROFIT_RATE`, `GROSS_PROFIT_RATE`, `DEBT_ASSET_RATE`, `RETURN_ON_EQUITY_RATE`, `ROIC`, `ROA_TTM`, `EBIT_TTM`, `EBITDA`, `OPERATING_MARGIN_TTM`, `EBIT_MARGIN`, `EBITDA_MARGIN`, `FINANCIAL_COST_RATE`, `OPERATING_PROFIT_TTM`, `SHAREHOLDER_NET_PROFIT_TTM`, `NET_PROFIT_CASH_COVER_TTM`, `CURRENT_RATIO`, `QUICK_RATIO`, `CURRENT_ASSET_RATIO`, `CURRENT_DEBT_RATIO`, `EQUITY_MULTIPLIER`, `PROPERTY_RATIO`, `CASH_AND_CASH_EQUIVALENTS`, `TOTAL_ASSET_TURNOVER`, `FIXED_ASSET_TURNOVER`, `INVENTORY_TURNOVER`, `OPERATING_CASH_FLOW_TTM`, `ACCOUNTS_RECEIVABLE`, `EBIT_GROWTH_RATE`, `OPERATING_PROFIT_GROWTH_RATE`, `TOTAL_ASSETS_GROWTH_RATE`, `PROFIT_TO_SHAREHOLDERS_GROWTH_RATE`, `PROFIT_BEFORE_TAX_GROWTH_RATE`, `EPS_GROWTH_RATE`, `ROE_GROWTH_RATE`, `ROIC_GROWTH_RATE`, `NOCF_GROWTH_RATE`, `NOCF_PER_SHARE_GROWTH_RATE`, `OPERATING_REVENUE_CASH_COVER`, `OPERATING_PROFIT_TO_TOTAL_PROFIT`, `BASIC_EPS`, `DILUTED_EPS`, `NOCF_PER_SHARE`

#### `pattern` (20)

`MA_ALIGNMENT_LONG`, `MA_ALIGNMENT_SHORT`, `EMA_ALIGNMENT_LONG`, `EMA_ALIGNMENT_SHORT`, `RSI_GOLD_CROSS_LOW`, `RSI_DEATH_CROSS_HIGH`, `RSI_TOP_DIVERGENCE`, `RSI_BOTTOM_DIVERGENCE`, `KDJ_GOLD_CROSS_LOW`, `KDJ_DEATH_CROSS_HIGH`, `KDJ_TOP_DIVERGENCE`, `KDJ_BOTTOM_DIVERGENCE`, `MACD_GOLD_CROSS_LOW`, `MACD_DEATH_CROSS_HIGH`, `MACD_TOP_DIVERGENCE`, `MACD_BOTTOM_DIVERGENCE`, `BOLL_BREAK_UPPER`, `BOLL_BREAK_LOWER`, `BOLL_CROSS_MIDDLE_UP`, `BOLL_CROSS_MIDDLE_DOWN`

#### `custom_indicator` (28)

`PRICE`, `MA5`, `MA10`, `MA20`, `MA30`, `MA60`, `MA120`, `MA250`, `RSI`, `EMA5`, `EMA10`, `EMA20`, `EMA30`, `EMA60`, `EMA120`, `EMA250`, `VALUE`, `MA`, `EMA`, `KDJ_K`, `KDJ_D`, `KDJ_J`, `MACD_DIFF`, `MACD_DEA`, `MACD`, `BOLL_UPPER`, `BOLL_MIDDLER`, `BOLL_LOWER`

- `sort_dir`: `ASCEND`, `DESCEND`
- `financial_quarter`: `ANNUAL`, `FIRST_QUARTER`, `INTERIM`, `THIRD_QUARTER`, `MOST_RECENT_QUARTER`
- `supported_pattern_ktype`: `K_60M`, `K_DAY`, `K_WEEK`, `K_MON`
- `relative_position`: `MORE`, `LESS`, `CROSS_UP`, `CROSS_DOWN`

### Futu Plate Choices

#### `ALL` (1)

| code | name | plate_type | plate_id |
|---|---|---|---|
|  | Futu get_plate_list(SZ, ALL) failed: 获取板块列表频率太高，请求失败，每30秒最多10次。 | ALL |  |

#### `CONCEPT` (1)

| code | name | plate_type | plate_id |
|---|---|---|---|
|  | Futu get_plate_list(SZ, CONCEPT) failed: 获取板块列表频率太高，请求失败，每30秒最多10次。 | CONCEPT |  |

#### `INDUSTRY` (1)

| code | name | plate_type | plate_id |
|---|---|---|---|
|  | Futu get_plate_list(SZ, INDUSTRY) failed: 获取板块列表频率太高，请求失败，每30秒最多10次。 | INDUSTRY |  |

#### `REGION` (1)

| code | name | plate_type | plate_id |
|---|---|---|---|
|  | Futu get_plate_list(SZ, REGION) failed: 获取板块列表频率太高，请求失败，每30秒最多10次。 | REGION |  |

## Market `US`

### App-Style Screener Categories

#### 行情 (`market_quote`)

Use market/plate to constrain the universe, then quote and accumulated fields to find liquid, tradable, strong or weak candidates.

- **交易所/市场**: `market_selector`
  - Supported values: `US, HK, SH, SZ`
  - LLM hint: Select the listing market. This is not a StockField.
- **所属行业/概念/板块**: `plate_selector`
  - LLM hint: Probe theme words as plate_keywords. If a useful plate_code is found, stock_filter can filter within that plate_code.
- **价格/52周位置/量比/委比/每手价格**: `stock_filter`
  - Fields: `CUR_PRICE, CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO, CUR_PRICE_TO_LOWEST52_WEEKS_RATIO, HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO, LOW_PRICE_TO_LOWEST52_WEEKS_RATIO, VOLUME_RATIO, BID_ASK_RATIO, LOT_PRICE`
  - LLM hint: Use these for tradability, momentum context, and avoiding illiquid tails.
- **涨跌幅/振幅/成交量/成交额/换手率**: `stock_filter`
  - Fields: `CHANGE_RATE, AMPLITUDE, VOLUME, TURNOVER, TURNOVER_RATE`
  - LLM hint: Use days plus sort/filter bounds to discover strong momentum or high liquidity names, e.g. TURNOVER DESCEND or CHANGE_RATE over 20/60 days.

#### 估值 (`valuation`)

Use valuation as a secondary screen after thematic relevance; do not let cheapness replace theme fit.

- **市值/流通市值/股本**: `stock_filter`
  - Fields: `MARKET_VAL, FLOAT_MARKET_VAL, TOTAL_SHARE, FLOAT_SHARE`
  - LLM hint: Use market cap to separate mega-cap anchors from small high-beta satellites.
- **市盈率（静态/TTM）/市净率/市销率/市现率**: `stock_filter`
  - Fields: `PE_ANNUAL, PE_TTM, PB_RATE, PS_TTM, PCF_TTM`
  - LLM hint: Useful for valuation risk flags and relative comparison, not theme discovery alone.
- **估值分位/行业估值分位**: `derived_or_future_adapter`
  - Alternate source: `derive from peer universe snapshots/financials; not exposed as current get_stock_filter StockField`
  - LLM hint: Do not pass valuation percentile as stock_filter_specs. Mark it as desired downstream enrichment if needed.

#### 分红 (`dividend`)

Dividend data is enrichment, not a get_stock_filter field in this SDK.

- **TTM 分红/股息率**: `non_stock_filter`
  - Alternate source: `get_market_snapshot fields dividend_ttm/dividend_ratio_ttm`
  - LLM hint: Do not create stock_filter_specs for dividends; request quote snapshot enrichment later.

#### 技术 (`technical`)

Use technical filters to discover trend candidates or confirm that a theme is active; avoid using them as the only reason a name belongs to a theme.

- **指标解读**: `derived_interpretation`
  - Alternate source: `derive from pattern/custom_indicator StockFields and K-line enrichment`
  - LLM hint: The app-style interpretation layer is not a single OpenAPI enum. Use the specific MA/EMA/KDJ/RSI/MACD/BOLL fields below.
- **MA/EMA 均线形态**: `stock_filter`
  - Fields: `MA_ALIGNMENT_LONG, MA_ALIGNMENT_SHORT, EMA_ALIGNMENT_LONG, EMA_ALIGNMENT_SHORT`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use for trend/momentum probes, e.g. MA_ALIGNMENT_LONG on K_DAY.
- **RSI/KDJ/MACD/BOLL 形态**: `stock_filter`
  - Fields: `RSI_GOLD_CROSS_LOW, RSI_DEATH_CROSS_HIGH, RSI_TOP_DIVERGENCE, RSI_BOTTOM_DIVERGENCE, KDJ_GOLD_CROSS_LOW, KDJ_DEATH_CROSS_HIGH, KDJ_TOP_DIVERGENCE, KDJ_BOTTOM_DIVERGENCE, MACD_GOLD_CROSS_LOW, MACD_DEATH_CROSS_HIGH, MACD_TOP_DIVERGENCE, MACD_BOTTOM_DIVERGENCE, BOLL_BREAK_UPPER, BOLL_BREAK_LOWER, BOLL_CROSS_MIDDLE_UP, BOLL_CROSS_MIDDLE_DOWN`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use for oscillator and Bollinger-band signals when discovering active candidates.
- **MA/EMA/KDJ/RSI/MACD/BOLL 自定义指标比较**: `stock_filter`
  - Fields: `PRICE, MA5, MA10, MA20, MA30, MA60, MA120, MA250, RSI, EMA5, EMA10, EMA20, EMA30, EMA60, EMA120, EMA250, VALUE, MA, EMA, KDJ_K, KDJ_D, KDJ_J, MACD_DIFF, MACD_DEA, MACD, BOLL_UPPER, BOLL_MIDDLER, BOLL_LOWER`
  - Supported values: `K_60M, K_DAY, K_WEEK, K_MON`
  - LLM hint: Use only when a simple pattern field is insufficient.

#### 财务 (`financial`)

Use fundamentals to filter quality and growth after the theme map is drafted.

- **利润/收入/利润率/ROE/ROIC/现金流/资产负债/增长率/EPS**: `stock_filter`
  - Fields: `NET_PROFIT, NET_PROFIX_GROWTH, SUM_OF_BUSINESS, SUM_OF_BUSINESS_GROWTH, NET_PROFIT_RATE, GROSS_PROFIT_RATE, DEBT_ASSET_RATE, RETURN_ON_EQUITY_RATE, ROIC, ROA_TTM, EBIT_TTM, EBITDA, OPERATING_MARGIN_TTM, EBIT_MARGIN, EBITDA_MARGIN, FINANCIAL_COST_RATE, OPERATING_PROFIT_TTM, SHAREHOLDER_NET_PROFIT_TTM, NET_PROFIT_CASH_COVER_TTM, CURRENT_RATIO, QUICK_RATIO, CURRENT_ASSET_RATIO, CURRENT_DEBT_RATIO, EQUITY_MULTIPLIER, PROPERTY_RATIO, CASH_AND_CASH_EQUIVALENTS, TOTAL_ASSET_TURNOVER, FIXED_ASSET_TURNOVER, INVENTORY_TURNOVER, OPERATING_CASH_FLOW_TTM, ACCOUNTS_RECEIVABLE, EBIT_GROWTH_RATE, OPERATING_PROFIT_GROWTH_RATE, TOTAL_ASSETS_GROWTH_RATE, PROFIT_TO_SHAREHOLDERS_GROWTH_RATE, PROFIT_BEFORE_TAX_GROWTH_RATE, EPS_GROWTH_RATE, ROE_GROWTH_RATE, ROIC_GROWTH_RATE, NOCF_GROWTH_RATE, NOCF_PER_SHARE_GROWTH_RATE, OPERATING_REVENUE_CASH_COVER, OPERATING_PROFIT_TO_TOTAL_PROFIT, BASIC_EPS, DILUTED_EPS, NOCF_PER_SHARE`
  - LLM hint: Useful probes: revenue growth, gross margin, ROE/ROIC, operating cash flow, debt ratio. Exact numbers still need downstream SEC/fundamental validation.

#### 分析 (`analysis`)

Analyst ratings/revisions are not exposed by get_stock_filter here.

- **评级/目标价/一致预期/盈利修正**: `external_or_future_adapter`
  - Alternate source: `analyst_estimate_revision_adapter`
  - LLM hint: Do not invent these facts. Mark as needed evidence if useful.

#### 期权 (`options`)

Options are not stock_filter fields. Use option-chain/surface enrichment after candidate discovery.

- **期权活跃度/IV/到期日/Put-Call context**: `option_chain_enrichment`
  - Alternate source: `get_option_expiration_date + get_option_chain + option snapshots`
  - LLM hint: Use later to assess tradability and income-strategy readiness, not as stock_filter_specs.

### OpenAPI Stock Filter Fields

#### `simple` (21)

`STOCK_CODE`, `STOCK_NAME`, `CUR_PRICE`, `CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO`, `CUR_PRICE_TO_LOWEST52_WEEKS_RATIO`, `HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO`, `LOW_PRICE_TO_LOWEST52_WEEKS_RATIO`, `VOLUME_RATIO`, `BID_ASK_RATIO`, `LOT_PRICE`, `MARKET_VAL`, `PE_ANNUAL`, `PE_TTM`, `PB_RATE`, `CHANGE_RATE_5MIN`, `CHANGE_RATE_BEGIN_YEAR`, `PS_TTM`, `PCF_TTM`, `TOTAL_SHARE`, `FLOAT_SHARE`, `FLOAT_MARKET_VAL`

#### `accumulate` (5)

`CHANGE_RATE`, `AMPLITUDE`, `VOLUME`, `TURNOVER`, `TURNOVER_RATE`

#### `financial` (46)

`NET_PROFIT`, `NET_PROFIX_GROWTH`, `SUM_OF_BUSINESS`, `SUM_OF_BUSINESS_GROWTH`, `NET_PROFIT_RATE`, `GROSS_PROFIT_RATE`, `DEBT_ASSET_RATE`, `RETURN_ON_EQUITY_RATE`, `ROIC`, `ROA_TTM`, `EBIT_TTM`, `EBITDA`, `OPERATING_MARGIN_TTM`, `EBIT_MARGIN`, `EBITDA_MARGIN`, `FINANCIAL_COST_RATE`, `OPERATING_PROFIT_TTM`, `SHAREHOLDER_NET_PROFIT_TTM`, `NET_PROFIT_CASH_COVER_TTM`, `CURRENT_RATIO`, `QUICK_RATIO`, `CURRENT_ASSET_RATIO`, `CURRENT_DEBT_RATIO`, `EQUITY_MULTIPLIER`, `PROPERTY_RATIO`, `CASH_AND_CASH_EQUIVALENTS`, `TOTAL_ASSET_TURNOVER`, `FIXED_ASSET_TURNOVER`, `INVENTORY_TURNOVER`, `OPERATING_CASH_FLOW_TTM`, `ACCOUNTS_RECEIVABLE`, `EBIT_GROWTH_RATE`, `OPERATING_PROFIT_GROWTH_RATE`, `TOTAL_ASSETS_GROWTH_RATE`, `PROFIT_TO_SHAREHOLDERS_GROWTH_RATE`, `PROFIT_BEFORE_TAX_GROWTH_RATE`, `EPS_GROWTH_RATE`, `ROE_GROWTH_RATE`, `ROIC_GROWTH_RATE`, `NOCF_GROWTH_RATE`, `NOCF_PER_SHARE_GROWTH_RATE`, `OPERATING_REVENUE_CASH_COVER`, `OPERATING_PROFIT_TO_TOTAL_PROFIT`, `BASIC_EPS`, `DILUTED_EPS`, `NOCF_PER_SHARE`

#### `pattern` (20)

`MA_ALIGNMENT_LONG`, `MA_ALIGNMENT_SHORT`, `EMA_ALIGNMENT_LONG`, `EMA_ALIGNMENT_SHORT`, `RSI_GOLD_CROSS_LOW`, `RSI_DEATH_CROSS_HIGH`, `RSI_TOP_DIVERGENCE`, `RSI_BOTTOM_DIVERGENCE`, `KDJ_GOLD_CROSS_LOW`, `KDJ_DEATH_CROSS_HIGH`, `KDJ_TOP_DIVERGENCE`, `KDJ_BOTTOM_DIVERGENCE`, `MACD_GOLD_CROSS_LOW`, `MACD_DEATH_CROSS_HIGH`, `MACD_TOP_DIVERGENCE`, `MACD_BOTTOM_DIVERGENCE`, `BOLL_BREAK_UPPER`, `BOLL_BREAK_LOWER`, `BOLL_CROSS_MIDDLE_UP`, `BOLL_CROSS_MIDDLE_DOWN`

#### `custom_indicator` (28)

`PRICE`, `MA5`, `MA10`, `MA20`, `MA30`, `MA60`, `MA120`, `MA250`, `RSI`, `EMA5`, `EMA10`, `EMA20`, `EMA30`, `EMA60`, `EMA120`, `EMA250`, `VALUE`, `MA`, `EMA`, `KDJ_K`, `KDJ_D`, `KDJ_J`, `MACD_DIFF`, `MACD_DEA`, `MACD`, `BOLL_UPPER`, `BOLL_MIDDLER`, `BOLL_LOWER`

- `sort_dir`: `ASCEND`, `DESCEND`
- `financial_quarter`: `ANNUAL`, `FIRST_QUARTER`, `INTERIM`, `THIRD_QUARTER`, `MOST_RECENT_QUARTER`
- `supported_pattern_ktype`: `K_60M`, `K_DAY`, `K_WEEK`, `K_MON`
- `relative_position`: `MORE`, `LESS`, `CROSS_UP`, `CROSS_DOWN`

### Futu Plate Choices

#### `ALL` (350)

| code | name | plate_type | plate_id |
|---|---|---|---|
| US.LIST20010 | 加密货币概念股 | ALL | LIST20010 |
| US.LIST2003 | 家庭和个人用品 | ALL | LIST2003 |
| US.LIST2004 | 互联网内容与信息 | ALL | LIST2004 |
| US.LIST2005 | 住宅施工 | ALL | LIST2005 |
| US.LIST2007 | 农产品 | ALL | LIST2007 |
| US.LIST20077 | 半导体精选 | ALL | LIST20077 |
| US.LIST2008 | 出版 | ALL | LIST2008 |
| US.LIST2010 | 农业投入品 | ALL | LIST2010 |
| US.LIST2011 | 诊断与研究 | ALL | LIST2011 |
| US.LIST2014 | 医疗设备和用品 | ALL | LIST2014 |
| US.LIST2015 | 半导体 | ALL | LIST2015 |
| US.LIST2016 | 半导体设备与材料 | ALL | LIST2016 |
| US.LIST2020 | 化学制品 | ALL | LIST2020 |
| US.LIST2030 | 广告代理 | ALL | LIST2030 |
| US.LIST2033 | 建筑工程 | ALL | LIST2033 |
| US.LIST2034 | 建筑材料 | ALL | LIST2034 |
| US.LIST2038 | 房地产服务 | ALL | LIST2038 |
| US.LIST2044 | 教育培训 | ALL | LIST2044 |
| US.LIST2046 | 度假村与赌场 | ALL | LIST2046 |
| US.LIST2047 | 太阳能 | ALL | LIST2047 |
| US.LIST2049 | 服装制造 | ALL | LIST2049 |
| US.LIST2052 | 木材与木制品生产 | ALL | LIST2052 |
| US.LIST2055 | 汽车零件 | ALL | LIST2055 |
| US.LIST2058 | 油气勘探与开发 | ALL | LIST2058 |
| US.LIST2060 | 油气炼制与销售 | ALL | LIST2060 |
| US.LIST2063 | 休闲 | ALL | LIST2063 |
| US.LIST2068 | 特殊化工用品 | ALL | LIST2068 |
| US.LIST2069 | 生物技术 | ALL | LIST2069 |
| US.LIST2072 | 电子元件 | ALL | LIST2072 |
| US.LIST20745 | 日企美股精选 | ALL | LIST20745 |
| US.LIST20747 | 另类 | ALL | LIST20747 |
| US.LIST20748 | 债券 | ALL | LIST20748 |
| US.LIST20749 | 商品 | ALL | LIST20749 |
| US.LIST2075 | 消费电子产品 | ALL | LIST2075 |
| US.LIST20750 | 股票 | ALL | LIST20750 |
| US.LIST20751 | 混合资产 | ALL | LIST20751 |
| US.LIST20752 | 优先股 | ALL | LIST20752 |
| US.LIST20753 | 房地产 | ALL | LIST20753 |
| US.LIST20754 | 波动性 | ALL | LIST20754 |
| US.LIST20755 | 非必需消费品 | ALL | LIST20755 |
| US.LIST20756 | 必需消费品 | ALL | LIST20756 |
| US.LIST20757 | 能源 | ALL | LIST20757 |
| US.LIST20758 | 金融 | ALL | LIST20758 |
| US.LIST20759 | 健康护理 | ALL | LIST20759 |
| US.LIST20760 | 工业 | ALL | LIST20760 |
| US.LIST20761 | 原材料 | ALL | LIST20761 |
| US.LIST20762 | 房地产 | ALL | LIST20762 |
| US.LIST20763 | 科技 | ALL | LIST20763 |
| US.LIST20764 | 电信 | ALL | LIST20764 |
| US.LIST20765 | 公用事业 | ALL | LIST20765 |
| US.LIST20766 | 非洲 | ALL | LIST20766 |
| US.LIST20767 | 泛亚地区 | ALL | LIST20767 |
| US.LIST20768 | 亚太发达地区 | ALL | LIST20768 |
| US.LIST20769 | 欧洲发达地区 | ALL | LIST20769 |
| US.LIST20770 | 亚太新兴市场 | ALL | LIST20770 |
| US.LIST20771 | 欧洲新兴地区 | ALL | LIST20771 |
| US.LIST20772 | 新兴市场 | ALL | LIST20772 |
| US.LIST20773 | 前沿市场 | ALL | LIST20773 |
| US.LIST20774 | 全球(美国除外) | ALL | LIST20774 |
| US.LIST20775 | 全球 | ALL | LIST20775 |
| US.LIST20776 | 拉丁美洲 | ALL | LIST20776 |
| US.LIST20777 | 中东地区 | ALL | LIST20777 |
| US.LIST20778 | 北美地区 | ALL | LIST20778 |
| US.LIST20779 | 发达地区 | ALL | LIST20779 |
| US.LIST20780 | 货币 | ALL | LIST20780 |
| US.LIST2080 | 百货公司 | ALL | LIST2080 |
| US.LIST2083 | 纸及纸制品 | ALL | LIST2083 |
| US.LIST2088 | 电信服务 | ALL | LIST2088 |
| US.LIST20882 | 英伟达持仓概念 | ALL | LIST20882 |
| US.LIST20883 | 佩洛西持仓 | ALL | LIST20883 |
| US.LIST2089 | 航空航天与国防 | ALL | LIST2089 |
| US.LIST2090 | 航空服务 | ALL | LIST2090 |
| US.LIST2092 | 医药零售 | ALL | LIST2092 |
| US.LIST2093 | 白银 | ALL | LIST2093 |
| US.LIST2094 | 博彩 | ALL | LIST2094 |
| US.LIST2095 | 杂货店 | ALL | LIST2095 |
| US.LIST2098 | 通讯设备 | ALL | LIST2098 |
| US.LIST2101 | 钢 | ALL | LIST2101 |
| US.LIST2102 | 铁路运输 | ALL | LIST2102 |
| US.LIST21028 | 日本 | ALL | LIST21028 |
| US.LIST21030 | 加密货币现货ETF | ALL | LIST21030 |
| US.LIST21031 | 日本ETF | ALL | LIST21031 |
| US.LIST21032 | 半导体ETF | ALL | LIST21032 |
| US.LIST21033 | AI ETF | ALL | LIST21033 |
| US.LIST21034 | 基因组编辑ETF | ALL | LIST21034 |
| US.LIST21035 | 生物科技ETF | ALL | LIST21035 |
| US.LIST21037 | 自动驾驶ETF | ALL | LIST21037 |
| US.LIST21038 | 黄金ETF | ALL | LIST21038 |
| US.LIST21039 | 原油ETF | ALL | LIST21039 |
| US.LIST21040 | 亚太新兴市场ETF | ALL | LIST21040 |
| US.LIST21041 | 能源ETF | ALL | LIST21041 |
| US.LIST21042 | 北美地产ETF | ALL | LIST21042 |
| US.LIST21043 | 股息ETF | ALL | LIST21043 |
| US.LIST21044 | 云计算ETF | ALL | LIST21044 |
| US.LIST21045 | 互联网ETF | ALL | LIST21045 |
| US.LIST21056 | 加密货币期货ETF | ALL | LIST21056 |
| US.LIST2106 | 鞋类及配饰 | ALL | LIST2106 |
| US.LIST2108 | 包装食品 | ALL | LIST2108 |
| US.LIST2110 | 黄金 | ALL | LIST2110 |
| US.LIST2113 | CAR-T疗法 | ALL | LIST2113 |
| US.LIST2116 | 段永平持仓概念 | ALL | LIST2116 |
| US.LIST2118 | 阿里概念 | ALL | LIST2118 |
| US.LIST2124 | 内地教育股 | ALL | LIST2124 |
| US.LIST2136 | 人工智能 | ALL | LIST2136 |
| US.LIST2139 | 虚拟现实 | ALL | LIST2139 |
| US.LIST2140 | 住宅REITs | ALL | LIST2140 |
| US.LIST2141 | 零售REITs | ALL | LIST2141 |
| US.LIST2145 | 多元化REITs | ALL | LIST2145 |
| US.LIST2147 | 室温超导概念 | ALL | LIST2147 |
| US.LIST2149 | 脑机接口 | ALL | LIST2149 |
| US.LIST2152 | 减肥药概念 | ALL | LIST2152 |
| US.LIST2153 | ARK持仓 | ALL | LIST2153 |
| US.LIST2157 | 美国知名零售商 | ALL | LIST2157 |
| US.LIST2203 | 个性化服务 | ALL | LIST2203 |
| US.LIST2211 | 铝 | ALL | LIST2211 |
| US.LIST2214 | 休闲车 | ALL | LIST2214 |
| US.LIST2216 | 综合企业 | ALL | LIST2216 |
| US.LIST2219 | 废物管理 | ALL | LIST2219 |
| US.LIST2220 | 医疗分销 | ALL | LIST2220 |
| US.LIST2224 | 综合油气 | ALL | LIST2224 |
| US.LIST2225 | 餐厅 | ALL | LIST2225 |
| US.LIST2226 | 油气中游 | ALL | LIST2226 |
| US.LIST2227 | 污染控制与治理 | ALL | LIST2227 |
| US.LIST2230 | 保险经纪 | ALL | LIST2230 |
| US.LIST2234 | 机场和航空服务 | ALL | LIST2234 |
| US.LIST2237 | 包装与容器 | ALL | LIST2237 |
| US.LIST2240 | 折扣店 | ALL | LIST2240 |
| US.LIST2243 | 工具及配件 | ALL | LIST2243 |
| US.LIST2245 | 工业分销 | ALL | LIST2245 |
| US.LIST2246 | 医疗计划 | ALL | LIST2246 |
| US.LIST2249 | 资产管理 | ALL | LIST2249 |
| US.LIST2252 | 信息技术服务 | ALL | LIST2252 |
| US.LIST2253 | 电子游戏与多媒体 | ALL | LIST2253 |
| US.LIST2256 | 汽车和卡车经销 | ALL | LIST2256 |
| US.LIST2257 | 油气设备与服务 | ALL | LIST2257 |
| US.LIST2260 | 资本市场 | ALL | LIST2260 |
| US.LIST2261 | 信贷 | ALL | LIST2261 |
| US.LIST2262 | 健康信息服务 | ALL | LIST2262 |
| US.LIST2263 | 住宿 | ALL | LIST2263 |
| US.LIST2264 | 烟草 | ALL | LIST2264 |
| US.LIST2267 | 货运 | ALL | LIST2267 |
| US.LIST2268 | 安全与保护 | ALL | LIST2268 |
| US.LIST2270 | 金属加工 | ALL | LIST2270 |
| US.LIST2273 | 租赁服务 | ALL | LIST2273 |
| US.LIST2274 | 油气钻探 | ALL | LIST2274 |
| US.LIST2275 | 科技仪器 | ALL | LIST2275 |
| US.LIST2276 | 奢侈品 | ALL | LIST2276 |
| US.LIST2280 | 医疗设备 | ALL | LIST2280 |
| US.LIST22834 | 中国 | ALL | LIST22834 |
| US.LIST22860 | 月度派息ETF | ALL | LIST22860 |
| US.LIST22861 | 季度派息ETF | ALL | LIST22861 |
| US.LIST22862 | 年度派息ETF | ALL | LIST22862 |
| US.LIST22865 | 铜概念 | ALL | LIST22865 |
| US.LIST22866 | 美国国债ETF | ALL | LIST22866 |
| US.LIST22907 | 红海危机概念 | ALL | LIST22907 |
| US.LIST22908 | AI PC | ALL | LIST22908 |
| US.LIST22916 | 英伟达ETF | ALL | LIST22916 |
| US.LIST22927 | 美国降息利好概念 | ALL | LIST22927 |
| US.LIST22962 | 特朗普概念股 | ALL | LIST22962 |
| US.LIST22995 | 流行病诊疗 | ALL | LIST22995 |
| US.LIST22996 | 美国降息利好概念ETF | ALL | LIST22996 |
| US.LIST23407 | 高股息ETF | ALL | LIST23407 |
| US.LIST23428 | MicroStrategy ETFs | ALL | LIST23428 |
| US.LIST23444 | 圣诞节概念 | ALL | LIST23444 |
| US.LIST23447 | Bitcoin ETF with Options | ALL | LIST23447 |
| US.LIST23478 | 黑色星期五概念 | ALL | LIST23478 |
| US.LIST23479 | GraniteShares ETFs | ALL | LIST23479 |
| US.LIST23492 | AI应用软件股 | ALL | LIST23492 |
| US.LIST23494 | 加密货币现货ETF | ALL | LIST23494 |
| US.LIST23548 | 无人机概念股 | ALL | LIST23548 |
| US.LIST23560 | 标普500指数十大成分股 | ALL | LIST23560 |
| US.LIST23562 | AI眼镜概念股 | ALL | LIST23562 |
| US.LIST23563 | BATMMAAN | ALL | LIST23563 |
| US.LIST23566 | 加密货币ETF | ALL | LIST23566 |
| US.LIST23582 | 星际之门概念股 | ALL | LIST23582 |
| US.LIST23585 | DeepSeek概念股 | ALL | LIST23585 |
| US.LIST23588 | 体育博彩 | ALL | LIST23588 |
| US.LIST23590 | 智能驾驶概念股 | ALL | LIST23590 |
| US.LIST23592 | AI医疗概念股 | ALL | LIST23592 |
| US.LIST23596 | China's Terrific | ALL | LIST23596 |
| US.LIST23679 | 避险资产 | ALL | LIST23679 |
| US.LIST23700 | 稀土概念 | ALL | LIST23700 |
| US.LIST23702 | 稳定币概念 | ALL | LIST23702 |
| US.LIST23775 | 以太坊储备概念 | ALL | LIST23775 |
| US.LIST23797 | SOL储备概念 | ALL | LIST23797 |
| US.LIST23852 | 摩根JPMorgan ETF | ALL | LIST23852 |
| US.LIST23920 | 加密储备概念 | ALL | LIST23920 |
| US.LIST23921 | 加密矿企 | ALL | LIST23921 |
| US.LIST23925 | 存储概念股 | ALL | LIST23925 |
| US.LIST23939 | 储能概念股 | ALL | LIST23939 |
| US.LIST23979 | 光通信 | ALL | LIST23979 |
| US.LIST23987 | 锂矿概念股 | ALL | LIST23987 |
| US.LIST24027 | 预测市场 | ALL | LIST24027 |
| US.LIST24030 | 白银ETF | ALL | LIST24030 |
| US.LIST24035 | 白银概念 | ALL | LIST24035 |
| US.LIST24057 | 韩国ETF | ALL | LIST24057 |
| US.LIST24171 | 特朗普持仓概念 | ALL | LIST24171 |
| US.LIST24173 | 太空主题ETF | ALL | LIST24173 |
| US.LIST2426 | 家具、灯具与家电 | ALL | LIST2426 |
| US.LIST2427 | 啤酒 | ALL | LIST2427 |
| US.LIST2428 | 一般药品制造商 | ALL | LIST2428 |
| US.LIST2429 | 多元化保险 | ALL | LIST2429 |
| US.LIST2430 | 铀 | ALL | LIST2430 |
| US.LIST2431 | 互联网零售 | ALL | LIST2431 |
| US.LIST2432 | 流媒体概念 | ALL | LIST2432 |
| US.LIST2433 | 社交媒体 | ALL | LIST2433 |
| US.LIST2434 | 腾讯概念 | ALL | LIST2434 |
| US.LIST2435 | 在线教育 | ALL | LIST2435 |
| US.LIST2436 | 特斯拉概念 | ALL | LIST2436 |
| US.LIST2437 | 苹果概念 | ALL | LIST2437 |
| US.LIST2438 | 直播概念 | ALL | LIST2438 |
| US.LIST2439 | 搜索引擎 | ALL | LIST2439 |
| US.LIST2442 | 5G概念 | ALL | LIST2442 |
| US.LIST2443 | 挪威政府全球养老基金持仓 | ALL | LIST2443 |
| US.LIST2444 | 无人驾驶 | ALL | LIST2444 |
| US.LIST2447 | 邮轮概念 | ALL | LIST2447 |
| US.LIST2448 | OLED概念 | ALL | LIST2448 |
| US.LIST2450 | 光伏太阳能 | ALL | LIST2450 |
| US.LIST2452 | 美国基建股 | ALL | LIST2452 |
| US.LIST2456 | 地区银行 | ALL | LIST2456 |
| US.LIST2457 | 办公室REITs | ALL | LIST2457 |
| US.LIST2458 | 受监管自来水 | ALL | LIST2458 |
| US.LIST2459 | 非酒精饮料 | ALL | LIST2459 |
| US.LIST2460 | 家居装修零售 | ALL | LIST2460 |
| US.LIST2461 | 可再生能源公用事业 | ALL | LIST2461 |
| US.LIST2462 | 独立电力生产商 | ALL | LIST2462 |
| US.LIST2463 | 专用工业机械 | ALL | LIST2463 |
| US.LIST2464 | 酿酒厂 | ALL | LIST2464 |
| US.LIST2465 | 专门保险 | ALL | LIST2465 |
| US.LIST2466 | 工业REITs | ALL | LIST2466 |
| US.LIST2467 | 再保险 | ALL | LIST2467 |
| US.LIST2468 | 汽车制造 | ALL | LIST2468 |
| US.LIST2469 | 抵押贷款REITs | ALL | LIST2469 |
| US.LIST2470 | 应用软件 | ALL | LIST2470 |
| US.LIST2471 | 农业和重型机械 | ALL | LIST2471 |
| US.LIST2472 | 受监管电力 | ALL | LIST2472 |
| US.LIST2473 | 糖果 | ALL | LIST2473 |
| US.LIST2474 | 专业商务服务 | ALL | LIST2474 |
| US.LIST2475 | 酒店与旅馆REITs | ALL | LIST2475 |
| US.LIST2476 | 咨询服务 | ALL | LIST2476 |
| US.LIST2477 | 基础设施运营 | ALL | LIST2477 |
| US.LIST2478 | 食品分销 | ALL | LIST2478 |
| US.LIST2479 | 抵押贷款金融 | ALL | LIST2479 |
| US.LIST2480 | 多元化房地产 | ALL | LIST2480 |
| US.LIST2481 | 多元化银行 | ALL | LIST2481 |
| US.LIST2482 | 专业REITs | ALL | LIST2482 |
| US.LIST2483 | 建筑产品与设备 | ALL | LIST2483 |
| US.LIST2484 | 财产和意外伤害保险 | ALL | LIST2484 |
| US.LIST2485 | 海运 | ALL | LIST2485 |
| US.LIST2486 | 医疗保健设施 | ALL | LIST2486 |
| US.LIST2487 | 动力煤 | ALL | LIST2487 |
| US.LIST2488 | 多元化公用事业 | ALL | LIST2488 |
| US.LIST2489 | 受监管天然气 | ALL | LIST2489 |
| US.LIST2490 | 金融数据与证券交易所 | ALL | LIST2490 |
| US.LIST2491 | 娱乐 | ALL | LIST2491 |
| US.LIST2492 | 计算机硬件 | ALL | LIST2492 |
| US.LIST2493 | 电气设备及零件 | ALL | LIST2493 |
| US.LIST2494 | 服装零售 | ALL | LIST2494 |
| US.LIST2495 | 金融集团 | ALL | LIST2495 |
| US.LIST2496 | 人力资源与就业服务 | ALL | LIST2496 |
| US.LIST2497 | 广播 | ALL | LIST2497 |
| US.LIST2498 | 旅游服务 | ALL | LIST2498 |
| US.LIST2499 | 纺织制造 | ALL | LIST2499 |
| US.LIST2500 | 综合货运与物流 | ALL | LIST2500 |
| US.LIST2501 | 其他工业金属与采矿 | ALL | LIST2501 |
| US.LIST2502 | 专业零售 | ALL | LIST2502 |
| US.LIST2503 | 医疗设施REITs | ALL | LIST2503 |
| US.LIST2504 | 空壳公司 | ALL | LIST2504 |
| US.LIST2505 | 焦炭 | ALL | LIST2505 |
| US.LIST2506 | 电子与计算机分销 | ALL | LIST2506 |
| US.LIST2507 | 其他贵金属和采矿 | ALL | LIST2507 |
| US.LIST2508 | 软件基础设施 | ALL | LIST2508 |
| US.LIST2509 | 商业设备和用品 | ALL | LIST2509 |
| US.LIST2510 | 铜 | ALL | LIST2510 |
| US.LIST2511 | 房地产开发 | ALL | LIST2511 |
| US.LIST2512 | 人寿保险 | ALL | LIST2512 |
| US.LIST2513 | 专业与通用药品制造商 | ALL | LIST2513 |
| US.LIST2515 | 远程办公概念 | ALL | LIST2515 |
| US.LIST2517 | 热门中概股 | ALL | LIST2517 |
| US.LIST2518 | 明星科技股 | ALL | LIST2518 |
| US.LIST2520 | SaaS概念 | ALL | LIST2520 |
| US.LIST2521 | IDC概念 | ALL | LIST2521 |
| US.LIST2528 | 医药外包概念 | ALL | LIST2528 |
| US.LIST2538 | 外卖概念 | ALL | LIST2538 |
| US.LIST2539 | 激光雷达概念 | ALL | LIST2539 |
| US.LIST2540 | 云计算服务商 | ALL | LIST2540 |
| US.LIST2542 | 昨日强势股 | ALL | LIST2542 |
| US.LIST2544 | 固态电池 | ALL | LIST2544 |
| US.LIST2546 | 氢能源 | ALL | LIST2546 |
| US.LIST2547 | 充电桩 | ALL | LIST2547 |
| US.LIST2548 | AI芯片 | ALL | LIST2548 |
| US.LIST2551 | ARK ETF合集 | ALL | LIST2551 |
| US.LIST2552 | 锂电池 | ALL | LIST2552 |
| US.LIST2553 | 3D打印 | ALL | LIST2553 |
| US.LIST2555 | WSB热门概念 | ALL | LIST2555 |
| US.LIST2556 | 太空概念 | ALL | LIST2556 |
| US.LIST2561 | NFT概念 | ALL | LIST2561 |
| US.LIST2567 | 元宇宙概念 | ALL | LIST2567 |
| US.LIST2568 | 新能源汽车 | ALL | LIST2568 |
| US.LIST2569 | 基因编辑 | ALL | LIST2569 |
| US.LIST2570 | 网络安全概念 | ALL | LIST2570 |
| US.LIST2571 | ESG概念 | ALL | LIST2571 |
| US.LIST2572 | 可再生能源 | ALL | LIST2572 |
| US.LIST2573 | 抗癌医药 | ALL | LIST2573 |
| US.LIST2574 | 体育 | ALL | LIST2574 |
| US.LIST2575 | 品牌服饰 | ALL | LIST2575 |
| US.LIST2576 | 物联网 | ALL | LIST2576 |
| US.LIST2577 | 无线充电 | ALL | LIST2577 |
| US.LIST2579 | 智能家居 | ALL | LIST2579 |
| US.LIST2581 | 枪械制造 | ALL | LIST2581 |
| US.LIST2582 | 天然气 | ALL | LIST2582 |
| US.LIST2583 | 核电 | ALL | LIST2583 |
| US.LIST2584 | 人脸识别 | ALL | LIST2584 |
| US.LIST2585 | 页岩油 | ALL | LIST2585 |
| US.LIST2586 | 软件外包 | ALL | LIST2586 |
| US.LIST2587 | 信用支付 | ALL | LIST2587 |
| US.LIST2588 | 数字音乐 | ALL | LIST2588 |
| US.LIST2589 | 财产管理 | ALL | LIST2589 |
| US.LIST2591 | MSCI ESG ETFs | ALL | LIST2591 |
| US.LIST2594 | 量子计算概念 | ALL | LIST2594 |
| US.LIST2617 | 节日概念股 | ALL | LIST2617 |
| US.LIST2650 | 股息贵族 | ALL | LIST2650 |
| US.LIST2652 | 猴痘概念 | ALL | LIST2652 |
| US.LIST2653 | 机器人概念股 | ALL | LIST2653 |
| US.LIST2654 | 美国国防航空 | ALL | LIST2654 |
| US.LIST2655 | 区块链概念 | ALL | LIST2655 |
| US.LIST2657 | 数字支付 | ALL | LIST2657 |
| US.LIST2702 | 中国概念ETF | ALL | LIST2702 |
| US.LIST2982 | 美股新经济 | ALL | LIST2982 |
| US.LIST2997 | 大麻股 | ALL | LIST2997 |
| US.LIST2998 | 双十一 | ALL | LIST2998 |
| US.LIST2999 | 巴菲特持仓 | ALL | LIST2999 |
| US.LIST91361 | 大型价值股 | ALL | LIST91361 |
| US.LIST91362 | 大型均衡股 | ALL | LIST91362 |
| US.LIST91363 | 大型成长股 | ALL | LIST91363 |
| US.LIST91364 | 中型价值股 | ALL | LIST91364 |
| US.LIST91365 | 中型均衡股 | ALL | LIST91365 |
| US.LIST91366 | 中型成长股 | ALL | LIST91366 |
| US.LIST91367 | 小型价值股 | ALL | LIST91367 |
| US.LIST91368 | 小型均衡股 | ALL | LIST91368 |
| US.LIST91369 | 小型成长股 | ALL | LIST91369 |
| US.LIST91388 | 低信用长久期 | ALL | LIST91388 |
| US.LIST91389 | 中信用长久期 | ALL | LIST91389 |
| US.LIST91390 | 高信用长久期 | ALL | LIST91390 |
| US.LIST91391 | 低信用中久期 | ALL | LIST91391 |
| US.LIST91392 | 中信用中久期 | ALL | LIST91392 |
| US.LIST91393 | 高信用中久期 | ALL | LIST91393 |
| US.LIST91394 | 低信用短久期 | ALL | LIST91394 |
| US.LIST91395 | 中信用短久期 | ALL | LIST91395 |
| US.LIST91396 | 高信用短久期 | ALL | LIST91396 |

#### `CONCEPT` (116)

| code | name | plate_type | plate_id |
|---|---|---|---|
| US.LIST20010 | 加密货币概念股 | CONCEPT | LIST20010 |
| US.LIST20077 | 半导体精选 | CONCEPT | LIST20077 |
| US.LIST20745 | 日企美股精选 | CONCEPT | LIST20745 |
| US.LIST20882 | 英伟达持仓概念 | CONCEPT | LIST20882 |
| US.LIST20883 | 佩洛西持仓 | CONCEPT | LIST20883 |
| US.LIST2093 | 白银 | CONCEPT | LIST2093 |
| US.LIST2113 | CAR-T疗法 | CONCEPT | LIST2113 |
| US.LIST2116 | 段永平持仓概念 | CONCEPT | LIST2116 |
| US.LIST2118 | 阿里概念 | CONCEPT | LIST2118 |
| US.LIST2124 | 内地教育股 | CONCEPT | LIST2124 |
| US.LIST2136 | 人工智能 | CONCEPT | LIST2136 |
| US.LIST2139 | 虚拟现实 | CONCEPT | LIST2139 |
| US.LIST2147 | 室温超导概念 | CONCEPT | LIST2147 |
| US.LIST2149 | 脑机接口 | CONCEPT | LIST2149 |
| US.LIST2152 | 减肥药概念 | CONCEPT | LIST2152 |
| US.LIST2153 | ARK持仓 | CONCEPT | LIST2153 |
| US.LIST2157 | 美国知名零售商 | CONCEPT | LIST2157 |
| US.LIST22865 | 铜概念 | CONCEPT | LIST22865 |
| US.LIST22907 | 红海危机概念 | CONCEPT | LIST22907 |
| US.LIST22908 | AI PC | CONCEPT | LIST22908 |
| US.LIST22927 | 美国降息利好概念 | CONCEPT | LIST22927 |
| US.LIST22962 | 特朗普概念股 | CONCEPT | LIST22962 |
| US.LIST22995 | 流行病诊疗 | CONCEPT | LIST22995 |
| US.LIST23444 | 圣诞节概念 | CONCEPT | LIST23444 |
| US.LIST23478 | 黑色星期五概念 | CONCEPT | LIST23478 |
| US.LIST23492 | AI应用软件股 | CONCEPT | LIST23492 |
| US.LIST23548 | 无人机概念股 | CONCEPT | LIST23548 |
| US.LIST23562 | AI眼镜概念股 | CONCEPT | LIST23562 |
| US.LIST23563 | BATMMAAN | CONCEPT | LIST23563 |
| US.LIST23582 | 星际之门概念股 | CONCEPT | LIST23582 |
| US.LIST23585 | DeepSeek概念股 | CONCEPT | LIST23585 |
| US.LIST23588 | 体育博彩 | CONCEPT | LIST23588 |
| US.LIST23590 | 智能驾驶概念股 | CONCEPT | LIST23590 |
| US.LIST23592 | AI医疗概念股 | CONCEPT | LIST23592 |
| US.LIST23596 | China's Terrific | CONCEPT | LIST23596 |
| US.LIST23679 | 避险资产 | CONCEPT | LIST23679 |
| US.LIST23700 | 稀土概念 | CONCEPT | LIST23700 |
| US.LIST23702 | 稳定币概念 | CONCEPT | LIST23702 |
| US.LIST23775 | 以太坊储备概念 | CONCEPT | LIST23775 |
| US.LIST23797 | SOL储备概念 | CONCEPT | LIST23797 |
| US.LIST23920 | 加密储备概念 | CONCEPT | LIST23920 |
| US.LIST23921 | 加密矿企 | CONCEPT | LIST23921 |
| US.LIST23925 | 存储概念股 | CONCEPT | LIST23925 |
| US.LIST23939 | 储能概念股 | CONCEPT | LIST23939 |
| US.LIST23979 | 光通信 | CONCEPT | LIST23979 |
| US.LIST23987 | 锂矿概念股 | CONCEPT | LIST23987 |
| US.LIST24027 | 预测市场 | CONCEPT | LIST24027 |
| US.LIST24035 | 白银概念 | CONCEPT | LIST24035 |
| US.LIST24171 | 特朗普持仓概念 | CONCEPT | LIST24171 |
| US.LIST24173 | 太空主题ETF | CONCEPT | LIST24173 |
| US.LIST2432 | 流媒体概念 | CONCEPT | LIST2432 |
| US.LIST2433 | 社交媒体 | CONCEPT | LIST2433 |
| US.LIST2434 | 腾讯概念 | CONCEPT | LIST2434 |
| US.LIST2435 | 在线教育 | CONCEPT | LIST2435 |
| US.LIST2436 | 特斯拉概念 | CONCEPT | LIST2436 |
| US.LIST2437 | 苹果概念 | CONCEPT | LIST2437 |
| US.LIST2438 | 直播概念 | CONCEPT | LIST2438 |
| US.LIST2439 | 搜索引擎 | CONCEPT | LIST2439 |
| US.LIST2442 | 5G概念 | CONCEPT | LIST2442 |
| US.LIST2444 | 无人驾驶 | CONCEPT | LIST2444 |
| US.LIST2447 | 邮轮概念 | CONCEPT | LIST2447 |
| US.LIST2448 | OLED概念 | CONCEPT | LIST2448 |
| US.LIST2450 | 光伏太阳能 | CONCEPT | LIST2450 |
| US.LIST2452 | 美国基建股 | CONCEPT | LIST2452 |
| US.LIST2515 | 远程办公概念 | CONCEPT | LIST2515 |
| US.LIST2517 | 热门中概股 | CONCEPT | LIST2517 |
| US.LIST2518 | 明星科技股 | CONCEPT | LIST2518 |
| US.LIST2520 | SaaS概念 | CONCEPT | LIST2520 |
| US.LIST2521 | IDC概念 | CONCEPT | LIST2521 |
| US.LIST2528 | 医药外包概念 | CONCEPT | LIST2528 |
| US.LIST2538 | 外卖概念 | CONCEPT | LIST2538 |
| US.LIST2539 | 激光雷达概念 | CONCEPT | LIST2539 |
| US.LIST2540 | 云计算服务商 | CONCEPT | LIST2540 |
| US.LIST2542 | 昨日强势股 | CONCEPT | LIST2542 |
| US.LIST2544 | 固态电池 | CONCEPT | LIST2544 |
| US.LIST2546 | 氢能源 | CONCEPT | LIST2546 |
| US.LIST2547 | 充电桩 | CONCEPT | LIST2547 |
| US.LIST2548 | AI芯片 | CONCEPT | LIST2548 |
| US.LIST2551 | ARK ETF合集 | CONCEPT | LIST2551 |
| US.LIST2552 | 锂电池 | CONCEPT | LIST2552 |
| US.LIST2553 | 3D打印 | CONCEPT | LIST2553 |
| US.LIST2555 | WSB热门概念 | CONCEPT | LIST2555 |
| US.LIST2556 | 太空概念 | CONCEPT | LIST2556 |
| US.LIST2561 | NFT概念 | CONCEPT | LIST2561 |
| US.LIST2567 | 元宇宙概念 | CONCEPT | LIST2567 |
| US.LIST2568 | 新能源汽车 | CONCEPT | LIST2568 |
| US.LIST2569 | 基因编辑 | CONCEPT | LIST2569 |
| US.LIST2570 | 网络安全概念 | CONCEPT | LIST2570 |
| US.LIST2571 | ESG概念 | CONCEPT | LIST2571 |
| US.LIST2572 | 可再生能源 | CONCEPT | LIST2572 |
| US.LIST2573 | 抗癌医药 | CONCEPT | LIST2573 |
| US.LIST2574 | 体育 | CONCEPT | LIST2574 |
| US.LIST2575 | 品牌服饰 | CONCEPT | LIST2575 |
| US.LIST2576 | 物联网 | CONCEPT | LIST2576 |
| US.LIST2577 | 无线充电 | CONCEPT | LIST2577 |
| US.LIST2579 | 智能家居 | CONCEPT | LIST2579 |
| US.LIST2581 | 枪械制造 | CONCEPT | LIST2581 |
| US.LIST2582 | 天然气 | CONCEPT | LIST2582 |
| US.LIST2583 | 核电 | CONCEPT | LIST2583 |
| US.LIST2584 | 人脸识别 | CONCEPT | LIST2584 |
| US.LIST2585 | 页岩油 | CONCEPT | LIST2585 |
| US.LIST2586 | 软件外包 | CONCEPT | LIST2586 |
| US.LIST2587 | 信用支付 | CONCEPT | LIST2587 |
| US.LIST2588 | 数字音乐 | CONCEPT | LIST2588 |
| US.LIST2589 | 财产管理 | CONCEPT | LIST2589 |
| US.LIST2591 | MSCI ESG ETFs | CONCEPT | LIST2591 |
| US.LIST2594 | 量子计算概念 | CONCEPT | LIST2594 |
| US.LIST2650 | 股息贵族 | CONCEPT | LIST2650 |
| US.LIST2652 | 猴痘概念 | CONCEPT | LIST2652 |
| US.LIST2653 | 机器人概念股 | CONCEPT | LIST2653 |
| US.LIST2654 | 美国国防航空 | CONCEPT | LIST2654 |
| US.LIST2655 | 区块链概念 | CONCEPT | LIST2655 |
| US.LIST2657 | 数字支付 | CONCEPT | LIST2657 |
| US.LIST2997 | 大麻股 | CONCEPT | LIST2997 |
| US.LIST2998 | 双十一 | CONCEPT | LIST2998 |
| US.LIST2999 | 巴菲特持仓 | CONCEPT | LIST2999 |

#### `INDUSTRY` (145)

| code | name | plate_type | plate_id |
|---|---|---|---|
| US.LIST2003 | 家庭和个人用品 | INDUSTRY | LIST2003 |
| US.LIST2004 | 互联网内容与信息 | INDUSTRY | LIST2004 |
| US.LIST2005 | 住宅施工 | INDUSTRY | LIST2005 |
| US.LIST2007 | 农产品 | INDUSTRY | LIST2007 |
| US.LIST2008 | 出版 | INDUSTRY | LIST2008 |
| US.LIST2010 | 农业投入品 | INDUSTRY | LIST2010 |
| US.LIST2011 | 诊断与研究 | INDUSTRY | LIST2011 |
| US.LIST2014 | 医疗设备和用品 | INDUSTRY | LIST2014 |
| US.LIST2015 | 半导体 | INDUSTRY | LIST2015 |
| US.LIST2016 | 半导体设备与材料 | INDUSTRY | LIST2016 |
| US.LIST2020 | 化学制品 | INDUSTRY | LIST2020 |
| US.LIST2030 | 广告代理 | INDUSTRY | LIST2030 |
| US.LIST2033 | 建筑工程 | INDUSTRY | LIST2033 |
| US.LIST2034 | 建筑材料 | INDUSTRY | LIST2034 |
| US.LIST2038 | 房地产服务 | INDUSTRY | LIST2038 |
| US.LIST2044 | 教育培训 | INDUSTRY | LIST2044 |
| US.LIST2046 | 度假村与赌场 | INDUSTRY | LIST2046 |
| US.LIST2047 | 太阳能 | INDUSTRY | LIST2047 |
| US.LIST2049 | 服装制造 | INDUSTRY | LIST2049 |
| US.LIST2052 | 木材与木制品生产 | INDUSTRY | LIST2052 |
| US.LIST2055 | 汽车零件 | INDUSTRY | LIST2055 |
| US.LIST2058 | 油气勘探与开发 | INDUSTRY | LIST2058 |
| US.LIST2060 | 油气炼制与销售 | INDUSTRY | LIST2060 |
| US.LIST2063 | 休闲 | INDUSTRY | LIST2063 |
| US.LIST2068 | 特殊化工用品 | INDUSTRY | LIST2068 |
| US.LIST2069 | 生物技术 | INDUSTRY | LIST2069 |
| US.LIST2072 | 电子元件 | INDUSTRY | LIST2072 |
| US.LIST2075 | 消费电子产品 | INDUSTRY | LIST2075 |
| US.LIST2080 | 百货公司 | INDUSTRY | LIST2080 |
| US.LIST2083 | 纸及纸制品 | INDUSTRY | LIST2083 |
| US.LIST2088 | 电信服务 | INDUSTRY | LIST2088 |
| US.LIST2089 | 航空航天与国防 | INDUSTRY | LIST2089 |
| US.LIST2090 | 航空服务 | INDUSTRY | LIST2090 |
| US.LIST2092 | 医药零售 | INDUSTRY | LIST2092 |
| US.LIST2093 | 白银 | INDUSTRY | LIST2093 |
| US.LIST2094 | 博彩 | INDUSTRY | LIST2094 |
| US.LIST2095 | 杂货店 | INDUSTRY | LIST2095 |
| US.LIST2098 | 通讯设备 | INDUSTRY | LIST2098 |
| US.LIST2101 | 钢 | INDUSTRY | LIST2101 |
| US.LIST2102 | 铁路运输 | INDUSTRY | LIST2102 |
| US.LIST2106 | 鞋类及配饰 | INDUSTRY | LIST2106 |
| US.LIST2108 | 包装食品 | INDUSTRY | LIST2108 |
| US.LIST2110 | 黄金 | INDUSTRY | LIST2110 |
| US.LIST2140 | 住宅REITs | INDUSTRY | LIST2140 |
| US.LIST2141 | 零售REITs | INDUSTRY | LIST2141 |
| US.LIST2145 | 多元化REITs | INDUSTRY | LIST2145 |
| US.LIST2203 | 个性化服务 | INDUSTRY | LIST2203 |
| US.LIST2211 | 铝 | INDUSTRY | LIST2211 |
| US.LIST2214 | 休闲车 | INDUSTRY | LIST2214 |
| US.LIST2216 | 综合企业 | INDUSTRY | LIST2216 |
| US.LIST2219 | 废物管理 | INDUSTRY | LIST2219 |
| US.LIST2220 | 医疗分销 | INDUSTRY | LIST2220 |
| US.LIST2224 | 综合油气 | INDUSTRY | LIST2224 |
| US.LIST2225 | 餐厅 | INDUSTRY | LIST2225 |
| US.LIST2226 | 油气中游 | INDUSTRY | LIST2226 |
| US.LIST2227 | 污染控制与治理 | INDUSTRY | LIST2227 |
| US.LIST2230 | 保险经纪 | INDUSTRY | LIST2230 |
| US.LIST2234 | 机场和航空服务 | INDUSTRY | LIST2234 |
| US.LIST2237 | 包装与容器 | INDUSTRY | LIST2237 |
| US.LIST2240 | 折扣店 | INDUSTRY | LIST2240 |
| US.LIST2243 | 工具及配件 | INDUSTRY | LIST2243 |
| US.LIST2245 | 工业分销 | INDUSTRY | LIST2245 |
| US.LIST2246 | 医疗计划 | INDUSTRY | LIST2246 |
| US.LIST2249 | 资产管理 | INDUSTRY | LIST2249 |
| US.LIST2252 | 信息技术服务 | INDUSTRY | LIST2252 |
| US.LIST2253 | 电子游戏与多媒体 | INDUSTRY | LIST2253 |
| US.LIST2256 | 汽车和卡车经销 | INDUSTRY | LIST2256 |
| US.LIST2257 | 油气设备与服务 | INDUSTRY | LIST2257 |
| US.LIST2260 | 资本市场 | INDUSTRY | LIST2260 |
| US.LIST2261 | 信贷 | INDUSTRY | LIST2261 |
| US.LIST2262 | 健康信息服务 | INDUSTRY | LIST2262 |
| US.LIST2263 | 住宿 | INDUSTRY | LIST2263 |
| US.LIST2264 | 烟草 | INDUSTRY | LIST2264 |
| US.LIST2267 | 货运 | INDUSTRY | LIST2267 |
| US.LIST2268 | 安全与保护 | INDUSTRY | LIST2268 |
| US.LIST2270 | 金属加工 | INDUSTRY | LIST2270 |
| US.LIST2273 | 租赁服务 | INDUSTRY | LIST2273 |
| US.LIST2274 | 油气钻探 | INDUSTRY | LIST2274 |
| US.LIST2275 | 科技仪器 | INDUSTRY | LIST2275 |
| US.LIST2276 | 奢侈品 | INDUSTRY | LIST2276 |
| US.LIST2280 | 医疗设备 | INDUSTRY | LIST2280 |
| US.LIST2426 | 家具、灯具与家电 | INDUSTRY | LIST2426 |
| US.LIST2427 | 啤酒 | INDUSTRY | LIST2427 |
| US.LIST2428 | 一般药品制造商 | INDUSTRY | LIST2428 |
| US.LIST2429 | 多元化保险 | INDUSTRY | LIST2429 |
| US.LIST2430 | 铀 | INDUSTRY | LIST2430 |
| US.LIST2431 | 互联网零售 | INDUSTRY | LIST2431 |
| US.LIST2456 | 地区银行 | INDUSTRY | LIST2456 |
| US.LIST2457 | 办公室REITs | INDUSTRY | LIST2457 |
| US.LIST2458 | 受监管自来水 | INDUSTRY | LIST2458 |
| US.LIST2459 | 非酒精饮料 | INDUSTRY | LIST2459 |
| US.LIST2460 | 家居装修零售 | INDUSTRY | LIST2460 |
| US.LIST2461 | 可再生能源公用事业 | INDUSTRY | LIST2461 |
| US.LIST2462 | 独立电力生产商 | INDUSTRY | LIST2462 |
| US.LIST2463 | 专用工业机械 | INDUSTRY | LIST2463 |
| US.LIST2464 | 酿酒厂 | INDUSTRY | LIST2464 |
| US.LIST2465 | 专门保险 | INDUSTRY | LIST2465 |
| US.LIST2466 | 工业REITs | INDUSTRY | LIST2466 |
| US.LIST2467 | 再保险 | INDUSTRY | LIST2467 |
| US.LIST2468 | 汽车制造 | INDUSTRY | LIST2468 |
| US.LIST2469 | 抵押贷款REITs | INDUSTRY | LIST2469 |
| US.LIST2470 | 应用软件 | INDUSTRY | LIST2470 |
| US.LIST2471 | 农业和重型机械 | INDUSTRY | LIST2471 |
| US.LIST2472 | 受监管电力 | INDUSTRY | LIST2472 |
| US.LIST2473 | 糖果 | INDUSTRY | LIST2473 |
| US.LIST2474 | 专业商务服务 | INDUSTRY | LIST2474 |
| US.LIST2475 | 酒店与旅馆REITs | INDUSTRY | LIST2475 |
| US.LIST2476 | 咨询服务 | INDUSTRY | LIST2476 |
| US.LIST2477 | 基础设施运营 | INDUSTRY | LIST2477 |
| US.LIST2478 | 食品分销 | INDUSTRY | LIST2478 |
| US.LIST2479 | 抵押贷款金融 | INDUSTRY | LIST2479 |
| US.LIST2480 | 多元化房地产 | INDUSTRY | LIST2480 |
| US.LIST2481 | 多元化银行 | INDUSTRY | LIST2481 |
| US.LIST2482 | 专业REITs | INDUSTRY | LIST2482 |
| US.LIST2483 | 建筑产品与设备 | INDUSTRY | LIST2483 |
| US.LIST2484 | 财产和意外伤害保险 | INDUSTRY | LIST2484 |
| US.LIST2485 | 海运 | INDUSTRY | LIST2485 |
| US.LIST2486 | 医疗保健设施 | INDUSTRY | LIST2486 |
| US.LIST2487 | 动力煤 | INDUSTRY | LIST2487 |
| US.LIST2488 | 多元化公用事业 | INDUSTRY | LIST2488 |
| US.LIST2489 | 受监管天然气 | INDUSTRY | LIST2489 |
| US.LIST2490 | 金融数据与证券交易所 | INDUSTRY | LIST2490 |
| US.LIST2491 | 娱乐 | INDUSTRY | LIST2491 |
| US.LIST2492 | 计算机硬件 | INDUSTRY | LIST2492 |
| US.LIST2493 | 电气设备及零件 | INDUSTRY | LIST2493 |
| US.LIST2494 | 服装零售 | INDUSTRY | LIST2494 |
| US.LIST2495 | 金融集团 | INDUSTRY | LIST2495 |
| US.LIST2496 | 人力资源与就业服务 | INDUSTRY | LIST2496 |
| US.LIST2497 | 广播 | INDUSTRY | LIST2497 |
| US.LIST2498 | 旅游服务 | INDUSTRY | LIST2498 |
| US.LIST2499 | 纺织制造 | INDUSTRY | LIST2499 |
| US.LIST2500 | 综合货运与物流 | INDUSTRY | LIST2500 |
| US.LIST2501 | 其他工业金属与采矿 | INDUSTRY | LIST2501 |
| US.LIST2502 | 专业零售 | INDUSTRY | LIST2502 |
| US.LIST2503 | 医疗设施REITs | INDUSTRY | LIST2503 |
| US.LIST2504 | 空壳公司 | INDUSTRY | LIST2504 |
| US.LIST2505 | 焦炭 | INDUSTRY | LIST2505 |
| US.LIST2506 | 电子与计算机分销 | INDUSTRY | LIST2506 |
| US.LIST2507 | 其他贵金属和采矿 | INDUSTRY | LIST2507 |
| US.LIST2508 | 软件基础设施 | INDUSTRY | LIST2508 |
| US.LIST2509 | 商业设备和用品 | INDUSTRY | LIST2509 |
| US.LIST2510 | 铜 | INDUSTRY | LIST2510 |
| US.LIST2511 | 房地产开发 | INDUSTRY | LIST2511 |
| US.LIST2512 | 人寿保险 | INDUSTRY | LIST2512 |
| US.LIST2513 | 专业与通用药品制造商 | INDUSTRY | LIST2513 |

#### `REGION` (0)

| code | name | plate_type | plate_id |
|---|---|---|---|

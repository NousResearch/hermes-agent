# SZ 财务 Screener Options

- Category key: `financial`

Use fundamentals to filter quality and growth after the theme map is drafted.

## Choices

### 利润/收入/利润率/ROE/ROIC/现金流/资产负债/增长率/EPS

- Capability: `stock_filter`
- Stock filter type: `financial`

| enum | label |
|---|---|
| `NET_PROFIT` | 净利润 |
| `NET_PROFIX_GROWTH` | 净利润增长率 |
| `SUM_OF_BUSINESS` | 营业收入 |
| `SUM_OF_BUSINESS_GROWTH` | 营业收入增长率 |
| `NET_PROFIT_RATE` | 净利率 |
| `GROSS_PROFIT_RATE` | 毛利率 |
| `DEBT_ASSET_RATE` | 资产负债率 |
| `RETURN_ON_EQUITY_RATE` | ROE 净资产收益率 |
| `ROIC` | ROIC 投入资本回报率 |
| `ROA_TTM` | ROA TTM |
| `EBIT_TTM` | EBIT TTM |
| `EBITDA` | EBITDA |
| `OPERATING_MARGIN_TTM` | 营业利润率 TTM |
| `EBIT_MARGIN` | EBIT 利润率 |
| `EBITDA_MARGIN` | EBITDA 利润率 |
| `FINANCIAL_COST_RATE` | 财务费用率 |
| `OPERATING_PROFIT_TTM` | 营业利润 TTM |
| `SHAREHOLDER_NET_PROFIT_TTM` | 归母净利润 TTM |
| `NET_PROFIT_CASH_COVER_TTM` | 净利润现金含量 TTM |
| `CURRENT_RATIO` | 流动比率 |
| `QUICK_RATIO` | 速动比率 |
| `CURRENT_ASSET_RATIO` | 流动资产比率 |
| `CURRENT_DEBT_RATIO` | 流动负债比率 |
| `EQUITY_MULTIPLIER` | 权益乘数 |
| `PROPERTY_RATIO` | 产权比率 |
| `CASH_AND_CASH_EQUIVALENTS` | 现金及现金等价物 |
| `TOTAL_ASSET_TURNOVER` | 总资产周转率 |
| `FIXED_ASSET_TURNOVER` | 固定资产周转率 |
| `INVENTORY_TURNOVER` | 存货周转率 |
| `OPERATING_CASH_FLOW_TTM` | 经营现金流 TTM |
| `ACCOUNTS_RECEIVABLE` | 应收账款 |
| `EBIT_GROWTH_RATE` | EBIT 增长率 |
| `OPERATING_PROFIT_GROWTH_RATE` | 营业利润增长率 |
| `TOTAL_ASSETS_GROWTH_RATE` | 总资产增长率 |
| `PROFIT_TO_SHAREHOLDERS_GROWTH_RATE` | 归母净利润增长率 |
| `PROFIT_BEFORE_TAX_GROWTH_RATE` | 税前利润增长率 |
| `EPS_GROWTH_RATE` | EPS 增长率 |
| `ROE_GROWTH_RATE` | ROE 增长率 |
| `ROIC_GROWTH_RATE` | ROIC 增长率 |
| `NOCF_GROWTH_RATE` | 经营现金流增长率 |
| `NOCF_PER_SHARE_GROWTH_RATE` | 每股经营现金流增长率 |
| `OPERATING_REVENUE_CASH_COVER` | 营业收入现金含量 |
| `OPERATING_PROFIT_TO_TOTAL_PROFIT` | 营业利润占利润总额比例 |
| `BASIC_EPS` | 基本 EPS |
| `DILUTED_EPS` | 摊薄 EPS |
| `NOCF_PER_SHARE` | 每股经营现金流 |
- `quarters`: `ANNUAL`, `FIRST_QUARTER`, `INTERIM`, `THIRD_QUARTER`, `MOST_RECENT_QUARTER`
- LLM hint: Useful probes: revenue growth, gross margin, ROE/ROIC, operating cash flow, debt ratio. Exact numbers still need downstream SEC/fundamental validation.

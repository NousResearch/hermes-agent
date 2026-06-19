# SZ 行情 Screener Options

- Category key: `market_quote`

Use market/plate to constrain the universe, then quote and accumulated fields to find liquid, tradable, strong or weak candidates.

## Choices

### 交易所/市场

- Capability: `market_selector`
- `supported_values`: `US`, `HK`, `SH`, `SZ`
- LLM hint: Select the listing market. This is not a StockField.

### 所属行业/概念/板块

- Capability: `plate_selector`
- LLM hint: Probe theme words as plate_keywords. If a useful plate_code is found, stock_filter can filter within that plate_code.

### 价格/52周位置/量比/委比/每手价格

- Capability: `stock_filter`
- Stock filter type: `simple`

| enum | label |
|---|---|
| `CUR_PRICE` | 当前价格 |
| `CUR_PRICE_TO_HIGHEST52_WEEKS_RATIO` | 当前价相对52周最高比例 |
| `CUR_PRICE_TO_LOWEST52_WEEKS_RATIO` | 当前价相对52周最低比例 |
| `HIGH_PRICE_TO_HIGHEST52_WEEKS_RATIO` | 最高价相对52周最高比例 |
| `LOW_PRICE_TO_LOWEST52_WEEKS_RATIO` | 最低价相对52周最低比例 |
| `VOLUME_RATIO` | 量比 |
| `BID_ASK_RATIO` | 委比 |
| `LOT_PRICE` | 每手价格 |
- LLM hint: Use these for tradability, momentum context, and avoiding illiquid tails.

### 涨跌幅/振幅/成交量/成交额/换手率

- Capability: `stock_filter`
- Stock filter type: `accumulate`

| enum | label |
|---|---|
| `CHANGE_RATE` | 涨跌幅 |
| `AMPLITUDE` | 振幅 |
| `VOLUME` | 成交量 |
| `TURNOVER` | 成交额 |
| `TURNOVER_RATE` | 换手率 |
- LLM hint: Use days plus sort/filter bounds to discover strong momentum or high liquidity names, e.g. TURNOVER DESCEND or CHANGE_RATE over 20/60 days.

# SZ 技术 Screener Options

- Category key: `technical`

Use technical filters to discover trend candidates or confirm that a theme is active; avoid using them as the only reason a name belongs to a theme.

## Choices

### 指标解读

- Capability: `derived_interpretation`
- Alternate source: `derive from pattern/custom_indicator StockFields and K-line enrichment`
- LLM hint: The app-style interpretation layer is not a single OpenAPI enum. Use the specific MA/EMA/KDJ/RSI/MACD/BOLL fields below.

### MA/EMA 均线形态

- Capability: `stock_filter`
- Stock filter type: `pattern`

| enum | label |
|---|---|
| `MA_ALIGNMENT_LONG` | 均线多头排列 |
| `MA_ALIGNMENT_SHORT` | 均线空头排列 |
| `EMA_ALIGNMENT_LONG` | EMA 多头排列 |
| `EMA_ALIGNMENT_SHORT` | EMA 空头排列 |
- `supported_ktype`: `K_60M`, `K_DAY`, `K_WEEK`, `K_MON`
- LLM hint: Use for trend/momentum probes, e.g. MA_ALIGNMENT_LONG on K_DAY.

### RSI/KDJ/MACD/BOLL 形态

- Capability: `stock_filter`
- Stock filter type: `pattern`

| enum | label |
|---|---|
| `RSI_GOLD_CROSS_LOW` | RSI 低位金叉 |
| `RSI_DEATH_CROSS_HIGH` | RSI 高位死叉 |
| `RSI_TOP_DIVERGENCE` | RSI 顶背离 |
| `RSI_BOTTOM_DIVERGENCE` | RSI 底背离 |
| `KDJ_GOLD_CROSS_LOW` | KDJ 低位金叉 |
| `KDJ_DEATH_CROSS_HIGH` | KDJ 高位死叉 |
| `KDJ_TOP_DIVERGENCE` | KDJ 顶背离 |
| `KDJ_BOTTOM_DIVERGENCE` | KDJ 底背离 |
| `MACD_GOLD_CROSS_LOW` | MACD 低位金叉 |
| `MACD_DEATH_CROSS_HIGH` | MACD 高位死叉 |
| `MACD_TOP_DIVERGENCE` | MACD 顶背离 |
| `MACD_BOTTOM_DIVERGENCE` | MACD 底背离 |
| `BOLL_BREAK_UPPER` | BOLL 突破上轨 |
| `BOLL_BREAK_LOWER` | BOLL 跌破下轨 |
| `BOLL_CROSS_MIDDLE_UP` | BOLL 上穿中轨 |
| `BOLL_CROSS_MIDDLE_DOWN` | BOLL 下穿中轨 |
- `supported_ktype`: `K_60M`, `K_DAY`, `K_WEEK`, `K_MON`
- LLM hint: Use for oscillator and Bollinger-band signals when discovering active candidates.

### MA/EMA/KDJ/RSI/MACD/BOLL 自定义指标比较

- Capability: `stock_filter`
- Stock filter type: `custom_indicator`

| enum | label |
|---|---|
| `PRICE` | 价格 |
| `MA5` | MA5 五日均线 |
| `MA10` | MA10 十日均线 |
| `MA20` | MA20 二十日均线 |
| `MA30` | MA30 三十日均线 |
| `MA60` | MA60 六十日均线 |
| `MA120` | MA120 一百二十日均线 |
| `MA250` | MA250 二百五十日均线 |
| `RSI` | RSI 相对强弱指标 |
| `EMA5` | EMA5 |
| `EMA10` | EMA10 |
| `EMA20` | EMA20 |
| `EMA30` | EMA30 |
| `EMA60` | EMA60 |
| `EMA120` | EMA120 |
| `EMA250` | EMA250 |
| `VALUE` | 自定义数值 |
| `MA` | MA 均线 |
| `EMA` | EMA 指数移动均线 |
| `KDJ_K` | KDJ K值 |
| `KDJ_D` | KDJ D值 |
| `KDJ_J` | KDJ J值 |
| `MACD_DIFF` | MACD DIFF |
| `MACD_DEA` | MACD DEA |
| `MACD` | MACD 柱 |
| `BOLL_UPPER` | BOLL 上轨 |
| `BOLL_MIDDLER` | BOLL 中轨 |
| `BOLL_LOWER` | BOLL 下轨 |
- `supported_ktype`: `K_60M`, `K_DAY`, `K_WEEK`, `K_MON`
- `relative_position`: `MORE`, `LESS`, `CROSS_UP`, `CROSS_DOWN`
- LLM hint: Use only when a simple pattern field is insufficient.

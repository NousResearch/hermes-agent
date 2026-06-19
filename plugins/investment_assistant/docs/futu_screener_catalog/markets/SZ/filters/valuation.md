# SZ 估值 Screener Options

- Category key: `valuation`

Use valuation as a secondary screen after thematic relevance; do not let cheapness replace theme fit.

## Choices

### 市值/流通市值/股本

- Capability: `stock_filter`
- Stock filter type: `simple`

| enum | label |
|---|---|
| `MARKET_VAL` | 总市值 |
| `FLOAT_MARKET_VAL` | 流通市值 |
| `TOTAL_SHARE` | 总股本 |
| `FLOAT_SHARE` | 流通股本 |
- LLM hint: Use market cap to separate mega-cap anchors from small high-beta satellites.

### 市盈率（静态/TTM）/市净率/市销率/市现率

- Capability: `stock_filter`
- Stock filter type: `simple`

| enum | label |
|---|---|
| `PE_ANNUAL` | 市盈率 静态/年度 |
| `PE_TTM` | 市盈率 TTM |
| `PB_RATE` | 市净率 PB |
| `PS_TTM` | 市销率 PS TTM |
| `PCF_TTM` | 市现率 PCF TTM |
- LLM hint: Useful for valuation risk flags and relative comparison, not theme discovery alone.

### 估值分位/行业估值分位

- Capability: `derived_or_future_adapter`
- Alternate source: `derive from peer universe snapshots/financials; not exposed as current get_stock_filter StockField`
- LLM hint: Do not pass valuation percentile as stock_filter_specs. Mark it as desired downstream enrichment if needed.

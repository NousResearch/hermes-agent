# HK 分红 Screener Options

- Category key: `dividend`

Dividend data is enrichment, not a get_stock_filter field in this SDK.

## Choices

### TTM 分红/股息率

- Capability: `non_stock_filter`
- Alternate source: `get_market_snapshot fields dividend_ttm/dividend_ratio_ttm`
- LLM hint: Do not create stock_filter_specs for dividends; request quote snapshot enrichment later.

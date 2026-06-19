# Futu Screener Catalog Index

- Generated at: `2026-05-21T15:25:17.432515+00:00`
- Source: `manual_futu_screener_catalog_export`
- Futu SDK version: `10.05.6508`
- JSON snapshot: `/Users/roy_li/gits/hermes-agent/plugins/investment_assistant/data/futu_screener_catalog.json`
- Aggregate Markdown: `/Users/roy_li/gits/hermes-agent/plugins/investment_assistant/docs/futu_screener_catalog.md`
- Refresh command: `.venv/bin/python plugins/investment_assistant/scripts/export_futu_screener_catalog.py --markets US,HK,SH,SZ`

Use this directory as a local searchable catalog of Futu screener options. It contains actual OpenAPI StockField values, sorting enums, technical indicators, financial fields, and Futu plate code/name lists.

## Markets

- [HK](markets/HK/index.md)
- [SH](markets/SH/index.md)
- [SZ](markets/SZ/index.md)
- [US](markets/US/index.md)

## Search Hints

- For industry/concept membership, search `plates/concept.md`, `plates/industry.md`, or `plates/all.md`.
- For market cap and valuation fields, search `filters/valuation.md` and `filters/stock_fields/simple.md`.
- For MA/EMA/KDJ/RSI/MACD/BOLL, search `filters/technical.md`, `filters/stock_fields/pattern.md`, and `filters/stock_fields/custom_indicator.md`.
- For fundamentals, search `filters/financial.md` and `filters/stock_fields/financial.md`.
- Some Futu App choices are documented as non-OpenAPI enrichment when they are not exposed by `get_stock_filter`.

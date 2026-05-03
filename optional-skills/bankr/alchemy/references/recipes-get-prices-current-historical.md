---
id: references/recipes-get-prices-current-historical.md
name: 'Recipe: Get Current and Historical Prices'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - data-prices-api.md
updated: 2026-02-05
---
# Recipe: Get Current and Historical Prices

## Goal
Fetch current prices and historical price series for tokens.

## Inputs
- `symbols` (e.g., `ETH`, `BTC`)
- `startTime`, `endTime`
- `ALCHEMY_API_KEY`

## Steps
1. Query current prices by symbol.
2. Query historical prices for a time range.

## Example (Current)
```bash
curl -s "https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY/tokens/by-symbol?symbols=ETH&symbols=BTC"
```

## Example (Historical)
```bash
curl -s -X POST "https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY/tokens/historical" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"ETH","startTime":"2024-01-01T00:00:00Z","endTime":"2024-01-02T00:00:00Z"}'
```

## Output
- Spot prices and historical time series.

## Related Files
- `data-prices-api.md`

## Official Docs
- [Prices API Reference](https://www.alchemy.com/docs/reference/prices-api)

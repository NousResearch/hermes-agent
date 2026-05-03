---
id: agentic-gateway/references/data-prices-api.md
name: 'Prices API'
description: 'Query token prices for current and historical data via the Agentic Gateway.'
tags:
  - agentic-gateway
  - data-apis
  - data
updated: 2026-02-23
---
# Prices API

Query current and historical token prices. REST endpoints (GET and POST).

**Base URL**: `https://x402.alchemy.com/prices/v1/`

---

## `GET /tokens/by-symbol`

Get current prices by token symbol.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbols` | string (query, repeated) | Yes | One `symbols` param per token (max 25). Example: `?symbols=ETH&symbols=BTC&symbols=USDC` |

### Request

```bash
curl -s -H "Authorization: SIWE $TOKEN" \
  "https://x402.alchemy.com/prices/v1/tokens/by-symbol?symbols=ETH&symbols=BTC"
```

### Response

```json
{
  "data": [
    {
      "symbol": "ETH",
      "prices": [
        { "currency": "usd", "value": "1970.69", "lastUpdatedAt": "2025-06-01T12:00:00Z" }
      ]
    },
    {
      "symbol": "BTC",
      "prices": [
        { "currency": "usd", "value": "67727.36", "lastUpdatedAt": "2025-06-01T12:00:00Z" }
      ]
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `data` | array | List of token price objects |
| `data[].symbol` | string | Token symbol |
| `data[].prices` | array | Price entries (typically one per currency) |
| `data[].prices[].currency` | string | Currency code (e.g., `"usd"`) |
| `data[].prices[].value` | string | Price as decimal string |
| `data[].prices[].lastUpdatedAt` | string | ISO 8601 timestamp |
| `data[].error` | object | Error details if symbol not found (omitted on success) |

> **Note**: This endpoint does not return market cap, 24h volume, or
> percent change. To get market data, use `POST /tokens/historical`
> with `withMarketData: true`.

---

## `POST /tokens/by-address`

Get current prices by contract address. Must be POST with JSON body.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `addresses` | array (body) | Yes | Token address/network pairs (max 25 addresses, max 3 networks) |
| `addresses[].network` | string | Yes | Network slug (e.g., `"eth-mainnet"`) |
| `addresses[].address` | string | Yes | Token contract address |

### Request

```bash
curl -s -X POST "https://x402.alchemy.com/prices/v1/tokens/by-address" \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "addresses": [
      { "network": "eth-mainnet", "address": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" },
      { "network": "eth-mainnet", "address": "0xdac17f958d2ee523a2206206994597c13d831ec7" }
    ]
  }'
```

### Response

```json
{
  "data": [
    {
      "network": "eth-mainnet",
      "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "prices": [
        { "currency": "usd", "value": "0.9998", "lastUpdatedAt": "2025-06-01T12:00:00Z" }
      ]
    },
    {
      "network": "eth-mainnet",
      "address": "0xdac17f958d2ee523a2206206994597c13d831ec7",
      "prices": [
        { "currency": "usd", "value": "1.0001", "lastUpdatedAt": "2025-06-01T12:00:00Z" }
      ]
    }
  ]
}
```

---

## `POST /tokens/historical`

Get historical token prices with configurable intervals. Must be POST with JSON body.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Conditional | Token symbol (e.g., `"ETH"`). Use this OR `network`+`address`. |
| `network` | string | Conditional | Network slug. Use with `address` instead of `symbol`. |
| `address` | string | Conditional | Token contract address. Use with `network` instead of `symbol`. |
| `startTime` | ISO 8601 string or integer | Yes | Start time (e.g., `"2025-01-01T00:00:00Z"` or `1735689600`) |
| `endTime` | ISO 8601 string or integer | Yes | End time (e.g., `"2025-01-07T00:00:00Z"` or `1736294400`) |
| `interval` | string | No | `"5m"`, `"1h"`, or `"1d"` (default: `"1d"`) |
| `withMarketData` | boolean | No | Include market cap and volume (default: `false`) |

> **Important:** When using Unix timestamps, pass them as JSON numbers (e.g., `1704067200`), **not** as JSON strings (e.g., `"1704067200"`). String-wrapped Unix timestamps return a 400 error. ISO 8601 timestamps must be strings (e.g., `"2025-01-01T00:00:00Z"`).

**Max ranges**: `5m` → 7 days, `1h` → 30 days, `1d` → 1 year.

### Request

```bash
# Using ISO 8601 strings
curl -s -X POST "https://x402.alchemy.com/prices/v1/tokens/historical" \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "symbol": "ETH",
    "startTime": "2025-01-01T00:00:00Z",
    "endTime": "2025-01-07T00:00:00Z",
    "interval": "1h",
    "withMarketData": true
  }'

# Using Unix timestamps (must be JSON numbers, not strings)
curl -s -X POST "https://x402.alchemy.com/prices/v1/tokens/historical" \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "symbol": "ETH",
    "startTime": 1735689600,
    "endTime": 1736294400,
    "interval": "1h",
    "withMarketData": true
  }'
```

### Response

```json
{
  "symbol": "ETH",
  "currency": "usd",
  "data": [
    {
      "value": "3350.12",
      "timestamp": "2025-01-01T00:01:13Z",
      "marketCap": "274292310008.22",
      "totalVolume": "6715146404.61"
    },
    {
      "value": "3348.50",
      "timestamp": "2025-01-01T01:03:18Z",
      "marketCap": "274100000000.00",
      "totalVolume": "6700000000.00"
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `symbol` | string | Token symbol |
| `currency` | string | Currency code (e.g., `"usd"`) |
| `data` | array | Chronologically sorted price points |
| `data[].value` | string | Price as decimal string |
| `data[].timestamp` | string | ISO 8601 timestamp |
| `data[].marketCap` | string | Market cap (only if `withMarketData: true`) |
| `data[].totalVolume` | string | 24h volume (only if `withMarketData: true`) |

---

## Notes

- Not all tokens are supported. Check the `error` field in the response for unsupported tokens.
- Historical data returns single price points per interval, not OHLCV candles.
- Symbol-based endpoints are network-agnostic. Address-based endpoints require specifying the network.

## Official Docs
- [Prices API Quickstart](https://www.alchemy.com/docs/reference/prices-api-quickstart)
- [Token Prices by Symbol](https://www.alchemy.com/docs/data/prices-api/prices-api-endpoints/prices-api-endpoints/get-token-prices-by-symbol)
- [Token Prices by Address](https://www.alchemy.com/docs/data/prices-api/prices-api-endpoints/prices-api-endpoints/get-token-prices-by-address)
- [Historical Token Prices](https://www.alchemy.com/docs/data/prices-api/prices-api-endpoints/prices-api-endpoints/get-historical-token-prices)

---
id: agentic-gateway/references/data-token-api.md
name: 'Token API'
description: 'Fetch token balances, metadata, and allowances via the Agentic Gateway without manual contract calls. Token API methods are exposed as Alchemy JSON-RPC methods.'
tags:
  - agentic-gateway
  - data-apis
  - data
updated: 2026-02-23
---
# Token API

Fetch token balances, metadata, and allowances without manual contract calls. All methods are JSON-RPC POST requests.

**Base URL**: `https://x402.alchemy.com/{chainNetwork}/v2`

---

## `alchemy_getTokenBalances`

Returns ERC-20 token balances for an address.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `address` | string | Yes | — | Wallet address (hex) |
| `tokenSpec` | string or string[] | No | `"erc20"` | `"erc20"`, `"NATIVE_TOKEN"`, or array of contract addresses |
| `options.pageKey` | string | No | — | Pagination cursor from previous response |
| `options.maxCount` | integer | No | 100 | Max results per page (max 100) |

### Request

```bash
curl -s -X POST https://x402.alchemy.com/eth-mainnet/v2 \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_getTokenBalances",
    "params": [
      "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "erc20",
      { "maxCount": 5 }
    ]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
    "tokenBalances": [
      {
        "contractAddress": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "tokenBalance": "0x0000000000000000000000000000000000000000000000000000000005f5e100"
      },
      {
        "contractAddress": "0xdac17f958d2ee523a2206206994597c13d831ec7",
        "tokenBalance": "0x0000000000000000000000000000000000000000000000000000000000000000"
      }
    ],
    "pageKey": "0x..."
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `address` | string | The queried wallet address |
| `tokenBalances` | array | List of token balance objects |
| `tokenBalances[].contractAddress` | string | Token contract address |
| `tokenBalances[].tokenBalance` | string | Balance in hex (raw units, use token decimals to convert) |
| `pageKey` | string | Pagination cursor (absent if no more results) |

---

## `alchemy_getTokenMetadata`

Returns metadata for a token contract: name, symbol, decimals, and logo.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `contractAddress` | string | Yes | Token contract address (hex, 20 bytes) |

### Request

```bash
curl -s -X POST https://x402.alchemy.com/eth-mainnet/v2 \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_getTokenMetadata",
    "params": ["0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "name": "USD Coin",
    "symbol": "USDC",
    "decimals": 6,
    "logo": "https://static.alchemyapi.io/images/assets/3408.png"
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Token name |
| `symbol` | string | Token ticker symbol |
| `decimals` | integer | Number of decimal places |
| `logo` | string | URL to token logo image (may be null) |

---

## `alchemy_getTokenAllowance`

Returns the amount a spender is allowed to withdraw from an owner for a specific token.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `contract` | string | Yes | Token contract address |
| `owner` | string | Yes | Token owner address |
| `spender` | string | Yes | Approved spender address |

### Request

```bash
curl -s -X POST https://x402.alchemy.com/eth-mainnet/v2 \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_getTokenAllowance",
    "params": [{
      "contract": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "owner": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "spender": "0xdef1c0ded9bec7f1a1670819833240f027b25eff"
    }]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "0"
}
```

The result is a string representing the allowance in the token's smallest unit. `"0"` means no allowance.

---

## Notes

- Token balances are returned in raw hex. Divide by `10^decimals` (from `getTokenMetadata`) to get human-readable amounts.
- `getTokenBalances` may return zero balances for tokens the address has interacted with historically. Filter for non-zero if needed.
- Use `contractAddresses` array in `tokenSpec` to query specific tokens instead of scanning all ERC-20s.

## Official Docs
- [Token API Quickstart](https://www.alchemy.com/docs/reference/token-api-quickstart)
- [alchemy_getTokenBalances](https://www.alchemy.com/docs/data/token-api/token-api-endpoints/alchemy-get-token-balances)
- [alchemy_getTokenMetadata](https://www.alchemy.com/docs/reference/alchemy-gettokenmetadata)
- [alchemy_getTokenAllowance](https://www.alchemy.com/docs/data/token-api/token-api-endpoints/alchemy-get-token-allowance)

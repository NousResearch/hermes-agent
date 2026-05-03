---
id: agentic-gateway/references/data-transfers-api.md
name: 'Transfers API'
description: 'Query historical transfers across ERC-20/721/1155 and native transfers via the Agentic Gateway without manual log scanning.'
tags:
  - agentic-gateway
  - data-apis
  - data
updated: 2026-02-23
---
# Transfers API

Query historical transfers across ERC-20, ERC-721, ERC-1155, and native transfers without scanning the chain. JSON-RPC POST request.

**Base URL**: `https://x402.alchemy.com/{chainNetwork}/v2`

**Supported chains**: Ethereum, Base, Polygon, Arbitrum, Optimism (and testnets). `internal` category only available on Ethereum Mainnet and Polygon Mainnet.

---

## `alchemy_getAssetTransfers`

### Parameters

Accepts a single object with:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `fromBlock` | string | No | `"0x0"` | Start block (hex or `"latest"`) |
| `toBlock` | string | No | `"latest"` | End block (hex, `"latest"`, or `"indexed"`) |
| `fromAddress` | string | No | — | Filter by sender address |
| `toAddress` | string | No | — | Filter by recipient address |
| `category` | string[] | Yes | — | Transfer types: `"external"`, `"internal"`, `"erc20"`, `"erc721"`, `"erc1155"`, `"specialnft"` |
| `contractAddresses` | string[] | No | — | Filter by token contract addresses |
| `excludeZeroValue` | boolean | No | `true` | Exclude zero-value transfers |
| `order` | string | No | `"asc"` | `"asc"` or `"desc"` by block number |
| `withMetadata` | boolean | No | `false` | Include block timestamp in response |
| `maxCount` | string | No | `"0x3e8"` | Max results per page (hex, max 1000) |
| `pageKey` | string | No | — | Pagination cursor (10-minute TTL) |

### Request

```bash
curl -s -X POST https://x402.alchemy.com/eth-mainnet/v2 \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_getAssetTransfers",
    "params": [{
      "fromBlock": "0x0",
      "toBlock": "latest",
      "toAddress": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "category": ["external", "erc20"],
      "withMetadata": true,
      "maxCount": "0x5",
      "order": "desc"
    }]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "transfers": [
      {
        "blockNum": "0x1234abc",
        "uniqueId": "0x...:log:0",
        "hash": "0x3847245c01829b043431067fb2bfa95f7b5bdc7e...",
        "from": "0xef4396d9ff8107086d215a1c9f8866c54795d7c7",
        "to": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "value": 0.5,
        "asset": "ETH",
        "category": "external",
        "erc721TokenId": null,
        "erc1155Metadata": null,
        "tokenId": null,
        "rawContract": {
          "value": "0x6f05b59d3b20000",
          "address": null,
          "decimal": "0x12"
        },
        "metadata": {
          "blockTimestamp": "2024-06-15T10:30:00.000Z"
        }
      }
    ],
    "pageKey": "a1b2c3d4-e5f6-..."
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `transfers` | array | List of transfer objects |
| `transfers[].blockNum` | string | Block number (hex) |
| `transfers[].uniqueId` | string | Unique transfer identifier |
| `transfers[].hash` | string | Transaction hash |
| `transfers[].from` | string | Sender address |
| `transfers[].to` | string | Recipient address |
| `transfers[].value` | number | Transferred amount (human-readable, may be null for NFTs) |
| `transfers[].asset` | string | Asset symbol (e.g., `"ETH"`, `"USDC"`) |
| `transfers[].category` | string | Transfer type (`"external"`, `"erc20"`, etc.) |
| `transfers[].erc721TokenId` | string | ERC-721 token ID (null if not applicable) |
| `transfers[].erc1155Metadata` | array | ERC-1155 metadata (null if not applicable) |
| `transfers[].tokenId` | string | Generic token ID field |
| `transfers[].rawContract.value` | string | Raw transfer value (hex) |
| `transfers[].rawContract.address` | string | Token contract address (null for native) |
| `transfers[].rawContract.decimal` | string | Token decimals (hex) |
| `transfers[].metadata.blockTimestamp` | string | Block timestamp (ISO 8601, only if `withMetadata: true`) |
| `pageKey` | string | Cursor for next page (absent if no more results) |

---

## Notes

- `fromAddress` and `toAddress` are optional but at least one filter (address, contract, or block range) is recommended to avoid large result sets.
- `pageKey` has a 10-minute TTL. Fetch subsequent pages promptly.
- Use narrow block ranges for polling patterns to reduce compute unit usage.
- The `internal` category tracks internal EVM calls and is only available on Ethereum Mainnet and Polygon Mainnet.

## Official Docs
- [Transfers API Quickstart](https://www.alchemy.com/docs/reference/transfers-api-quickstart)
- [alchemy_getAssetTransfers](https://www.alchemy.com/docs/reference/alchemy-getassettransfers)

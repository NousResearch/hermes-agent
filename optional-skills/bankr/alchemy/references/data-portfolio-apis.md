---
id: agentic-gateway/references/data-portfolio-apis.md
name: 'Portfolio APIs'
description: 'Portfolio APIs provide consolidated wallet views (tokens, NFTs, and transaction history) across multiple networks via the Agentic Gateway.'
tags:
  - agentic-gateway
  - data-apis
  - data
related:
  - data-token-api.md
  - data-nft-api.md
  - data-transfers-api.md
updated: 2026-02-23
---
# Portfolio APIs

Consolidated wallet views (tokens, NFTs, transaction history) across multiple networks in single requests. All endpoints are `POST` with JSON bodies.

**Base URL**: `https://x402.alchemy.com/data/v1/`

**Supported chains** (in `networks` array): `eth-mainnet`, `base-mainnet`, `polygon-mainnet`, `arb-mainnet`, `opt-mainnet`, `sol-mainnet`, and more.

---

## `POST /assets/tokens/by-address`

Returns fungible tokens (native + ERC-20 + SPL) with balances, prices, and metadata for one or more wallets across multiple networks.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addresses` | array | Yes | — | Wallet/network pairs (max 2 addresses, max 5 networks each) |
| `addresses[].address` | string | Yes | — | Wallet address |
| `addresses[].networks` | string[] | Yes | — | Network slugs to query |
| `withMetadata` | boolean | No | `true` | Include token metadata (name, symbol, decimals, logo) |
| `withPrices` | boolean | No | `true` | Include token prices |
| `includeNativeTokens` | boolean | No | `true` | Include native tokens (ETH, MATIC, etc.) |
| `includeErc20Tokens` | boolean | No | `true` | Include ERC-20 tokens |
| `pageKey` | string | No | — | Pagination cursor |

### Request

```bash
curl -s -X POST "https://x402.alchemy.com/data/v1/assets/tokens/by-address" \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "addresses": [{
      "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "networks": ["eth-mainnet", "base-mainnet", "arb-mainnet"]
    }]
  }'
```

### Response

```json
{
  "data": {
    "tokens": [
      {
        "network": "eth-mainnet",
        "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "tokenAddress": null,
        "tokenBalance": "0x000000000000000000000000000000000000000000000001bdd951817729c4f9",
        "tokenMetadata": {
          "symbol": null,
          "decimals": null,
          "name": null,
          "logo": null
        },
        "tokenPrices": [
          {
            "currency": "usd",
            "value": "1970.69",
            "lastUpdatedAt": "2025-06-01T12:00:00Z"
          }
        ]
      },
      {
        "network": "eth-mainnet",
        "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "tokenAddress": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "tokenBalance": "0x000000000000000000000000000000000000000000000000000000003b9aca00",
        "tokenMetadata": {
          "name": "USD Coin",
          "symbol": "USDC",
          "decimals": 6,
          "logo": "https://..."
        },
        "tokenPrices": [
          {
            "currency": "usd",
            "value": "0.9998",
            "lastUpdatedAt": "2025-06-01T12:00:00Z"
          }
        ]
      }
    ],
    "pageKey": null
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `data.tokens` | array | Token objects |
| `data.tokens[].network` | string | Network slug |
| `data.tokens[].address` | string | Wallet address |
| `data.tokens[].tokenAddress` | string | Token contract address (`null` for native token) |
| `data.tokens[].tokenBalance` | string | Raw balance (hex, use token decimals to convert) |
| `data.tokens[].tokenMetadata` | object | Token name, symbol, decimals, logo |
| `data.tokens[].tokenPrices` | array | Price entries per currency (each has currency, value, lastUpdatedAt) |
| `data.pageKey` | string | Pagination cursor (null if no more results) |

---

## `POST /assets/tokens/balances/by-address`

Returns raw token balances (without prices/metadata). More efficient when you only need balances.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addresses` | array | Yes | — | Wallet/network pairs (max 3 pairs, max 20 networks total) |
| `addresses[].address` | string | Yes | — | Wallet address |
| `addresses[].networks` | string[] | Yes | — | Network slugs |
| `includeNativeTokens` | boolean | No | `true` | Include native tokens |
| `includeErc20Tokens` | boolean | No | `true` | Include ERC-20 tokens |
| `pageKey` | string | No | — | Pagination cursor |

### Request

```bash
curl -s -X POST "https://x402.alchemy.com/data/v1/assets/tokens/balances/by-address" \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "addresses": [{
      "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "networks": ["eth-mainnet", "base-mainnet", "arb-mainnet", "opt-mainnet", "polygon-mainnet"]
    }]
  }'
```

### Response

```json
{
  "data": {
    "tokens": [
      {
        "network": "eth-mainnet",
        "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "tokenAddress": null,
        "tokenBalance": "0x112210f47de98000"
      },
      {
        "network": "eth-mainnet",
        "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "tokenAddress": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "tokenBalance": "0x3b9aca00"
      }
    ],
    "pageKey": null
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `data.tokens[].network` | string | Network slug |
| `data.tokens[].address` | string | Wallet address |
| `data.tokens[].tokenAddress` | string | Token contract address (`null` for native token) |
| `data.tokens[].tokenBalance` | string | Raw balance (hex) |
| `data.pageKey` | string | Pagination cursor |

---

## `POST /assets/nfts/by-address`

Returns NFTs owned by a wallet across multiple networks.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addresses` | array | Yes | — | Wallet/network pairs (max 2 pairs, max 15 networks each) |
| `addresses[].address` | string | Yes | — | Wallet address |
| `addresses[].networks` | string[] | Yes | — | Network slugs |
| `excludeFilters` | string[] | No | `["SPAM"]` | Exclude `"SPAM"`, `"AIRDROPS"` |
| `includeFilters` | string[] | No | — | Include only `"SPAM"`, `"AIRDROPS"` (mutually exclusive with excludeFilters) |
| `spamConfidenceLevel` | string | No | — | `"VERY_HIGH"`, `"HIGH"`, `"MEDIUM"`, `"LOW"` |
| `withMetadata` | boolean | No | `true` | Include NFT metadata |
| `pageKey` | string | No | — | Pagination cursor |
| `pageSize` | integer | No | `100` | Results per page (max 100) |
| `orderBy` | string | No | — | `"transferTime"` |
| `sortOrder` | string | No | — | `"asc"` or `"desc"` |

### Request

```bash
curl -s -X POST "https://x402.alchemy.com/data/v1/assets/nfts/by-address" \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "addresses": [{
      "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "networks": ["eth-mainnet"]
    }],
    "withMetadata": true,
    "pageSize": 2
  }'
```

### Response

```json
{
  "data": {
    "ownedNfts": [
      {
        "network": "eth-mainnet",
        "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "contract": {
          "address": "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d",
          "name": "BoredApeYachtClub",
          "symbol": "BAYC",
          "tokenType": "ERC721",
          "openSeaMetadata": { "collectionName": "Bored Ape Yacht Club", "floorPrice": 10.5 },
          "isSpam": false
        },
        "tokenId": "1234",
        "tokenType": "ERC721",
        "name": "Bored Ape #1234",
        "description": "...",
        "image": { "cachedUrl": "https://nft-cdn.alchemy.com/..." },
        "collection": { "name": "Bored Ape Yacht Club" },
        "acquiredAt": { "blockTimestamp": "2024-03-15T10:30:00.000Z" }
      }
    ],
    "totalCount": 42,
    "pageKey": "..."
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `data.ownedNfts` | array | NFT objects (same structure as NFT API's `getNFTsForOwner` plus `network` and `address` fields) |
| `data.ownedNfts[].network` | string | Network slug |
| `data.ownedNfts[].address` | string | Wallet address |
| `data.ownedNfts[].contract` | object | Contract info (address, name, symbol, tokenType, openSeaMetadata, isSpam) |
| `data.ownedNfts[].tokenId` | string | Token ID |
| `data.ownedNfts[].name` | string | NFT name |
| `data.ownedNfts[].image` | object | Image URLs |
| `data.ownedNfts[].acquiredAt` | object | Acquisition timestamp |
| `data.totalCount` | integer | Total NFTs across all pages |
| `data.pageKey` | string | Pagination cursor |

---

## `POST /assets/nfts/contracts/by-address`

Returns NFT collections (contracts) owned by a wallet across multiple networks.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addresses` | array | Yes | — | Wallet/network pairs (max 2 pairs, max 15 networks each) |
| `addresses[].address` | string | Yes | — | Wallet address |
| `addresses[].networks` | string[] | Yes | — | Network slugs |
| `withMetadata` | boolean | No | `true` | Include contract metadata |
| `pageKey` | string | No | — | Pagination cursor |
| `pageSize` | integer | No | `100` | Results per page (max 100) |
| `orderBy` | string | No | — | `"transferTime"` |
| `sortOrder` | string | No | — | `"asc"` or `"desc"` |

### Request

```bash
curl -s -X POST "https://x402.alchemy.com/data/v1/assets/nfts/contracts/by-address" \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "addresses": [{
      "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "networks": ["eth-mainnet"]
    }],
    "withMetadata": true
  }'
```

### Response

```json
{
  "data": {
    "contracts": [
      {
        "network": "eth-mainnet",
        "address": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "contract": {
          "address": "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d",
          "name": "BoredApeYachtClub",
          "symbol": "BAYC",
          "totalSupply": "10000",
          "tokenType": "ERC721",
          "contractDeployer": "0xaba7161a7fb69c88e16ed9f455ce62b791ee4d03",
          "deployedBlockNumber": 12287507,
          "openSeaMetadata": {
            "collectionName": "Bored Ape Yacht Club",
            "floorPrice": 10.5
          },
          "isSpam": false,
          "totalBalance": "1",
          "numDistinctTokensOwned": "1",
          "displayNft": {
            "tokenId": "1234",
            "name": "Bored Ape #1234"
          },
          "image": {
            "cachedUrl": "https://nft-cdn.alchemy.com/...",
            "thumbnailUrl": "https://nft-cdn.alchemy.com/...",
            "pngUrl": "https://nft-cdn.alchemy.com/...",
            "contentType": "image/png",
            "originalUrl": "ipfs://..."
          }
        }
      }
    ],
    "totalCount": 5,
    "pageKey": null
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `data.contracts` | array | Contract objects |
| `data.contracts[].network` | string | Network slug |
| `data.contracts[].address` | string | Wallet address |
| `data.contracts[].contract` | object | Contract info (same as NFT API `getContractMetadata`) |
| `data.contracts[].contract.totalBalance` | string | Total NFTs owned from this contract |
| `data.contracts[].contract.numDistinctTokensOwned` | string | Number of distinct token IDs owned |
| `data.contracts[].contract.displayNft` | object | Representative NFT (tokenId, name) |
| `data.contracts[].contract.image` | object | Image URLs for the display NFT |
| `data.totalCount` | integer | Total collections |
| `data.pageKey` | string | Pagination cursor |

---

## `POST /transactions/history/by-address` (Beta)

Returns transaction history for a wallet across networks.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `addresses` | array | Yes | — | Wallet/network pairs (max 1 pair, max 2 networks) |
| `addresses[].address` | string | Yes | — | Wallet address |
| `addresses[].networks` | string[] | Yes | — | Network slugs (beta: `eth-mainnet`, `base-mainnet` only) |
| `before` | string | No | — | Cursor for previous page |
| `after` | string | No | — | Cursor for next page |
| `limit` | integer | No | `25` | Results per page (max 50) |

### Request

```bash
curl -s -X POST "https://x402.alchemy.com/data/v1/transactions/history/by-address" \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "addresses": [{
      "address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "networks": ["eth-mainnet"]
    }],
    "limit": 5
  }'
```

### Response

```json
{
  "transactions": [
    {
      "network": "eth-mainnet",
      "hash": "0x3847245c01829b043431067fb2bfa95f7b5bdc7e...",
      "timeStamp": "2025-06-01T10:30:00.000Z",
      "blockNumber": "20000000",
      "blockHash": "0x...",
      "nonce": "42",
      "transactionIndex": "5",
      "fromAddress": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
      "toAddress": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "contractAddress": null,
      "value": "0",
      "gasUsed": "46000",
      "effectiveGasPrice": "15000000000",
      "cumulativeGasUsed": "1234567",
      "logs": [],
      "internalTxns": []
    }
  ],
  "before": "cursor_prev...",
  "after": "cursor_next...",
  "totalCount": 500
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `transactions` | array | Transaction objects |
| `transactions[].network` | string | Network slug |
| `transactions[].hash` | string | Transaction hash |
| `transactions[].timeStamp` | string | ISO 8601 timestamp |
| `transactions[].blockNumber` | string | Block number |
| `transactions[].fromAddress` | string | Sender address |
| `transactions[].toAddress` | string | Recipient address |
| `transactions[].contractAddress` | string | Deployed contract address (null if not a deployment) |
| `transactions[].value` | string | ETH value transferred |
| `transactions[].gasUsed` | string | Gas used |
| `transactions[].effectiveGasPrice` | string | Effective gas price |
| `transactions[].logs` | array | Event logs |
| `transactions[].internalTxns` | array | Internal transactions |
| `before` | string | Cursor for previous page |
| `after` | string | Cursor for next page |
| `totalCount` | integer | Total transactions |

---

## Notes

- Use the smallest `networks` list you need — each network adds to response time.
- NFT metadata can be missing or malformed. Always handle null fields.
- The Transactions endpoint is in beta and only supports Ethereum and Base mainnets.
- Paginate large wallets using `pageKey` (for asset endpoints) or `before`/`after` (for transactions).

## Official Docs
- [Portfolio APIs Overview](https://www.alchemy.com/docs/reference/portfolio-apis)
- [Tokens by Address](https://www.alchemy.com/docs/data/portfolio-apis/portfolio-api-endpoints/portfolio-api-endpoints/get-tokens-by-address)
- [Token Balances by Address](https://www.alchemy.com/docs/data/portfolio-apis/portfolio-api-endpoints/portfolio-api-endpoints/get-token-balances-by-address)
- [NFTs by Address](https://www.alchemy.com/docs/data/portfolio-apis/portfolio-api-endpoints/portfolio-api-endpoints/get-nfts-by-address)
- [NFT Contracts by Address](https://www.alchemy.com/docs/data/portfolio-apis/portfolio-api-endpoints/portfolio-api-endpoints/get-nft-contracts-by-address)
- [Transaction History by Address (Beta)](https://www.alchemy.com/docs/data/beta-apis/beta-api-endpoints/beta-api-endpoints/get-transaction-history-by-address)

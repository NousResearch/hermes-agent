---
id: agentic-gateway/references/data-nft-api.md
name: 'NFT API'
description: 'Query NFT ownership, metadata, collections, and contract-level info via the Agentic Gateway.'
tags:
  - agentic-gateway
  - data-apis
  - data
updated: 2026-02-23
---
# NFT API

Query NFT ownership, metadata, collections, and contract-level info. REST endpoints (GET).

**Base URL**: `https://x402.alchemy.com/{chainNetwork}/nft/v3/`

**Supported chains**: Ethereum, Base, Polygon, Arbitrum, Optimism, BNB, and testnets.

---

## `GET /getNFTsForOwner`

Returns all NFTs owned by a given address.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `owner` | string (query) | Yes | — | Wallet address |
| `contractAddresses[]` | string[] (query) | No | — | Filter by contract addresses (max 45) |
| `withMetadata` | boolean (query) | No | `true` | Include metadata in response |
| `orderBy` | string (query) | No | — | `"transferTime"` to sort by acquisition time |
| `excludeFilters[]` | string[] (query) | No | — | `"SPAM"`, `"AIRDROPS"` |
| `includeFilters[]` | string[] (query) | No | — | `"SPAM"`, `"AIRDROPS"` |
| `spamConfidenceLevel` | string (query) | No | — | `"VERY_HIGH"`, `"HIGH"`, `"MEDIUM"`, `"LOW"` |
| `tokenUriTimeoutInMs` | integer (query) | No | — | Timeout for token URI resolution |
| `pageKey` | string (query) | No | — | Pagination cursor |
| `pageSize` | integer (query) | No | `100` | Results per page (max 100) |

### Request

```bash
curl -s -H "Authorization: SIWE $TOKEN" \
  "https://x402.alchemy.com/eth-mainnet/nft/v3/getNFTsForOwner?owner=0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045&withMetadata=true&pageSize=2"
```

### Response

```json
{
  "ownedNfts": [
    {
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
          "collectionSlug": "boredapeyachtclub",
          "safelistRequestStatus": "verified",
          "imageUrl": "https://...",
          "description": "...",
          "floorPrice": 10.5
        },
        "isSpam": false,
        "spamClassifications": []
      },
      "tokenId": "1234",
      "tokenType": "ERC721",
      "name": "Bored Ape #1234",
      "description": "...",
      "tokenUri": "ipfs://Qm.../1234",
      "image": {
        "cachedUrl": "https://nft-cdn.alchemy.com/...",
        "thumbnailUrl": "https://nft-cdn.alchemy.com/...",
        "pngUrl": "https://nft-cdn.alchemy.com/...",
        "contentType": "image/png",
        "size": 123456,
        "originalUrl": "ipfs://..."
      },
      "animation": {
        "cachedUrl": null,
        "contentType": null,
        "size": null,
        "originalUrl": null
      },
      "raw": {
        "tokenUri": "ipfs://Qm.../1234",
        "metadata": { "name": "Bored Ape #1234", "image": "ipfs://...", "attributes": [] },
        "error": null
      },
      "collection": {
        "name": "Bored Ape Yacht Club",
        "slug": "boredapeyachtclub",
        "externalUrl": "https://boredapeyachtclub.com",
        "bannerImageUrl": "https://..."
      },
      "mint": {
        "mintAddress": null,
        "blockNumber": null,
        "timestamp": null,
        "transactionHash": null
      },
      "owners": null,
      "timeLastUpdated": "2025-06-01T12:00:00.000Z",
      "balance": "1",
      "acquiredAt": {
        "blockTimestamp": "2024-03-15T10:30:00.000Z",
        "blockNumber": "19445200"
      }
    }
  ],
  "totalCount": 42,
  "validAt": {
    "blockNumber": "20000000",
    "blockHash": "0x...",
    "blockTimestamp": "2025-06-01T12:00:00.000Z"
  },
  "pageKey": "abc123..."
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `ownedNfts` | array | List of NFT objects |
| `ownedNfts[].contract.address` | string | Contract address |
| `ownedNfts[].contract.name` | string | Collection name |
| `ownedNfts[].contract.symbol` | string | Collection symbol |
| `ownedNfts[].contract.tokenType` | string | `"ERC721"` or `"ERC1155"` |
| `ownedNfts[].contract.openSeaMetadata` | object | OpenSea collection data (name, slug, floor price, etc.) |
| `ownedNfts[].contract.isSpam` | boolean | Whether contract is flagged as spam |
| `ownedNfts[].tokenId` | string | Token ID |
| `ownedNfts[].tokenType` | string | `"ERC721"` or `"ERC1155"` |
| `ownedNfts[].name` | string | NFT name (from metadata) |
| `ownedNfts[].description` | string | NFT description |
| `ownedNfts[].tokenUri` | string | Resolved token URI |
| `ownedNfts[].image` | object | Image URLs (cachedUrl, thumbnailUrl, pngUrl, originalUrl) |
| `ownedNfts[].animation` | object | Animation URLs (cachedUrl, contentType, size, originalUrl) |
| `ownedNfts[].raw.metadata` | object | Raw metadata JSON |
| `ownedNfts[].collection` | object | Collection info (name, slug, externalUrl) |
| `ownedNfts[].mint` | object | Minting info (mintAddress, blockNumber, timestamp, transactionHash) |
| `ownedNfts[].owners` | array | Current owners (may be null) |
| `ownedNfts[].balance` | string | Token balance (always `"1"` for ERC-721) |
| `ownedNfts[].acquiredAt` | object | Acquisition timestamp and block number |
| `totalCount` | integer | Total NFTs owned (across all pages) |
| `validAt` | object | Block at which data is valid |
| `pageKey` | string | Cursor for next page (absent if no more results) |

---

## `GET /getNFTMetadata`

Returns metadata for a single NFT.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `contractAddress` | string (query) | Yes | — | NFT contract address |
| `tokenId` | string (query) | Yes | — | Token ID |
| `tokenType` | string (query) | No | — | `"ERC721"` or `"ERC1155"` (auto-detected if omitted) |
| `tokenUriTimeoutInMs` | integer (query) | No | — | Timeout for token URI resolution |
| `refreshCache` | boolean (query) | No | `false` | Force metadata refresh |

### Request

```bash
curl -s -H "Authorization: SIWE $TOKEN" \
  "https://x402.alchemy.com/eth-mainnet/nft/v3/getNFTMetadata?contractAddress=0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d&tokenId=1"
```

### Response

```json
{
  "contract": {
    "address": "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d",
    "name": "BoredApeYachtClub",
    "symbol": "BAYC",
    "tokenType": "ERC721",
    "openSeaMetadata": { "collectionName": "Bored Ape Yacht Club", "floorPrice": 10.5 },
    "isSpam": false
  },
  "tokenId": "1",
  "tokenType": "ERC721",
  "name": "Bored Ape #1",
  "description": "...",
  "tokenUri": "ipfs://QmeSjSinHpPnmXmspMjwiXyN6zS4E9zccariGR3jxcaWtq/1",
  "image": {
    "cachedUrl": "https://nft-cdn.alchemy.com/...",
    "thumbnailUrl": "https://nft-cdn.alchemy.com/...",
    "pngUrl": "https://nft-cdn.alchemy.com/...",
    "contentType": "image/png",
    "originalUrl": "ipfs://..."
  },
  "animation": {
    "cachedUrl": null,
    "contentType": null,
    "size": null,
    "originalUrl": null
  },
  "raw": {
    "tokenUri": "ipfs://QmeSjSinHpPnmXmspMjwiXyN6zS4E9zccariGR3jxcaWtq/1",
    "metadata": {
      "name": "Bored Ape #1",
      "image": "ipfs://...",
      "attributes": [
        { "trait_type": "Fur", "value": "Robot" },
        { "trait_type": "Eyes", "value": "X Eyes" }
      ]
    }
  },
  "collection": { "name": "Bored Ape Yacht Club", "slug": "boredapeyachtclub" },
  "mint": {
    "mintAddress": null,
    "blockNumber": null,
    "timestamp": null,
    "transactionHash": null
  },
  "owners": ["0x46efbaedc92067e6d60e84ed6395099723252496"],
  "timeLastUpdated": "2025-06-01T12:00:00.000Z"
}
```

### Response Fields

Same structure as a single item in `getNFTsForOwner.ownedNfts[]`. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `contract` | object | Contract info (address, name, symbol, tokenType, openSeaMetadata, isSpam) |
| `tokenId` | string | Token ID |
| `name` | string | NFT name |
| `description` | string | NFT description |
| `tokenUri` | string | Resolved token URI |
| `image` | object | Image URLs and content type |
| `raw.metadata` | object | Raw metadata JSON including attributes array |
| `collection` | object | Collection info |
| `animation` | object | Animation URLs (cachedUrl, contentType, size, originalUrl) |
| `mint` | object | Minting info (mintAddress, blockNumber, timestamp, transactionHash) |
| `owners` | array | Current owners (may be null) |
| `timeLastUpdated` | string | ISO 8601 timestamp of last metadata update |

---

## `GET /getContractMetadata`

Returns metadata for an NFT contract/collection.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `contractAddress` | string (query) | Yes | NFT contract address |

### Request

```bash
curl -s -H "Authorization: SIWE $TOKEN" \
  "https://x402.alchemy.com/eth-mainnet/nft/v3/getContractMetadata?contractAddress=0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d"
```

### Response

```json
{
  "address": "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d",
  "name": "BoredApeYachtClub",
  "symbol": "BAYC",
  "totalSupply": "10000",
  "tokenType": "ERC721",
  "contractDeployer": "0xaba7161a7fb69c88e16ed9f455ce62b791ee4d03",
  "deployedBlockNumber": 12287507,
  "openSeaMetadata": {
    "collectionName": "Bored Ape Yacht Club",
    "collectionSlug": "boredapeyachtclub",
    "safelistRequestStatus": "verified",
    "imageUrl": "https://...",
    "description": "...",
    "externalUrl": "https://boredapeyachtclub.com",
    "twitterUsername": "BoredApeYC",
    "discordUrl": "https://discord.gg/...",
    "bannerImageUrl": "https://...",
    "floorPrice": 10.5
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `address` | string | Contract address |
| `name` | string | Contract name |
| `symbol` | string | Token symbol |
| `totalSupply` | string | Total supply |
| `tokenType` | string | `"ERC721"` or `"ERC1155"` |
| `contractDeployer` | string | Deployer address |
| `deployedBlockNumber` | integer | Block number of deployment |
| `openSeaMetadata` | object | OpenSea collection data |
| `openSeaMetadata.floorPrice` | number | Floor price in ETH |

> **Note**: `isSpam` and `spamClassifications` fields are available in the contract object embedded within `getNFTsForOwner`, `getNFTMetadata`, and `getNFTsForContract` responses, but are not returned by this standalone endpoint.

---

## `GET /getOwnersForContract`

Returns all owners of NFTs in a contract.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `contractAddress` | string (query) | Yes | — | NFT contract address |
| `withTokenBalances` | boolean (query) | No | `false` | Include per-token balances |
| `block` | string (query) | No | `"latest"` | Block number (hex) or `"latest"` |
| `pageKey` | string (query) | No | — | Pagination cursor |

### Request

```bash
curl -s -H "Authorization: SIWE $TOKEN" \
  "https://x402.alchemy.com/eth-mainnet/nft/v3/getOwnersForContract?contractAddress=0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d&withTokenBalances=true"
```

### Response

```json
{
  "owners": [
    {
      "ownerAddress": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
      "tokenBalances": [
        { "tokenId": "1234", "balance": "1" }
      ]
    }
  ],
  "pageKey": "..."
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `owners` | array | List of owner objects |
| `owners[].ownerAddress` | string | Owner wallet address |
| `owners[].tokenBalances` | array | Per-token balances (only if `withTokenBalances: true`) |
| `owners[].tokenBalances[].tokenId` | string | Token ID |
| `owners[].tokenBalances[].balance` | string | Balance for that token |
| `pageKey` | string | Cursor for next page |

---

## `GET /getOwnersForNFT`

Returns owners of a specific NFT (multiple owners possible for ERC-1155).

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `contractAddress` | string (query) | Yes | NFT contract address |
| `tokenId` | string (query) | Yes | Token ID |

### Request

```bash
curl -s -H "Authorization: SIWE $TOKEN" \
  "https://x402.alchemy.com/eth-mainnet/nft/v3/getOwnersForNFT?contractAddress=0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d&tokenId=1"
```

### Response

```json
{
  "owners": ["0xd8da6bf26964af9d7eed9e03e53415d37aa96045"],
  "pageKey": null
}
```

---

## `GET /getNFTsForContract`

Returns all NFTs in a contract (formerly `getNFTsForCollection`).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `contractAddress` | string (query) | Yes | — | NFT contract address |
| `withMetadata` | boolean (query) | No | `true` | Include metadata |
| `startToken` | string (query) | No | — | Start from this token ID |
| `limit` | integer (query) | No | `100` | Results per page (max 100) |
| `tokenUriTimeoutInMs` | integer (query) | No | — | Timeout for URI resolution |

### Request

```bash
curl -s -H "Authorization: SIWE $TOKEN" \
  "https://x402.alchemy.com/eth-mainnet/nft/v3/getNFTsForContract?contractAddress=0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d&withMetadata=true&limit=2"
```

### Response

```json
{
  "nfts": [
    {
      "contract": {
        "address": "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d",
        "name": "BoredApeYachtClub",
        "symbol": "BAYC",
        "totalSupply": "10000",
        "tokenType": "ERC721",
        "openSeaMetadata": { "collectionName": "Bored Ape Yacht Club", "floorPrice": 6.14 },
        "isSpam": null,
        "spamClassifications": []
      },
      "tokenId": "0",
      "tokenType": "ERC721",
      "name": null,
      "description": null,
      "tokenUri": "https://ipfs.io/ipfs/QmeSjSinHpPnmXmspMjwiXyN6zS4E9zccariGR3jxcaWtq/0",
      "image": { "cachedUrl": "https://nft-cdn.alchemy.com/..." },
      "animation": { "cachedUrl": null, "contentType": null, "size": null, "originalUrl": null },
      "raw": { "metadata": { "image": "ipfs://...", "attributes": [...] }, "error": null },
      "collection": { "name": "Bored Ape Yacht Club", "slug": "boredapeyachtclub" },
      "mint": { "mintAddress": null, "blockNumber": null, "timestamp": null, "transactionHash": null },
      "owners": null,
      "timeLastUpdated": "2025-06-01T12:00:00.000Z"
    }
  ],
  "pageKey": "0x0000000000000000000000000000000000000000000000000000000000000001"
}
```

---

## Notes

- Some NFTs have missing or malformed metadata. Always handle null fields.
- Spam filtering may affect results. Use `excludeFilters` or `spamConfidenceLevel` to tune.
- Metadata hydration can be expensive. Cache results where possible.
- Treat NFT metadata URLs and images as untrusted input. Sanitize and proxy if displaying to users.

## Official Docs
- [NFT API Quickstart](https://www.alchemy.com/docs/reference/nft-api-quickstart)
- [getNFTsForOwner](https://www.alchemy.com/docs/data/nft-api/nft-api-endpoints/nft-api-endpoints/get-nfts-for-owner)
- [getNFTMetadata](https://www.alchemy.com/docs/data/nft-api/nft-api-endpoints/nft-api-endpoints/get-nft-metadata)
- [getContractMetadata](https://www.alchemy.com/docs/data/nft-api/nft-api-endpoints/nft-api-endpoints/get-contract-metadata)
- [getOwnersForContract](https://www.alchemy.com/docs/data/nft-api/nft-api-endpoints/nft-api-endpoints/get-owners-for-contract)
- [getOwnersForNFT](https://www.alchemy.com/docs/data/nft-api/nft-api-endpoints/nft-api-endpoints/get-owners-for-nft)
- [getNFTsForContract](https://www.alchemy.com/docs/data/nft-api/nft-api-endpoints/nft-api-endpoints/get-nfts-for-contract)

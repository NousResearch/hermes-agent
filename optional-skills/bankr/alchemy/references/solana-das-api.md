---
id: references/solana-das-api.md
name: 'Solana DAS (Digital Asset Standard) API'
description: 'DAS provides normalized access to Solana NFT and compressed asset data.'
tags:
  - alchemy
  - solana
related:
  - solana-rpc.md
updated: 2026-02-23
---
# Solana DAS (Digital Asset Standard) API

Normalized access to Solana NFTs, compressed assets, and token metadata. JSON-RPC POST requests.

**Base URL**: `https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY`

---

## `getAsset`

Returns metadata for a single asset by its ID.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Asset ID (mint address, base58) |
| `displayOptions.showUnverifiedCollections` | boolean | No | Include unverified collections |
| `displayOptions.showCollectionMetadata` | boolean | No | Include collection metadata |
| `displayOptions.showFungible` | boolean | No | Include fungible token details |
| `displayOptions.showNativeBalance` | boolean | No | Include native SOL balance |
| `displayOptions.showInscriptions` | boolean | No | Include inscription data |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getAsset",
    "params": { "id": "F9Lw3ki3hJ7PF9HQXsBzoY8GyE6sPoEZZdXJBsTTD2rk" }
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "id": "F9Lw3ki3hJ7PF9HQXsBzoY8GyE6sPoEZZdXJBsTTD2rk",
    "interface": "V1_NFT",
    "content": {
      "json_uri": "https://...",
      "metadata": {
        "name": "My NFT",
        "symbol": "MNFT",
        "description": "...",
        "attributes": [
          { "trait_type": "Color", "value": "Blue" }
        ]
      },
      "files": [
        { "uri": "https://...", "mime": "image/png" }
      ],
      "links": { "image": "https://..." }
    },
    "authorities": [
      { "address": "...", "scopes": ["full"] }
    ],
    "compression": {
      "eligible": false,
      "compressed": false,
      "data_hash": "",
      "creator_hash": "",
      "asset_hash": "",
      "tree": "",
      "seq": 0,
      "leaf_id": 0
    },
    "grouping": [
      { "group_key": "collection", "group_value": "..." }
    ],
    "royalty": {
      "royalty_model": "creators",
      "target": null,
      "percent": 0.05,
      "basis_points": 500,
      "primary_sale_happened": true
    },
    "ownership": {
      "frozen": false,
      "delegated": false,
      "delegate": null,
      "ownership_model": "single",
      "owner": "83astBRguLMdt2h5U1Tpdq5tjFoJ6noeGwaY3mDLVcri"
    },
    "supply": { "print_max_supply": 0, "print_current_supply": 0, "edition_nonce": 254 },
    "mutable": true,
    "burnt": false
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Asset mint address |
| `interface` | string | Asset type: `"V1_NFT"`, `"V1_PRINT"`, `"LEGACY_NFT"`, `"FungibleAsset"`, `"FungibleToken"` |
| `content.metadata` | object | Name, symbol, description, attributes |
| `content.files` | array | Associated files (URI + MIME type) |
| `content.links` | object | Image/external URLs |
| `compression` | object | Compressed NFT data (tree, leaf ID, hashes) |
| `grouping` | array | Collection grouping (group_key + group_value) |
| `royalty` | object | Royalty info (percentage, basis points) |
| `ownership` | object | Owner address, delegation, frozen state |
| `supply` | object | Edition supply info |
| `mutable` | boolean | Whether metadata can be updated |
| `burnt` | boolean | Whether asset is burned |

---

## `getAssetsByOwner`

Returns all assets owned by an address.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ownerAddress` | string | Yes | — | Owner wallet (base58) |
| `page` | integer | No | `1` | Page number (1-indexed) |
| `limit` | integer | No | `1000` | Results per page (max 1000) |
| `sortBy` | object | No | — | `{ "sortBy": "created", "sortDirection": "asc" }` |
| `displayOptions.showFungible` | boolean | No | `false` | Include fungible tokens |
| `displayOptions.showCollectionMetadata` | boolean | No | `false` | Include collection metadata |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getAssetsByOwner",
    "params": {
      "ownerAddress": "83astBRguLMdt2h5U1Tpdq5tjFoJ6noeGwaY3mDLVcri",
      "page": 1,
      "limit": 10
    }
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "total": 42,
    "limit": 10,
    "page": 1,
    "items": [
      {
        "id": "F9Lw3ki3hJ7PF9HQXsBzoY8GyE6sPoEZZdXJBsTTD2rk",
        "interface": "V1_NFT",
        "content": { "metadata": { "name": "My NFT" }, "links": { "image": "https://..." } },
        "ownership": { "owner": "83astBRguLMdt2h5U1Tpdq5tjFoJ6noeGwaY3mDLVcri" },
        "grouping": [{ "group_key": "collection", "group_value": "..." }]
      }
    ]
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `total` | integer | Total assets owned |
| `limit` | integer | Page size |
| `page` | integer | Current page |
| `items` | array | Asset objects (same schema as `getAsset` result) |

---

## `getAssetsByGroup`

Returns assets in a group (e.g., a collection).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `groupKey` | string | Yes | — | Grouping key (e.g., `"collection"`) |
| `groupValue` | string | Yes | — | Group value (e.g., collection mint address) |
| `page` | integer | No | `1` | Page number |
| `limit` | integer | No | `1000` | Results per page (max 1000) |
| `sortBy` | object | No | — | Sort configuration |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getAssetsByGroup",
    "params": {
      "groupKey": "collection",
      "groupValue": "J1S9H3QjnRtBbbuD4HjPV6RpRhwuk4zKbxsnCHuTgh9w",
      "page": 1,
      "limit": 10
    }
  }'
```

### Response

Same structure as `getAssetsByOwner` (`total`, `limit`, `page`, `items`).

---

## `searchAssets`

Search assets by various criteria.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ownerAddress` | string | No | Filter by owner |
| `grouping` | array | No | Filter by group (e.g., `[{ "group_key": "collection", "group_value": "..." }]`) |
| `burnt` | boolean | No | Filter by burn status |
| `frozen` | boolean | No | Filter by frozen status |
| `interface` | string | No | Filter by interface type |
| `page` | integer | No | Page number |
| `limit` | integer | No | Results per page |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "searchAssets",
    "params": {
      "ownerAddress": "83astBRguLMdt2h5U1Tpdq5tjFoJ6noeGwaY3mDLVcri",
      "grouping": [{ "group_key": "collection", "group_value": "J1S9H3QjnRtBbbuD4HjPV6RpRhwuk4zKbxsnCHuTgh9w" }],
      "page": 1,
      "limit": 10
    }
  }'
```

### Response

Same structure as `getAssetsByOwner`.

---

## `getAssetProof`

Returns the Merkle proof for a compressed asset (needed for transfers and burns).

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Asset ID (mint address) |

### Request

```bash
curl -s -X POST https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getAssetProof",
    "params": { "id": "F9Lw3ki3hJ7PF9HQXsBzoY8GyE6sPoEZZdXJBsTTD2rk" }
  }'
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `root` | string | Merkle tree root |
| `proof` | string[] | Array of proof hashes |
| `node_index` | integer | Leaf index in tree |
| `leaf` | string | Leaf hash |
| `tree_id` | string | Merkle tree address |

---

## Notes

- DAS uses named parameters (object), not positional arrays like standard Solana RPC.
- Asset grouping fields vary by program and collection structure.
- Pagination is required for large wallets. Use `page` and `limit`.
- Compressed assets require `getAssetProof` for on-chain operations.

## Official Docs
- [DAS APIs for Solana](https://www.alchemy.com/docs/chains/solana/das-api)
- [getAsset](https://www.alchemy.com/docs/chains/solana/solana-api-endpoints/getasset)
- [getAssetsByOwner](https://www.alchemy.com/docs/chains/solana/solana-api-endpoints/getassetsbyowner)
- [getAssetsByGroup](https://www.alchemy.com/docs/chains/solana/solana-api-endpoints/getassetsbygroup)
- [searchAssets](https://www.alchemy.com/docs/chains/solana/solana-api-endpoints/searchassets)

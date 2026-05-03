# Solana DAS API (Digital Asset Standard) Reference

The Metaplex Digital Asset Standard (DAS) API is a comprehensive service for querying Solana digital assets efficiently. It supports standard and compressed NFTs, fungible tokens, MPL Core Assets, and Token 2022 Assets.

**Docs:** https://www.quicknode.com/docs/solana/solana-das-api

## Prerequisites

Enable the **Metaplex Digital Asset Standard (DAS) API** add-on on your QuickNode Solana endpoint via the [Marketplace](https://marketplace.quicknode.com/).

## Methods Overview

| Method | Description |
|--------|-------------|
| `getAsset` | Get metadata for a single asset |
| `getAssets` | Get metadata for multiple assets in one call |
| `getAssetProof` | Get Merkle proof for a compressed asset |
| `getAssetProofs` | Get Merkle proofs for multiple compressed assets |
| `getAssetsByAuthority` | List assets controlled by an authority |
| `getAssetsByCreator` | List assets by creator address |
| `getAssetsByGroup` | List assets by group (e.g., collection) |
| `getAssetsByOwner` | List assets owned by a wallet |
| `getAssetSignatures` | Get transaction signatures for compressed assets |
| `getTokenAccounts` | List token accounts by mint or owner |
| `getNftEditions` | Get edition details of a master NFT |
| `searchAssets` | Search assets with flexible filters |

## Supported Asset Types

- **Standard NFTs** — traditional Solana NFTs
- **Compressed NFTs (cNFTs)** — Merkle tree-based, cost-efficient NFTs
- **Fungible Tokens** — SPL tokens
- **MPL Core Assets** — single-account design NFTs
- **Token 2022 Assets** — tokens using the Token Extensions program

## getAsset

Retrieve metadata for a single asset by its mint address.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAsset',
    params: {
      id: '9ARngHhVaCtH5JFieRdSS5Y8cdZk2TMF4tfGSWFB9iSK',
      options: {
        showFungible: true,
        showCollectionMetadata: true
      }
    }
  })
});
const { result } = await response.json();
// result.content — metadata (name, description, image, attributes)
// result.ownership — owner, delegate, frozen status
// result.compression — tree, leaf, proof info (if compressed)
// result.royalty — royalty model, basis points, creators
// result.creators — array of creator addresses with verified status
// result.supply — edition/print supply info
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | The asset mint address |
| `options.showFungible` | boolean | No | Include fungible token info |
| `options.showCollectionMetadata` | boolean | No | Include collection metadata |
| `options.showUnverifiedCollections` | boolean | No | Include unverified collections |

## getAssets

Fetch metadata for multiple assets in a single request.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssets',
    params: {
      ids: [
        '9ARngHhVaCtH5JFieRdSS5Y8cdZk2TMF4tfGSWFB9iSK',
        'BwJHge5FmE5RBkmWPoKzCWwxZFXsnqCMKHiiibXPJias'
      ]
    }
  })
});
const { result } = await response.json();
// result.items — array of asset metadata objects
```

## getAssetProof

Get the Merkle proof for a compressed asset. Required for transferring or modifying compressed NFTs on-chain.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetProof',
    params: {
      id: 'D85MZkvir9yQZFDHt8U2ZmS7D3LXKdiSjvw2MBdscJJa'
    }
  })
});
const { result } = await response.json();
// result.root — Merkle tree root hash
// result.proof — array of proof nodes
// result.node_index — index in the tree
// result.leaf — leaf hash
// result.tree_id — Merkle tree address
```

## getAssetProofs

Retrieve Merkle proofs for multiple compressed assets in one call.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetProofs',
    params: {
      ids: [
        'D85MZkvir9yQZFDHt8U2ZmS7D3LXKdiSjvw2MBdscJJa',
        'AnotherCompressedAssetMint...'
      ]
    }
  })
});
const { result } = await response.json();
// result — object keyed by asset ID, each containing root, proof, node_index, leaf, tree_id
```

## getAssetsByOwner

List all digital assets owned by a wallet address.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetsByOwner',
    params: {
      ownerAddress: 'E645TckHQnDcavVv92Etc6xSWQaq8zzPtPRGBheviRAk',
      limit: 10,
      sortBy: { sortBy: 'recent_action', sortDirection: 'desc' },
      options: {
        showFungible: true,
        showCollectionMetadata: true
      }
    }
  })
});
const { result } = await response.json();
// result.total — total assets owned
// result.items — array of asset metadata objects
// result.cursor — use in next request for pagination
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ownerAddress` | string | Yes | Wallet address |
| `limit` | integer | No | Max results per page |
| `page` | integer | No | Page number (page-based pagination) |
| `cursor` | string | No | Cursor from previous response (cursor-based pagination) |
| `before` / `after` | string | No | Range-based pagination |
| `sortBy` | object | No | `{ sortBy: "created" \| "recent_action" \| "id" \| "none", sortDirection: "asc" \| "desc" }` |
| `options.showFungible` | boolean | No | Include fungible tokens |
| `options.showCollectionMetadata` | boolean | No | Include collection metadata |

## getAssetsByCreator

List assets created by a specific address.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetsByCreator',
    params: {
      creatorAddress: '3pMvTLUA9NzZQd4gi725p89mvND1wRNQM3C8XEv1hTdA',
      limit: 10
    }
  })
});
const { result } = await response.json();
// result.total, result.items, result.cursor
```

## getAssetsByGroup

List assets by group identifier (e.g., collection address).

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetsByGroup',
    params: {
      groupKey: 'collection',
      groupValue: 'CollectionMintAddress...',
      limit: 10
    }
  })
});
const { result } = await response.json();
// result.total, result.items, result.cursor
```

## getAssetsByAuthority

List assets controlled by a specific authority.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetsByAuthority',
    params: {
      authorityAddress: 'AuthorityPubkey...',
      limit: 10
    }
  })
});
const { result } = await response.json();
// result.total, result.items, result.cursor
```

## getAssetSignatures

Get transaction signatures associated with a compressed asset.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetSignatures',
    params: {
      id: 'CompressedAssetMint...',
      limit: 10
    }
  })
});
const { result } = await response.json();
// result.items — array of transaction signature objects
```

## getTokenAccounts

List token accounts and balances by mint address or owner address. Useful for finding all holders of a token or all tokens held by a wallet.

```javascript
// By mint address — find holders of a token
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getTokenAccounts',
    params: {
      mintAddress: 'So11111111111111111111111111111111111111112',
      limit: 10
    }
  })
});
const { result } = await response.json();
// result.total — total token accounts
// result.token_accounts — array of accounts with:
//   address, mint, owner, amount, delegated_amount, frozen
```

```javascript
// By owner address — find all tokens held by a wallet
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getTokenAccounts',
    params: {
      ownerAddress: 'WalletPubkey...',
      limit: 10
    }
  })
});
```

## getNftEditions

Retrieve edition details for a master NFT, including all printed editions.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getNftEditions',
    params: {
      mintAddress: 'MasterEditionMint...',
      limit: 10
    }
  })
});
const { result } = await response.json();
// result.items — array of edition details
```

## searchAssets

Search for assets using flexible filter criteria. The most powerful query method in the DAS API.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'searchAssets',
    params: {
      ownerAddress: 'WalletPubkey...',
      compressed: true,
      limit: 10,
      sortBy: { sortBy: 'recent_action', sortDirection: 'desc' }
    }
  })
});
const { result } = await response.json();
// result.total, result.items, result.cursor
```

**Search Filter Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ownerAddress` | string | Filter by asset owner |
| `creatorAddress` | string | Filter by creator |
| `authorityAddress` | string | Filter by authority |
| `grouping` | array | Filter by group (e.g., `[["collection", "address"]]`) |
| `delegateAddress` | string | Filter by delegate |
| `compressed` | boolean | Filter by compression status |
| `compressible` | boolean | Filter by compressibility |
| `frozen` | boolean | Filter by frozen status |
| `burnt` | boolean | Filter by burn status |
| `supply` | integer | Filter by supply amount |
| `supplyMint` | string | Filter by supply mint |
| `interface` | string | Filter by asset interface type |
| `ownerType` | string | Filter by owner type |
| `royaltyTargetType` | string | Filter by royalty target type |
| `royaltyTarget` | string | Filter by royalty recipient |
| `royaltyAmount` | integer | Filter by royalty basis points |
| `jsonUri` | string | Filter by metadata URI |
| `negate` | boolean | Invert filter logic |
| `conditionType` | string | Condition type for filters |

**Sorting & Pagination:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sortBy` | object | `{ sortBy: "created" \| "recent_action" \| "id" \| "none", sortDirection: "asc" \| "desc" }` |
| `limit` | integer | Max results per page |
| `page` | integer | Page number |
| `cursor` | string | Cursor for next page |
| `before` / `after` | string | Range-based pagination |
| `showFungible` | boolean | Include fungible tokens |
| `showCollectionMetadata` | boolean | Include collection metadata |

## Pagination

The DAS API supports three pagination modes:

**Page-based** — simple numbered pages:
```javascript
{ page: 1, limit: 100 }
```

**Cursor-based** — use cursor from previous response (recommended for large datasets):
```javascript
{ cursor: 'previousResponse.result.cursor', limit: 100 }
```

**Range-based** — before/after asset identifiers:
```javascript
{ before: 'assetId', after: 'assetId', limit: 100 }
```

## Common Use Cases

### Get all NFTs in a collection
```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetsByGroup',
    params: {
      groupKey: 'collection',
      groupValue: 'CollectionMintAddress...',
      limit: 100
    }
  })
});
```

### Get all compressed NFTs owned by a wallet
```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'searchAssets',
    params: {
      ownerAddress: 'WalletPubkey...',
      compressed: true,
      limit: 100
    }
  })
});
```

### Transfer a compressed NFT (get proof first)
```javascript
// 1. Get the asset proof
const proofResponse = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetProof',
    params: { id: 'CompressedNftMint...' }
  })
});
const { result: proof } = await proofResponse.json();

// 2. Get asset details
const assetResponse = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAsset',
    params: { id: 'CompressedNftMint...' }
  })
});
const { result: asset } = await assetResponse.json();

// 3. Use proof.root, proof.proof, proof.node_index, and asset data
//    to construct the transfer instruction via @metaplex-foundation/mpl-bubblegum
```

### Get all token holders for a mint
```javascript
let cursor = null;
const allAccounts = [];

do {
  const params = { mintAddress: 'TokenMint...', limit: 100 };
  if (cursor) params.cursor = cursor;

  const response = await fetch(process.env.QUICKNODE_RPC_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'getTokenAccounts',
      params
    })
  });
  const { result } = await response.json();
  allAccounts.push(...result.token_accounts);
  cursor = result.cursor;
} while (cursor);
```

## Documentation

- **DAS API Overview**: https://www.quicknode.com/docs/solana/solana-das-api
- **getAsset**: https://www.quicknode.com/docs/solana/getAsset
- **getAssets**: https://www.quicknode.com/docs/solana/getAssets
- **getAssetProof**: https://www.quicknode.com/docs/solana/getAssetProof
- **getAssetProofs**: https://www.quicknode.com/docs/solana/getAssetProofs
- **getAssetsByAuthority**: https://www.quicknode.com/docs/solana/getAssetsByAuthority
- **getAssetsByCreator**: https://www.quicknode.com/docs/solana/getAssetsByCreator
- **getAssetsByGroup**: https://www.quicknode.com/docs/solana/getAssetsByGroup
- **getAssetsByOwner**: https://www.quicknode.com/docs/solana/getAssetsByOwner
- **getAssetSignatures**: https://www.quicknode.com/docs/solana/getAssetSignatures
- **getTokenAccounts**: https://www.quicknode.com/docs/solana/getTokenAccounts
- **getNftEditions**: https://www.quicknode.com/docs/solana/getNftEditions
- **searchAssets**: https://www.quicknode.com/docs/solana/searchAssets
- **Marketplace Add-on**: https://marketplace.quicknode.com/add-on/metaplex-digital-asset-standard-api

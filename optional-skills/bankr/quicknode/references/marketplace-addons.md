# Quicknode Marketplace Add-ons Reference

Quicknode Marketplace provides enhanced blockchain APIs as add-ons to standard RPC endpoints. Enable add-ons in the Quicknode dashboard to access these methods.

## Ethereum Token APIs

### qn_getWalletTokenBalance

Get all ERC-20 token balances for an address.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'qn_getWalletTokenBalance',
    params: [{
      wallet: '0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045',
      contracts: [] // Empty array for all tokens
    }]
  })
});

const { result } = await response.json();
// result.assets: Array of token balances
```

**Response:**
```json
{
  "result": {
    "owner": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "assets": [
      {
        "address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "name": "USD Coin",
        "symbol": "USDC",
        "decimals": 6,
        "balance": "1000000000",
        "balanceUSD": "1000.00"
      }
    ]
  }
}
```

### qn_getTokenMetadataByContractAddress

Get token metadata for a specific contract.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'qn_getTokenMetadataByContractAddress',
    params: [{
      contract: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'
    }]
  })
});
```

**Response:**
```json
{
  "result": {
    "name": "USD Coin",
    "symbol": "USDC",
    "decimals": "6",
    "contractAddress": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
  }
}
```

### qn_getTokenMetadataBySymbol

Get token metadata by symbol.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'qn_getTokenMetadataBySymbol',
    params: [{
      symbol: 'USDC'
    }]
  })
});
```

## Ethereum NFT APIs

### qn_fetchNFTs

Fetch NFTs owned by an address.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'qn_fetchNFTs',
    params: [{
      wallet: '0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045',
      page: 1,
      perPage: 10,
      contracts: [] // Optional: filter by contracts
    }]
  })
});

const { result } = await response.json();
// result.assets: Array of NFTs
// result.totalItems: Total count
// result.pageNumber: Current page
```

**Response:**
```json
{
  "result": {
    "owner": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
    "assets": [
      {
        "collectionName": "Bored Ape Yacht Club",
        "collectionAddress": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",
        "collectionTokenId": "1234",
        "name": "Bored Ape #1234",
        "description": "A bored ape",
        "imageUrl": "ipfs://...",
        "traits": [
          { "trait_type": "Background", "value": "Blue" }
        ]
      }
    ],
    "totalItems": 42,
    "pageNumber": 1
  }
}
```

### qn_fetchNFTCollectionDetails

Get collection-level details.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'qn_fetchNFTCollectionDetails',
    params: [{
      contracts: ['0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D']
    }]
  })
});
```

**Response:**
```json
{
  "result": [
    {
      "address": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",
      "name": "Bored Ape Yacht Club",
      "erc": "erc721",
      "totalSupply": "10000"
    }
  ]
}
```

### qn_fetchNFTsByCollection

Fetch NFTs from a specific collection.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'qn_fetchNFTsByCollection',
    params: [{
      collection: '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D',
      tokens: ['1', '2', '3'], // Optional: specific tokens
      page: 1,
      perPage: 10
    }]
  })
});
```

### qn_verifyNFTsOwner

Verify NFT ownership.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'qn_verifyNFTsOwner',
    params: [{
      wallet: '0xWalletAddress...',
      contracts: [
        {
          address: '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D',
          tokenIds: ['1234']
        }
      ]
    }]
  })
});
```

## Solana Add-ons

### Priority Fee API

Get recommended priority fees for Solana transactions.

```javascript
import { createSolanaRpc } from '@solana/kit';

const rpc = createSolanaRpc(process.env.QUICKNODE_RPC_URL!);

const response = await rpc.request('qn_estimatePriorityFees', {
  last_n_blocks: 100,
  account: 'YourAccountPubkey...'
}).send();

// Response includes recommended fees by percentile
// per_compute_unit.low, medium, high, extreme
```

**Response:**
```json
{
  "result": {
    "per_compute_unit": {
      "low": 100,
      "medium": 1000,
      "high": 10000,
      "extreme": 100000
    },
    "per_transaction": {
      "low": 1000,
      "medium": 10000,
      "high": 100000,
      "extreme": 1000000
    }
  }
}
```

### DAS API (Digital Asset Standard)

Comprehensive API for querying Solana digital assets — standard NFTs, compressed NFTs, fungible tokens, MPL Core Assets, and Token 2022 Assets. Requires the Metaplex DAS API add-on enabled on your endpoint.

**Docs:** https://www.quicknode.com/docs/solana/solana-das-api

```javascript
// Get assets by owner
const response = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetsByOwner',
    params: { ownerAddress: 'WalletPubkey...', limit: 10 }
  })
});

// Get single asset details
const asset = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAsset',
    params: { id: 'AssetMintAddress...' }
  })
});

// Search assets with filters
const search = await fetch(process.env.QUICKNODE_RPC_URL, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'searchAssets',
    params: { ownerAddress: 'WalletPubkey...', compressed: true, limit: 10 }
  })
});
```

**Available methods:** `getAsset`, `getAssets`, `getAssetProof`, `getAssetProofs`, `getAssetsByAuthority`, `getAssetsByCreator`, `getAssetsByGroup`, `getAssetsByOwner`, `getAssetSignatures`, `getTokenAccounts`, `getNftEditions`, `searchAssets`

See [solana-das-api-reference.md](solana-das-api-reference.md) for complete DAS API documentation with all methods, parameters, and examples.

### Metis Jupiter API

Access Jupiter DEX aggregator for swaps via REST endpoints on your QuickNode Solana endpoint.

> **Endpoint:** Set `QUICKNODE_METIS_URL` to your QuickNode Metis endpoint (e.g., `https://jupiter-swap-api.quiknode.pro/YOUR_TOKEN`). Enable the Metis - Jupiter V6 Swap API add-on in your QuickNode dashboard. Do not use the public Jupiter API for production — it has lower rate limits and no SLA.

**Docs:** https://www.quicknode.com/docs/solana/metis-overview

```javascript
// Get swap quote (GET request)
const quoteUrl = new URL(`${process.env.QUICKNODE_METIS_URL}/quote`);
quoteUrl.searchParams.set('inputMint', 'So11111111111111111111111111111111111111112'); // SOL
quoteUrl.searchParams.set('outputMint', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'); // USDC
quoteUrl.searchParams.set('amount', '1000000000'); // 1 SOL in lamports
quoteUrl.searchParams.set('slippageBps', '50'); // 0.5% slippage

const quoteResponse = await fetch(quoteUrl.toString());
const quote = await quoteResponse.json();

// Execute swap (POST request)
const swapResponse = await fetch(`${process.env.QUICKNODE_METIS_URL}/swap`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    userPublicKey: 'YourPubkey...',
    quoteResponse: quote
  })
});

const { swapTransaction, lastValidBlockHeight } = await swapResponse.json();
// swapTransaction is a serialized transaction ready for signing and sending
```

**Using the Jupiter SDK:**

```typescript
import { createJupiterApiClient } from '@jup-ag/api';

const jupiterApi = createJupiterApiClient({
  basePath: `${process.env.QUICKNODE_METIS_URL}`
});

const quote = await jupiterApi.quoteGet({
  inputMint: 'So11111111111111111111111111111111111111112',
  outputMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  amount: 1000000000,
  slippageBps: 50
});

const swapResult = await jupiterApi.swapPost({
  swapRequest: {
    quoteResponse: quote,
    userPublicKey: 'YourPubkey...'
  }
});
```

### Yellowstone gRPC

High-performance streaming for Solana data.

```javascript
// Configure in endpoint settings
// Use gRPC client to connect

const client = new YellowstoneClient({
  endpoint: 'YOUR_GRPC_ENDPOINT',
  token: process.env.QUICKNODE_API_KEY!
});

// Subscribe to account updates
const stream = client.subscribe({
  accounts: {
    accountIds: ['AccountPubkey...']
  }
});

stream.on('data', (update) => {
  console.log('Account updated:', update);
});
```

For full Yellowstone gRPC documentation including all filter types, subscription examples, and multi-language setup, see [yellowstone-grpc-reference.md](yellowstone-grpc-reference.md).

### Jito Bundles

MEV protection and bundle submission.

```javascript
// Submit bundle
const bundleResult = await rpc.request('sendBundle', {
  transactions: [
    'Base64EncodedTx1...',
    'Base64EncodedTx2...'
  ]
}).send();

// Get bundle status
const status = await rpc.request('getBundleStatuses', {
  bundleIds: [bundleResult.bundleId]
}).send();
```

## EVM Trace & Debug APIs

### trace_call

Trace a call without executing.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'trace_call',
    params: [
      {
        to: '0xContractAddress...',
        data: '0xFunctionSelector...'
      },
      ['trace'],
      'latest'
    ]
  })
});
```

### trace_transaction

Get execution trace for a transaction.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'trace_transaction',
    params: ['0xTransactionHash...']
  })
});
```

### debug_traceTransaction

Detailed transaction debugging.

```javascript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'debug_traceTransaction',
    params: [
      '0xTransactionHash...',
      { tracer: 'callTracer' }
    ]
  })
});
```

## Archive Data

Access historical blockchain state.

```javascript
// Get balance at specific block
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'eth_getBalance',
    params: [
      '0xAddress...',
      '0xF4240' // Block 1,000,000
    ]
  })
});

// Call contract at historical block
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'eth_call',
    params: [
      {
        to: '0xContract...',
        data: '0xFunctionSelector...'
      },
      '0xF4240' // Block 1,000,000
    ]
  })
});
```

## Using with Quicknode SDK

```typescript
import { Core } from '@quicknode/sdk';

const core = new Core({
  endpointUrl: process.env.QUICKNODE_RPC_URL!,
});

// Token API
const tokenBalances = await core.client.qn_getWalletTokenBalance({
  wallet: '0x...',
  contracts: []
});

// NFT API
const nfts = await core.client.qn_fetchNFTs({
  wallet: '0x...',
  page: 1,
  perPage: 10
});

// Collection details
const collection = await core.client.qn_fetchNFTCollectionDetails({
  contracts: ['0x...']
});
```

## Add-on Availability by Chain

| Add-on | Ethereum | Polygon | Arbitrum | Base | Solana |
|--------|----------|---------|----------|------|--------|
| Token API | Yes | - | - | - | - |
| NFT API | Yes | - | - | - | DAS |
| Trace API | Yes | Yes | Yes | Yes | - |
| Debug API | Yes | Yes | Yes | Yes | - |
| Archive | Yes | Yes | Yes | Yes | - |
| Priority Fee | - | - | - | - | Yes |
| Jupiter/Metis | - | - | - | - | Yes |
| Yellowstone | - | - | - | - | Yes |
| Jito | - | - | - | - | Yes |

## Rate Limits

Add-on methods consume credits based on complexity:

| Method Type | Credits |
|-------------|---------|
| Token balance | 50 |
| NFT fetch | 100 |
| Collection details | 50 |
| Trace call | 200 |
| Debug trace | 500 |
| Archive query | 100 |

## Documentation

- **Marketplace**: https://marketplace.quicknode.com/
- **Token API**: https://www.quicknode.com/docs/ethereum/qn_getWalletTokenBalance
- **NFT API**: https://www.quicknode.com/docs/ethereum/qn_fetchNFTs
- **Solana Add-ons**: https://www.quicknode.com/docs/solana
- **Metis Jupiter API**: https://www.quicknode.com/docs/solana/metis-overview
- **Trace API**: https://www.quicknode.com/docs/ethereum/trace_call
- **DAS API**: https://www.quicknode.com/docs/solana/solana-das-api
- **Guides**: https://www.quicknode.com/guides/tags/marketplace

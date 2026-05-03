# Quicknode SDK Reference

The Quicknode SDK provides a type-safe JavaScript/TypeScript client for interacting with Quicknode services.

## Installation

```bash
npm install @quicknode/sdk
```

## Core Setup

```typescript
import { Core } from '@quicknode/sdk';

const core = new Core({
  endpointUrl: process.env.QUICKNODE_RPC_URL!,
});
```

## Configuration Options

```typescript
const core = new Core({
  // Required: Your Quicknode endpoint URL
  endpointUrl: process.env.QUICKNODE_RPC_URL!,

  // Optional: Chain ID (auto-detected from endpoint)
  chain: 1,

  // Optional: Request timeout in milliseconds
  timeout: 30000,

  // Optional: Custom fetch implementation
  fetch: customFetch,

  // Optional: Custom headers
  headers: {
    'X-Custom-Header': 'value'
  }
});
```

## Standard RPC Methods

### Ethereum/EVM

```typescript
// Get balance
const balance = await core.client.getBalance({
  address: '0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045'
});

// Get block
const block = await core.client.getBlock({
  blockNumber: 'latest'
});

// Get transaction
const tx = await core.client.getTransaction({
  hash: '0xTransactionHash...'
});

// Get transaction receipt
const receipt = await core.client.getTransactionReceipt({
  hash: '0xTransactionHash...'
});

// Call contract
const result = await core.client.call({
  to: '0xContractAddress...',
  data: '0xFunctionSelector...'
});

// Estimate gas
const gas = await core.client.estimateGas({
  from: '0xSender...',
  to: '0xRecipient...',
  value: '0x0'
});

// Get logs
const logs = await core.client.getLogs({
  address: '0xContractAddress...',
  fromBlock: 18000000,
  toBlock: 'latest',
  topics: ['0xEventSignature...']
});
```

## Token API Methods

Requires Token API add-on enabled.

### qn_getWalletTokenBalance

```typescript
const tokenBalances = await core.client.qn_getWalletTokenBalance({
  wallet: '0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045',
  contracts: [] // Empty for all tokens, or specify addresses
});

console.log('Tokens:', tokenBalances.assets);
```

### qn_getTokenMetadataByContractAddress

```typescript
const metadata = await core.client.qn_getTokenMetadataByContractAddress({
  contract: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'
});

console.log(`${metadata.name} (${metadata.symbol})`);
```

### qn_getTokenMetadataBySymbol

```typescript
const metadata = await core.client.qn_getTokenMetadataBySymbol({
  symbol: 'USDC'
});
```

## NFT API Methods

Requires NFT API add-on enabled.

### qn_fetchNFTs

```typescript
const nfts = await core.client.qn_fetchNFTs({
  wallet: '0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045',
  page: 1,
  perPage: 10,
  contracts: [] // Optional: filter by contracts
});

console.log(`Total NFTs: ${nfts.totalItems}`);
nfts.assets.forEach(nft => {
  console.log(`${nft.name} - ${nft.collectionName}`);
});
```

### qn_fetchNFTCollectionDetails

```typescript
const collections = await core.client.qn_fetchNFTCollectionDetails({
  contracts: [
    '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D',
    '0x60E4d786628Fea6478F785A6d7e704777c86a7c6'
  ]
});

collections.forEach(collection => {
  console.log(`${collection.name}: ${collection.totalSupply} items`);
});
```

### qn_fetchNFTsByCollection

```typescript
const collectionNFTs = await core.client.qn_fetchNFTsByCollection({
  collection: '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D',
  tokens: ['1', '2', '3'], // Optional: specific token IDs
  page: 1,
  perPage: 10
});
```

### qn_verifyNFTsOwner

```typescript
const verification = await core.client.qn_verifyNFTsOwner({
  wallet: '0xOwnerAddress...',
  contracts: [
    {
      address: '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D',
      tokenIds: ['1234', '5678']
    }
  ]
});

console.log('Owns NFTs:', verification.owner);
```

## Multi-Chain Setup

```typescript
import { Core } from '@quicknode/sdk';

// Create clients for multiple chains
const chains = {
  ethereum: new Core({
    endpointUrl: 'https://eth-endpoint.quiknode.pro/KEY/'
  }),
  polygon: new Core({
    endpointUrl: 'https://polygon-endpoint.quiknode.pro/KEY/'
  }),
  arbitrum: new Core({
    endpointUrl: 'https://arbitrum-endpoint.quiknode.pro/KEY/'
  }),
  base: new Core({
    endpointUrl: 'https://base-endpoint.quiknode.pro/KEY/'
  })
};

// Use appropriate chain
async function getBalance(chain: keyof typeof chains, address: string) {
  return chains[chain].client.getBalance({ address });
}

const ethBalance = await getBalance('ethereum', '0x...');
const polyBalance = await getBalance('polygon', '0x...');
```

## Custom RPC Calls

For methods not directly exposed by the SDK:

```typescript
// Generic request method
const result = await core.client.request({
  method: 'trace_transaction',
  params: ['0xTransactionHash...']
});

// Or use raw fetch
const response = await fetch(core.config.endpointUrl, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'debug_traceTransaction',
    params: ['0xTransactionHash...', { tracer: 'callTracer' }]
  })
});
```

## Error Handling

```typescript
import { Core, QuicknodeError } from '@quicknode/sdk';

const core = new Core({
  endpointUrl: process.env.QUICKNODE_RPC_URL!
});

try {
  const balance = await core.client.getBalance({
    address: '0x...'
  });
} catch (error) {
  if (error instanceof QuicknodeError) {
    console.error('Quicknode Error:', error.message);
    console.error('Code:', error.code);
    console.error('Details:', error.data);
  } else {
    throw error;
  }
}
```

## TypeScript Types

```typescript
import type {
  GetBalanceParams,
  GetBalanceResult,
  GetBlockParams,
  GetBlockResult,
  QnFetchNFTsParams,
  QnFetchNFTsResult,
  QnGetWalletTokenBalanceParams,
  QnGetWalletTokenBalanceResult
} from '@quicknode/sdk';

// Use types for better IDE support
const params: QnFetchNFTsParams = {
  wallet: '0x...',
  page: 1,
  perPage: 10
};

const result: QnFetchNFTsResult = await core.client.qn_fetchNFTs(params);
```

## Common Patterns

### Batch Balance Check

```typescript
async function getMultipleBalances(addresses: string[]) {
  const balancePromises = addresses.map(address =>
    core.client.getBalance({ address })
  );

  const balances = await Promise.all(balancePromises);

  return addresses.map((address, index) => ({
    address,
    balance: balances[index]
  }));
}
```

### Token Portfolio

```typescript
async function getPortfolio(wallet: string) {
  const [ethBalance, tokens, nfts] = await Promise.all([
    core.client.getBalance({ address: wallet }),
    core.client.qn_getWalletTokenBalance({ wallet, contracts: [] }),
    core.client.qn_fetchNFTs({ wallet, page: 1, perPage: 100 })
  ]);

  return {
    eth: ethBalance,
    tokens: tokens.assets,
    nfts: nfts.assets,
    nftCount: nfts.totalItems
  };
}
```

### Retry with Backoff

```typescript
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 1000
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (attempt === maxRetries - 1) throw error;

      const delay = baseDelay * Math.pow(2, attempt);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw new Error('Max retries exceeded');
}

// Usage
const balance = await withRetry(() =>
  core.client.getBalance({ address: '0x...' })
);
```

### Caching Layer

```typescript
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 60000; // 1 minute

async function cachedCall<T>(
  key: string,
  fn: () => Promise<T>
): Promise<T> {
  const cached = cache.get(key);

  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.data as T;
  }

  const data = await fn();
  cache.set(key, { data, timestamp: Date.now() });
  return data;
}

// Usage
const balance = await cachedCall(
  `balance:${address}`,
  () => core.client.getBalance({ address })
);
```

## Browser Usage

```html
<script type="module">
import { Core } from 'https://esm.sh/@quicknode/sdk';

const core = new Core({
  endpointUrl: process.env.QUICKNODE_RPC_URL!
});

const balance = await core.client.getBalance({
  address: '0x...'
});
console.log('Balance:', balance);
</script>
```

## Node.js Best Practices

```typescript
import { Core } from '@quicknode/sdk';

// Use environment variables for API keys
const core = new Core({
  endpointUrl: process.env.QUICKNODE_ENDPOINT_URL!
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  // Cleanup if needed
  process.exit(0);
});
```

## Documentation

- **SDK Overview**: https://www.quicknode.com/docs/quicknode-sdk
- **npm Package**: https://www.npmjs.com/package/@quicknode/sdk
- **Guides**: https://www.quicknode.com/guides/tags/quicknode-sdk

# RPC Endpoints Reference

Quicknode provides low-latency JSON-RPC, WebSocket, and REST endpoints for 80+ blockchain networks with built-in authentication, global load balancing, and per-method documentation.

## Overview

| Property | Value |
|----------|-------|
| **Protocol** | JSON-RPC 2.0 (HTTP + WebSocket), REST (Beacon Chain) |
| **Chains** | 80+ networks (EVM, Solana, Bitcoin, and more) |
| **Authentication** | Token in URL path, optional JWT or IP allowlisting |
| **EVM Libraries** | ethers.js, viem, web3.js |
| **Solana Libraries** | @solana/kit, @solana/web3.js |
| **Bitcoin** | Raw JSON-RPC via `fetch` |
| **Endpoint Format** | `https://{name}.{network}.quiknode.pro/{token}/` |
| **WebSocket Format** | `wss://{name}.{network}.quiknode.pro/{token}/` |
| **Per-Method Docs** | `https://www.quicknode.com/docs/{chain}/{method}` |

## Connection Setup

### EVM Chains

```typescript
// ethers.js — HTTP
import { JsonRpcProvider } from 'ethers';
const provider = new JsonRpcProvider(process.env.QUICKNODE_RPC_URL!);

// ethers.js — WebSocket
import { WebSocketProvider } from 'ethers';
const wsProvider = new WebSocketProvider(process.env.QUICKNODE_WSS_URL!);

// viem — HTTP
import { createPublicClient, http } from 'viem';
import { mainnet } from 'viem/chains';
const client = createPublicClient({
  chain: mainnet,
  transport: http(process.env.QUICKNODE_RPC_URL!),
});

// viem — WebSocket
import { createPublicClient, webSocket } from 'viem';
import { mainnet } from 'viem/chains';
const wsClient = createPublicClient({
  chain: mainnet,
  transport: webSocket(process.env.QUICKNODE_WSS_URL!),
});
```

### Solana

```typescript
// @solana/kit — HTTP
import { createSolanaRpc } from '@solana/kit';
const rpc = createSolanaRpc(process.env.QUICKNODE_RPC_URL!);

// @solana/kit — WebSocket
import { createSolanaRpcSubscriptions } from '@solana/kit';
const rpcSubscriptions = createSolanaRpcSubscriptions(process.env.QUICKNODE_WSS_URL!);
```

### Bitcoin

```typescript
// Raw JSON-RPC helper
async function btcRpc(method: string, params: unknown[] = []) {
  const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ jsonrpc: '2.0', id: 1, method, params }),
  });
  const { result, error } = await response.json();
  if (error) throw new Error(`${error.code}: ${error.message}`);
  return result;
}
```

## EVM RPC Methods

### Core Methods

| Category | Methods |
|----------|---------|
| **Account** | `eth_getBalance`, `eth_getCode`, `eth_getStorageAt`, `eth_getAccount`, `eth_getTransactionCount`, `eth_getProof` |
| **Block** | `eth_blockNumber`, `eth_getBlockByHash`, `eth_getBlockByNumber`, `eth_getBlockReceipts`, `eth_getBlockTransactionCountByHash`, `eth_getBlockTransactionCountByNumber` |
| **Transaction** | `eth_getTransactionByHash`, `eth_getTransactionByBlockHashAndIndex`, `eth_getTransactionByBlockNumberAndIndex`, `eth_getTransactionReceipt`, `eth_sendRawTransaction`, `eth_getRawTransactionByHash` |
| **Call & Simulate** | `eth_call`, `eth_estimateGas`, `eth_simulateV1`, `eth_callMany` |
| **Logs & Filters** | `eth_getLogs`, `eth_newFilter`, `eth_newBlockFilter`, `eth_newPendingTransactionFilter`, `eth_getFilterChanges`, `eth_getFilterLogs`, `eth_uninstallFilter` |
| **Gas & Fees** | `eth_gasPrice`, `eth_maxPriorityFeePerGas`, `eth_feeHistory`, `eth_blobBaseFee` |
| **Network** | `eth_chainId`, `eth_syncing`, `net_version`, `net_listening`, `net_peerCount`, `web3_clientVersion`, `web3_sha3` |
| **Subscription** | `eth_subscribe`, `eth_unsubscribe` |

### Code Examples

**Get balance:**

```typescript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_getBalance',
    params: ['0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045', 'latest'],
  }),
});
const { result } = await response.json();
// result: "0x..." (balance in wei, hex-encoded)
```

**Send raw transaction:**

```typescript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_sendRawTransaction',
    params: ['0xSignedTransactionData...'],
  }),
});
const { result } = await response.json();
// result: "0x..." (transaction hash)
```

**Get logs (filter by contract events):**

```typescript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_getLogs',
    params: [{
      fromBlock: '0x118C5E0',
      toBlock: '0x118C5FF',
      address: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
      topics: [
        '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef', // Transfer
      ],
    }],
  }),
});
const { result } = await response.json();
// result: Array of log objects { address, topics, data, blockNumber, transactionHash, ... }
```

**Call a contract (read-only):**

```typescript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'eth_call',
    params: [{
      to: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
      data: '0x70a08231000000000000000000000000d8dA6BF26964aF9D7eEd9e03E53415D37aA96045', // balanceOf(address)
    }, 'latest'],
  }),
});
const { result } = await response.json();
// result: ABI-encoded return value
```

### Debug, Trace & Extended Namespaces

| Namespace | Methods |
|-----------|---------|
| **debug** | `debug_traceTransaction`, `debug_traceCall`, `debug_traceBlock`, `debug_traceBlockByHash`, `debug_traceBlockByNumber`, `debug_getBadBlocks`, `debug_storageRangeAt`, `debug_getTrieFlushInterval` |
| **trace** (Erigon) | `trace_block`, `trace_call`, `trace_callMany`, `trace_filter`, `trace_rawTransaction`, `trace_replayBlockTransactions`, `trace_replayTransaction`, `trace_transaction` |
| **erigon** | `erigon_blockNumber`, `erigon_forks`, `erigon_getBlockByTimestamp`, `erigon_getBlockReceiptsByBlockHash`, `erigon_getHeaderByHash`, `erigon_getHeaderByNumber`, `erigon_getLatestLogs`, `erigon_getLogsByHash` |
| **txpool** (Geth) | `txpool_content`, `txpool_contentFrom`, `txpool_inspect`, `txpool_status` |

### Quicknode Custom Methods (qn_*)

| Method | Description |
|--------|-------------|
| `qn_getBlockFromTimestamp` | Find block closest to a Unix timestamp |
| `qn_getBlocksInTimestampRange` | List blocks within a timestamp range |
| `qn_getBlockWithReceipts` | Get block data with all transaction receipts |
| `qn_getReceipts` | Batch-fetch receipts for a block |
| `qn_broadcastRawTransaction` | Multi-region transaction broadcast |
| `qn_resolveENS` | Resolve ENS name to address (and reverse) |
| `qn_sendRawTransactionWithWebhook` | Send transaction and receive webhook notification |
| `qn_fetchNFTs` | Fetch NFTs owned by an address |
| `qn_fetchNFTCollectionDetails` | Get collection-level metadata |
| `qn_fetchNFTsByCollection` | Fetch NFTs from a specific collection |
| `qn_getTokenMetadataByContractAddress` | Token metadata by contract |
| `qn_getTokenMetadataBySymbol` | Token metadata by symbol |
| `qn_getWalletTokenBalance` | All ERC-20 balances for a wallet |
| `qn_getWalletTokenTransactions` | Token transfer history for a wallet |
| `qn_getTransactionsByAddress` | Transaction history for an address |
| `qn_getTransfersByNFT` | Transfer history for an NFT |
| `qn_verifyNFTsOwner` | Verify NFT ownership |

### Beacon Chain REST Endpoints

Beacon Chain data is accessible via REST endpoints on Ethereum endpoints.

| Category | Endpoints |
|----------|-----------|
| **Blobs** | `GET /eth/v1/beacon/blob_sidecars/{block_id}`, `GET /eth/v1/beacon/blobs/{block_id}` |
| **Blocks** | `GET /eth/v2/beacon/blocks/{block_id}`, `GET /eth/v1/beacon/blocks/{block_id}/root`, `GET /eth/v1/beacon/headers`, `GET /eth/v1/beacon/headers/{block_id}` |
| **State** | `GET /eth/v1/beacon/states/{state_id}/root`, `GET /eth/v1/beacon/states/{state_id}/fork`, `GET /eth/v1/beacon/states/{state_id}/finality_checkpoints` |
| **Validators** | `GET /eth/v1/beacon/states/{state_id}/validators`, `GET /eth/v1/beacon/states/{state_id}/validators/{validator_id}`, `GET /eth/v1/beacon/states/{state_id}/validator_balances`, `GET /eth/v1/beacon/states/{state_id}/committees`, `GET /eth/v1/beacon/states/{state_id}/sync_committees` |
| **Pending** | `GET /eth/v1/beacon/states/{state_id}/pending_deposits`, `GET /eth/v1/beacon/states/{state_id}/pending_consolidations` |
| **Rewards** | `POST /eth/v1/beacon/rewards/attestations/{epoch}`, `GET /eth/v1/beacon/rewards/blocks/{block_id}`, `POST /eth/v1/beacon/rewards/sync_committee/{block_id}` |
| **Pool** | `GET /eth/v1/beacon/pool/voluntary_exits` |
| **Config** | `GET /eth/v1/beacon/genesis`, `GET /eth/v1/config/deposit_contract`, `GET /eth/v1/config/fork_schedule`, `GET /eth/v1/config/spec` |
| **Validator Duties** | `POST /eth/v1/validator/duties/attester/{epoch}`, `GET /eth/v1/validator/duties/proposer/{epoch}`, `POST /eth/v1/validator/duties/sync/{epoch}`, `GET /eth/v1/validator/blinded_blocks/{slot}`, `GET /eth/v1/validator/sync_committee_contribution` |
| **Events** | `GET /eth/v1/events` (SSE: `head`, `block`, `attestation`, `voluntary_exit`, `finalized_checkpoint`, `chain_reorg`) |
| **Node** | `GET /eth/v1/node/peer_count`, `GET /eth/v1/node/peers`, `GET /eth/v1/node/syncing`, `GET /eth/v1/node/version` |
| **Debug** | `GET /eth/v1/debug/beacon/data_column_sidecars/{block_id}`, `GET /eth/v2/debug/beacon/states/{state_id}` |

## Solana RPC Methods

### Standard Methods

| Category | Methods |
|----------|---------|
| **Account** | `getAccountInfo`, `getMultipleAccounts`, `getProgramAccounts`, `getLargestAccounts`, `getMinimumBalanceForRentExemption` |
| **Balance** | `getBalance`, `getTokenAccountBalance`, `getTokenAccountsByOwner`, `getTokenAccountsByDelegate`, `getTokenLargestAccounts`, `getTokenSupply` |
| **Block** | `getBlock`, `getBlockCommitment`, `getBlockHeight`, `getBlockProduction`, `getBlocks`, `getBlocksWithLimit`, `getBlockTime`, `getFirstAvailableBlock` |
| **Transaction** | `getTransaction`, `getParsedTransaction`, `getTransactionCount`, `getSignaturesForAddress`, `getSignatureStatuses`, `simulateTransaction`, `sendTransaction` |
| **Slot** | `getSlot`, `getSlotLeader`, `getSlotLeaders`, `getHighestSnapshotSlot`, `getMaxRetransmitSlot`, `getMaxShredInsertSlot` |
| **Fees** | `getFeeForMessage`, `getRecentPrioritizationFees` |
| **Epoch & Inflation** | `getEpochInfo`, `getEpochSchedule`, `getInflationGovernor`, `getInflationRate`, `getInflationReward`, `getLeaderSchedule` |
| **Network** | `getClusterNodes`, `getHealth`, `getIdentity`, `getVersion`, `getGenesisHash`, `getSupply`, `getVoteAccounts`, `getStakeMinimumDelegation`, `getRecentPerformanceSamples`, `minimumLedgerSlot` |
| **Utility** | `isBlockhashValid`, `requestAirdrop` (testnet/devnet only) |

### Code Examples

**Get balance and account info:**

```typescript
import { createSolanaRpc } from '@solana/kit';
import { address } from '@solana/addresses';

const rpc = createSolanaRpc(process.env.QUICKNODE_RPC_URL!);

const balance = await rpc.getBalance(address('vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg')).send();
// balance.value: bigint (lamports)

const accountInfo = await rpc.getAccountInfo(address('vines1vzrYbzLMRdu58ou5XTby4qAqVRLmqo36NKPTg'), {
  encoding: 'base64',
}).send();
// accountInfo.value: { data, executable, lamports, owner, rentEpoch }
```

**Send transaction:**

```typescript
import { createSolanaRpc } from '@solana/kit';

const rpc = createSolanaRpc(process.env.QUICKNODE_RPC_URL!);

// Transaction must be signed before sending
const signature = await rpc.sendTransaction(signedTransactionBytes, {
  encoding: 'base64',
  skipPreflight: false,
  preflightCommitment: 'confirmed',
}).send();
// signature: base-58 encoded transaction signature
```

**Get program accounts (with filters):**

```typescript
import { createSolanaRpc } from '@solana/kit';

const rpc = createSolanaRpc(process.env.QUICKNODE_RPC_URL!);

const accounts = await rpc.getProgramAccounts(
  address('TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA'),
  {
    encoding: 'base64',
    filters: [
      { dataSize: 165n }, // Token account size
      { memcmp: { offset: 32n, bytes: 'OwnerPubkeyBase58...' as `${string}`, encoding: 'base58' } },
    ],
  }
).send();
// accounts: Array of { pubkey, account: { data, executable, lamports, owner } }
```

### WebSocket Subscriptions

| Subscription | Description |
|-------------|-------------|
| `accountSubscribe` | Monitor changes to a specific account |
| `programSubscribe` | Monitor all accounts owned by a program |
| `logsSubscribe` | Subscribe to transaction log output |
| `signatureSubscribe` | Track confirmation of a specific transaction |
| `slotSubscribe` | Monitor slot progression |
| `blockSubscribe` | Track new confirmed/finalized blocks |
| `rootSubscribe` | Receive root slot notifications |
| `slotsUpdatesSubscribe` | Detailed slot update notifications |

```typescript
import { createSolanaRpcSubscriptions } from '@solana/kit';
import { address } from '@solana/addresses';

const rpcSubscriptions = createSolanaRpcSubscriptions(process.env.QUICKNODE_WSS_URL!);

// Subscribe to account changes
const accountNotifications = await rpcSubscriptions
  .accountNotifications(address('AccountPubkey...'), { commitment: 'confirmed' })
  .subscribe({ abortSignal: AbortSignal.timeout(60_000) });

for await (const notification of accountNotifications) {
  console.log('Account changed:', notification.value.lamports);
}
```

### Solana-Specific Add-on Methods

| Category | Methods |
|----------|---------|
| **Priority Fees** | `qn_estimatePriorityFees` |
| **DAS (Digital Asset Standard)** | `getAsset`, `getAssets`, `getAssetProof`, `getAssetProofs`, `getAssetsByOwner`, `getAssetsByCreator`, `getAssetsByAuthority`, `getAssetsByGroup`, `getAssetSignatures`, `getTokenAccounts`, `getNftEditions`, `searchAssets` |
| **Jito Bundles** | `sendBundle`, `getBundleStatuses`, `getInflightBundleStatuses`, `simulateBundle`, `getTipAccounts`, `getTipFloor`, `getRegions` |
| **Jito Transaction** | `sendTransaction` (Jito-routed) |
| **Metis (Jupiter)** | `/quote`, `/swap`, `/swap-instructions`, `/tokens`, `/price`, `/new-pools`, `/program-id-to-label` |
| **Metis Limit Orders** | `/limit-orders/{pubkey}`, `/limit-orders/create`, `/limit-orders/cancel`, `/limit-orders/fee`, `/limit-orders/history`, `/limit-orders/open` |
| **Metis Pump.fun** | `/pump-fun/quote`, `/pump-fun/swap`, `/pump-fun/swap-instructions` |

## Bitcoin RPC Methods

### Standard Methods

| Category | Methods |
|----------|---------|
| **Blockchain** | `getbestblockhash`, `getblock`, `getblockchaininfo`, `getblockcount`, `getblockhash`, `getblockheader`, `getblockstats`, `getchaintips`, `getchaintxstats` |
| **Transaction** | `getrawtransaction`, `decoderawtransaction`, `decodescript`, `sendrawtransaction`, `gettxout`, `gettxoutproof`, `gettxoutsetinfo`, `testmempoolaccept`, `submitpackage` |
| **Mempool** | `getrawmempool`, `getmempoolancestors`, `getmempooldescendants`, `getmempoolinfo` |
| **Mining & Network** | `getdifficulty`, `getmininginfo`, `estimatesmartfee`, `getconnectioncount`, `getnetworkinfo`, `getmemoryinfo`, `getindexinfo` |
| **Validation** | `validateaddress`, `verifymessage` |

### Code Examples

**Get block count and block data:**

```typescript
// Get current block height
const blockCount = await btcRpc('getblockcount');
console.log('Block height:', blockCount);

// Get block hash for a specific height
const blockHash = await btcRpc('getblockhash', [blockCount]);

// Get full block data (verbosity 2 = include decoded transactions)
const block = await btcRpc('getblock', [blockHash, 2]);
console.log('Block:', {
  hash: block.hash,
  height: block.height,
  time: block.time,
  nTx: block.nTx,
  size: block.size,
});
```

**Get raw transaction:**

```typescript
// Get decoded transaction (verbose = true)
const tx = await btcRpc('getrawtransaction', [
  'txid...',
  true, // verbose: return JSON object instead of hex
]);
console.log('Transaction:', {
  txid: tx.txid,
  size: tx.size,
  vout: tx.vout.map((o: any) => ({ value: o.value, address: o.scriptPubKey?.address })),
});
```

### Ordinals, Runes & Blockbook

| Category | Methods |
|----------|---------|
| **Ordinals** | `ord_getInscription`, `ord_getInscriptions`, `ord_getInscriptionsByBlock`, `ord_getContent`, `ord_getMetadata`, `ord_getChildren`, `ord_getCollections`, `ord_getInscriptionRecursive` |
| **Sats** | `ord_getSat`, `ord_getSatAtIndex`, `ord_getSatRecursive` |
| **Runes** | `ord_getRune`, `ord_getRunes` |
| **Ordinals Utility** | `ord_getBlockHash`, `ord_getBlockInfo`, `ord_getCurrentBlockHash`, `ord_getCurrentBlockHeight`, `ord_getCurrentBlockTime`, `ord_getOutput`, `ord_getStatus`, `ord_getTx` |
| **Quicknode** | `qn_getBlockFromTimestamp`, `qn_getBlocksInTimestampRange` |
| **Blockbook** | `bb_getAddress`, `bb_getBalanceHistory`, `bb_getBlock`, `bb_getBlockHash`, `bb_getTx`, `bb_getTxSpecific`, `bb_getUTXOs`, `bb_getXPUB`, `bb_getTickers`, `bb_getTickersList` |

## WebSocket Patterns

### EVM Subscriptions

| Type | Description |
|------|-------------|
| `newHeads` | New block headers as they are mined |
| `logs` | Log entries matching a filter (address, topics) |
| `newPendingTransactions` | Transaction hashes entering the mempool |
| `syncing` | Node sync status changes |

```typescript
import { WebSocketProvider } from 'ethers';

const wsProvider = new WebSocketProvider(process.env.QUICKNODE_WSS_URL!);

// Subscribe to new blocks
wsProvider.on('block', (blockNumber) => {
  console.log('New block:', blockNumber);
});

// Subscribe to pending transactions
wsProvider.on('pending', (txHash) => {
  console.log('Pending tx:', txHash);
});

// Subscribe to contract events
const filter = {
  address: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
  topics: ['0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'], // Transfer
};
wsProvider.on(filter, (log) => {
  console.log('Transfer event:', log);
});
```

### Solana Subscriptions

See the [Solana WebSocket Subscriptions](#websocket-subscriptions) table above. Use `@solana/kit`'s `createSolanaRpcSubscriptions` for typed subscription handling.

## Batch Requests

JSON-RPC supports sending multiple calls in a single HTTP request by wrapping them in an array. This reduces round trips and is ideal for reading multiple pieces of data at once.

```typescript
const response = await fetch(process.env.QUICKNODE_RPC_URL!, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify([
    { jsonrpc: '2.0', id: 1, method: 'eth_blockNumber', params: [] },
    { jsonrpc: '2.0', id: 2, method: 'eth_gasPrice', params: [] },
    { jsonrpc: '2.0', id: 3, method: 'eth_getBalance', params: ['0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045', 'latest'] },
  ]),
});
const results = await response.json();
// results: Array of { jsonrpc, id, result } in request order
// results[0].result: block number (hex)
// results[1].result: gas price (hex)
// results[2].result: balance (hex)
```

Batch requests work the same way for Bitcoin (`getblockcount`, `getbestblockhash`, etc.) and any JSON-RPC endpoint. Solana also supports batching via the standard JSON-RPC interface.

## Best Practices

1. **Use WebSocket for subscriptions** — HTTP polling wastes requests and adds latency. Use `wss://` endpoints for real-time data (new blocks, pending transactions, account changes).
2. **Batch read requests** — Combine multiple `eth_getBalance`, `eth_call`, or similar reads into a single batch request to reduce round trips and credit usage.
3. **Cache immutable data** — Block data, transaction receipts, and finalized results never change. Cache them locally to avoid redundant calls.
4. **Retry with exponential backoff** — On 429 (rate limit) or network errors, retry with increasing delays: 1s, 2s, 4s, up to 30s max.
5. **Use archive endpoints for historical data** — Queries against old blocks require archive mode. Enable it on your Quicknode endpoint if you need `eth_getBalance` at historical blocks or Solana snapshots beyond the current epoch.
6. **Set Solana commitment levels appropriately** — Use `confirmed` for most reads, `finalized` when irreversibility matters (e.g., payment verification), and `processed` only when you need lowest latency and can handle rollbacks.
7. **Consult chain-specific llms.txt for method details** — Each chain has detailed per-method documentation at `https://www.quicknode.com/docs/{chain}/llms.txt` (e.g., `ethereum`, `solana`, `bitcoin`).

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `Method not found` | Method not available on your plan or endpoint type | Check method availability in the chain docs; some methods require add-ons or archive mode |
| `429 Too Many Requests` | Rate limit exceeded | Implement backoff/retry; batch requests; upgrade plan if persistent |
| `execution reverted` | Smart contract call failed | Check the `to` address, `data` encoding, and block tag; use `eth_estimateGas` first to catch revert reasons |
| Empty `eth_getLogs` result | Block range too narrow, wrong address, or wrong topics | Widen the block range; verify the contract address and topic hashes; check the chain |
| Solana `blockhash expired` | Transaction submitted too late after fetching blockhash | Fetch a fresh blockhash immediately before signing; use `isBlockhashValid` to check |
| Bitcoin `Work queue depth exceeded` | Too many concurrent requests | Reduce concurrency; add request queuing with rate limiting |
| WebSocket disconnects | Idle timeout or server maintenance | Implement automatic reconnection with exponential backoff; send periodic pings |

## Documentation

### Chain-Specific Docs

- **Ethereum**: https://www.quicknode.com/docs/ethereum
- **Solana**: https://www.quicknode.com/docs/solana
- **Bitcoin**: https://www.quicknode.com/docs/bitcoin
- **Polygon**: https://www.quicknode.com/docs/polygon
- **Arbitrum**: https://www.quicknode.com/docs/arbitrum
- **Base**: https://www.quicknode.com/docs/base
- **Full Chain List**: https://www.quicknode.com/chains

### LLM-Optimized Documentation (llms.txt)

- **Platform Overview**: https://www.quicknode.com/llms.txt
- **Docs Index**: https://www.quicknode.com/docs/llms.txt
- **Ethereum Methods**: https://www.quicknode.com/docs/ethereum/llms.txt
- **Solana Methods**: https://www.quicknode.com/docs/solana/llms.txt
- **Bitcoin Methods**: https://www.quicknode.com/docs/bitcoin/llms.txt
- **Pattern**: `https://www.quicknode.com/docs/{chain}/llms.txt`

### Guides

- **Quicknode Guides**: https://www.quicknode.com/guides

### Related References

- [SDK Reference](sdk-reference.md) — Quicknode SDK with typed client methods
- [Marketplace Add-ons](marketplace-addons.md) — Token API, NFT API, DAS, Jito, trace/debug
- [Yellowstone gRPC](yellowstone-grpc-reference.md) — Solana Geyser streaming via gRPC

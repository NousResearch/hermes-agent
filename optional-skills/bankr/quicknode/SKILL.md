---
name: quicknode
description: Blockchain RPC and data access via Quicknode. Use when an agent needs to read onchain data (balances, token prices, transaction status, gas estimates, block data) across Base, Ethereum, Polygon, Solana, or Unichain. Supports both API key access and x402 wallet-based pay-per-request access with no account needed. Triggers on mentions of RPC, blockchain data, onchain queries, token balances, gas estimation, block number, transaction receipt, Quicknode, or x402.
---

# Quicknode: Blockchain Data Access for Agents

Quicknode provides high-performance RPC endpoints across 77+ blockchain networks including all chains Bankr supports: Base, Ethereum, Polygon, Solana, and Unichain.

Two ways to access:

- **API key**: Create a Quicknode account, get an endpoint URL with auth baked in. Full access to all products and settings.
- **x402 (no account needed)**: Any wallet with USDC can authenticate and pay per request. Install `@quicknode/x402` and start querying immediately.

## x402 Access (Recommended for Agents)

x402 is ideal for autonomous agents. No signup, no API keys. Pay with USDC on Base, Polygon, or Solana.

```typescript
import { createQuicknodeX402Client } from "@quicknode/x402";

const client = await createQuicknodeX402Client({
  baseUrl: 'https://x402.quicknode.com',
  network: "eip155:84532",   // pay on Base Sepolia (testnet)
  evmPrivateKey: process.env.PRIVATE_KEY,
  preAuth: true,             // pre-authenticates via SIWX for faster payment flow
});

// Pay on Base, query any chain
const res = await client.fetch("https://x402.quicknode.com/ethereum-mainnet", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ jsonrpc: "2.0", method: "eth_blockNumber", params: [], id: 1 }),
});
```

Install: `npm install @quicknode/x402`

Credit pricing:
- Testnet: 100 credits for $0.01 USDC
- Mainnet: 1,000,000 credits for $10 USDC
- 1 credit per successful JSON-RPC response

Full x402 docs: https://x402.quicknode.com/llms.txt

## API Key Access

Quicknode endpoints include authentication in the URL:

```
https://{ENDPOINT_NAME}.{NETWORK}.quiknode.pro/{API_KEY}/
```

```typescript
import { createPublicClient, http } from "viem";
import { base } from "viem/chains";

const client = createPublicClient({
  chain: base,
  transport: http(process.env.QUICKNODE_RPC_URL),
});

const block = await client.getBlockNumber();
```

## Common Agent Operations

### Check Native Balance (EVM)

```typescript
const balance = await client.getBalance({ address: "0x..." });
```

Or raw RPC:
```json
{ "jsonrpc": "2.0", "method": "eth_getBalance", "params": ["0x...", "latest"], "id": 1 }
```

### Check ERC-20 Token Balance (EVM)

Use `eth_call` with the ERC-20 `balanceOf(address)` selector (`0x70a08231`):

```json
{
  "jsonrpc": "2.0",
  "method": "eth_call",
  "params": [{
    "to": "0xTOKEN_CONTRACT",
    "data": "0x70a08231000000000000000000000000WALLET_ADDRESS_NO_0x"
  }, "latest"],
  "id": 1
}
```

### Get Gas Estimate (EVM)

```json
{ "jsonrpc": "2.0", "method": "eth_gasPrice", "params": [], "id": 1 }
```

### Check Transaction Status (EVM)

```json
{ "jsonrpc": "2.0", "method": "eth_getTransactionReceipt", "params": ["0xTX_HASH"], "id": 1 }
```

### Solana Balance

```json
{ "jsonrpc": "2.0", "method": "getBalance", "params": ["WALLET_PUBKEY"], "id": 1 }
```

### Solana Token Accounts

```json
{
  "jsonrpc": "2.0",
  "method": "getTokenAccountsByOwner",
  "params": [
    "WALLET_PUBKEY",
    { "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" },
    { "encoding": "jsonParsed" }
  ],
  "id": 1
}
```

## Quicknode Marketplace Add-ons

Quicknode endpoints can be enhanced with Marketplace add-ons. Relevant ones for trading agents:

- **Token API**: `qn_getWalletTokenBalance` returns all ERC-20 balances for a wallet in one call. No need to query each token contract individually.
- **NFT API**: `qn_fetchNFTs` returns NFTs owned by an address with metadata.
- **Solana Priority Fee API**: `qn_estimatePriorityFees` returns recommended priority fees based on recent network activity. Useful for ensuring transactions land quickly.
- **Solana DAS API**: Query compressed NFTs, fungible tokens, and digital assets via methods like `getAssetsByOwner` and `searchAssets`.
- **Metis - Solana Trading API**: Jupiter-powered token swaps on Solana. Get quotes and execute swaps via `quoteGet` and `swapPost` endpoints. Docs: https://www.quicknode.com/docs/solana/metis-overview

See all add-ons: https://marketplace.quicknode.com/

These add-ons are available on API key endpoints. Enable them in the Quicknode dashboard.

## Supported Networks

All Bankr-supported chains are available on Quicknode:

| Chain | x402 Network Slug | API Key Docs |
|-------|-------------------|--------------|
| Base | `base-mainnet` | https://www.quicknode.com/docs/base |
| Ethereum | `ethereum-mainnet` | https://www.quicknode.com/docs/ethereum |
| Polygon | `polygon-mainnet` | https://www.quicknode.com/docs/polygon |
| Solana | `solana-mainnet` | https://www.quicknode.com/docs/solana |
| Unichain | `unichain-mainnet` | https://www.quicknode.com/docs/unichain |

x402 base URL: `https://x402.quicknode.com/{network-slug}`

See full list of supported chains: https://www.quicknode.com/chains

## Error Handling

- **429 Too Many Requests**: Back off and retry. Use exponential backoff.
- **402 Payment Required** (x402): Credits depleted. `@quicknode/x402` handles this automatically by triggering a new USDC payment.
- **JSON-RPC errors** (e.g., `-32000`): Method-specific errors. Check params and retry.

## Resources

- AI & Agents docs: https://www.quicknode.com/docs/build-with-ai
- Full RPC documentation (all chains): https://www.quicknode.com/docs/llms.txt
- x402 technical details: https://x402.quicknode.com/llms.txt
- Code examples (x402): https://github.com/quiknode-labs/qn-x402-examples
- Marketplace add-ons: https://marketplace.quicknode.com
- Full Quicknode skill with extended references: https://github.com/quiknode-labs/blockchain-skills/tree/main/skills/quicknode-skill

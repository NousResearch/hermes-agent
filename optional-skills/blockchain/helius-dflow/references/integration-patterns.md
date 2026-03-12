# Integration Patterns — Helius x DFlow

## What This Covers

End-to-end patterns for combining DFlow trading APIs with Helius infrastructure. These patterns show how the two systems connect at the transaction, data, and monitoring layers.

**DFlow** handles trade routing and execution — getting quotes, building swap transactions, prediction market orders, and market data streaming.

**Helius** handles infrastructure — transaction submission (Sender), fee optimization (Priority Fees), token/NFT data (DAS), real-time on-chain monitoring (WebSockets), shred-level streaming (LaserStream), and wallet intelligence (Wallet API).

---

## Pattern 1: DFlow Imperative Swap via Helius Sender

The most critical integration. DFlow's `/order` returns a base64-encoded transaction. Submit it via Helius Sender for optimal block inclusion.

### Flow

1. Get a quote from DFlow `/order`
2. Deserialize the returned base64 transaction
3. Sign the transaction
4. Submit via Helius Sender endpoint
5. Confirm the transaction

### TypeScript Example (@solana/web3.js)

```typescript
import {
  Connection,
  VersionedTransaction,
  Keypair,
} from '@solana/web3.js';

const DFLOW_API = 'https://dev-quote-api.dflow.net'; // or production endpoint
const SENDER_URL = 'https://sender.helius-rpc.com/fast';

async function swapViaDFlowAndSender(
  keypair: Keypair,
  inputMint: string,
  outputMint: string,
  amount: string, // atomic units
  slippageBps: number | 'auto' = 'auto'
): Promise<string> {
  // 1. Get quote and transaction from DFlow
  const params = new URLSearchParams({
    userPublicKey: keypair.publicKey.toBase58(),
    inputMint,
    outputMint,
    amount,
    slippageBps: slippageBps.toString(),
    priorityLevel: 'high', // DFlow handles priority fee
  });

  const quoteRes = await fetch(`${DFLOW_API}/order?${params}`);
  const quote = await quoteRes.json();

  if (quote.error) throw new Error(`DFlow error: ${quote.error}`);

  // 2. Deserialize the transaction
  const txBuffer = Buffer.from(quote.transaction, 'base64');
  const transaction = VersionedTransaction.deserialize(txBuffer);

  // 3. Sign
  transaction.sign([keypair]);

  // 4. Submit via Helius Sender
  const sendRes = await fetch(SENDER_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      jsonrpc: '2.0',
      id: Date.now().toString(),
      method: 'sendTransaction',
      params: [
        Buffer.from(transaction.serialize()).toString('base64'),
        { encoding: 'base64', skipPreflight: true, maxRetries: 0 }
      ]
    })
  });

  const sendResult = await sendRes.json();
  if (sendResult.error) throw new Error(`Sender error: ${sendResult.error.message}`);

  const signature = sendResult.result;

  // 5. Handle async execution if needed
  if (quote.executionMode === 'async') {
    return await pollOrderStatus(signature);
  }

  return signature;
}

async function pollOrderStatus(
  signature: string,
  lastValidBlockHeight?: number
): Promise<{ signature: string; fills?: any[] }> {
  const maxAttempts = 60; // 2 minutes at 2s intervals
  for (let i = 0; i < maxAttempts; i++) {
    const url = new URL(`${DFLOW_API}/order-status`);
    url.searchParams.set('signature', signature);
    if (lastValidBlockHeight) {
      url.searchParams.set('lastValidBlockHeight', lastValidBlockHeight.toString());
    }

    const res = await fetch(url.toString());
    const result = await res.json();

    switch (result.status) {
      case 'closed':
        // Success — check fills for execution details
        return { signature, fills: result.fills };
      case 'expired':
        // Blockhash expired — caller should rebuild and resubmit
        throw new Error('Order expired: rebuild transaction with fresh blockhash and retry');
      case 'failed':
        // Execution failed — check error, verify market is still active
        throw new Error(`Order failed: ${result.error || 'unknown error'}`);
      case 'open':
      case 'pendingClose':
      case 'pending':
        // Still in progress — keep polling
        break;
      default:
        throw new Error(`Unknown order status: ${result.status}`);
    }

    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  throw new Error('Order status polling timeout after 2 minutes');
}
```

### Key Points

- **Helius Sender** dual-routes to validators AND Jito for maximum block inclusion probability
- DFlow's `/order` includes priority fees when you pass `priorityLevel` — no need to add your own compute budget instructions
- Always use `skipPreflight: true` and `maxRetries: 0` with Sender
- For `executionMode: "async"`, poll `/order-status` — the trade settles across multiple transactions
- Use Sender's HTTPS endpoint (`sender.helius-rpc.com/fast`) for browser apps, regional HTTP endpoints for backends

---

## CORS Proxy for Web Apps

The DFlow Trading API does not set CORS headers. Any browser `fetch` to `/order`, `/intent`, or `/order-status` will fail. You MUST proxy these calls through your own backend. Helius APIs (Sender, DAS, RPC) do NOT have this restriction — they can be called directly from the browser.

### Express / Node.js Proxy

```typescript
import express from 'express';

const app = express();
app.use(express.json());

const DFLOW_API = process.env.DFLOW_API_URL || 'https://dev-quote-api.dflow.net';

// Proxy DFlow /order requests
app.get('/api/dflow/order', async (req, res) => {
  const params = new URLSearchParams(req.query as Record<string, string>);
  const response = await fetch(`${DFLOW_API}/order?${params}`);
  const data = await response.json();
  res.json(data);
});

// Proxy DFlow /intent requests
app.get('/api/dflow/intent', async (req, res) => {
  const params = new URLSearchParams(req.query as Record<string, string>);
  const response = await fetch(`${DFLOW_API}/intent?${params}`);
  const data = await response.json();
  res.json(data);
});

// Proxy DFlow /submit-intent requests
app.post('/api/dflow/submit-intent', async (req, res) => {
  const response = await fetch(`${DFLOW_API}/submit-intent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body),
  });
  const data = await response.json();
  res.json(data);
});

// Proxy DFlow /order-status requests
app.get('/api/dflow/order-status', async (req, res) => {
  const params = new URLSearchParams(req.query as Record<string, string>);
  const response = await fetch(`${DFLOW_API}/order-status?${params}`);
  const data = await response.json();
  res.json(data);
});

app.listen(3001);
```

### Vercel Edge Function / Next.js Route Handler

```typescript
// app/api/dflow/order/route.ts (Next.js App Router)
import { NextRequest, NextResponse } from 'next/server';

const DFLOW_API = process.env.DFLOW_API_URL || 'https://dev-quote-api.dflow.net';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const response = await fetch(`${DFLOW_API}/order?${searchParams.toString()}`);
  const data = await response.json();
  return NextResponse.json(data);
}
```

### What Does NOT Need a Proxy

- **Helius Sender** (`sender.helius-rpc.com/fast`) — has CORS headers, call directly from browser
- **Helius RPC** (`mainnet.helius-rpc.com`) — has CORS headers
- **Helius DAS API** — has CORS headers
- **DFlow WebSockets** (`wss://prediction-markets-api.dflow.net`) — WebSocket protocol, no CORS issue
- **Proof KYC verify** (`proof.dflow.net/verify/`) — read-only GET, typically no CORS issue

---

## Pattern 2: Token List from Helius DAS for Swap UI

Build a rich token selector by combining DFlow's supported tokens with Helius DAS metadata.

### Flow

1. Get the user's wallet tokens via Helius DAS
2. Enrich with metadata (icons, names, prices)
3. Build the "From" token list (user's holdings) and "To" token list (supported outputs)

### TypeScript Example

```typescript
// Get all tokens in user's wallet (both fungible and NFTs)
// Use the getAssetsByOwner MCP tool or call the DAS API:
const response = await fetch(`https://mainnet.helius-rpc.com/?api-key=${HELIUS_API_KEY}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'getAssetsByOwner',
    params: {
      ownerAddress: walletAddress,
      displayOptions: { showFungible: true, showNativeBalance: true },
    }
  })
});

const { result } = await response.json();

// Filter to fungible tokens for the "From" list
const fromTokens = result.items
  .filter(asset => asset.interface === 'FungibleToken' || asset.interface === 'FungibleAsset')
  .map(asset => ({
    mint: asset.id,
    symbol: asset.content?.metadata?.symbol,
    name: asset.content?.metadata?.name,
    image: asset.content?.links?.image,
    balance: asset.token_info?.balance,
    decimals: asset.token_info?.decimals,
    priceUsd: asset.token_info?.price_info?.price_per_token,
  }));

// "To" list: fixed set of known output tokens
const toTokens = [
  { mint: 'So11111111111111111111111111111111111111112', symbol: 'SOL' },
  { mint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', symbol: 'USDC' },
  // ... add more supported tokens
];
```

---

## Pattern 3: Trade Confirmation via Helius WebSockets

After submitting a DFlow trade via Sender, monitor confirmation in real time using Helius Enhanced WebSockets.

### Flow

1. Submit trade (Pattern 1)
2. Subscribe to signature confirmation via Helius WebSocket
3. Optionally parse the confirmed transaction for human-readable details

### TypeScript Example

```typescript
import WebSocket from 'ws';

function monitorTradeConfirmation(
  signature: string,
  heliusApiKey: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(`wss://mainnet.helius-rpc.com/?api-key=${heliusApiKey}`);

    ws.on('open', () => {
      // Subscribe to transaction updates for the signature
      ws.send(JSON.stringify({
        jsonrpc: '2.0',
        id: 1,
        method: 'transactionSubscribe',
        params: [
          { signature },
          {
            commitment: 'confirmed',
            encoding: 'jsonParsed',
            maxSupportedTransactionVersion: 0,
          }
        ]
      }));
    });

    ws.on('message', (data) => {
      const message = JSON.parse(data.toString());
      if (message.result !== undefined) return; // subscription confirmation

      // Transaction confirmed
      console.log('Trade confirmed:', message);
      ws.close();
      resolve();
    });

    ws.on('error', reject);

    // Timeout after 60 seconds
    setTimeout(() => {
      ws.close();
      reject(new Error('Confirmation timeout'));
    }, 60_000);
  });
}
```

---

## Pattern 4: Low-Latency Trading with LaserStream

For latency-critical trading (bots, liquidation engines, HFT), use Helius LaserStream for shred-level on-chain data alongside DFlow for execution.

DFlow themselves use LaserStream — it "saved over eight hours of recurring engineering overhead, maintained 100% uptime with uninterrupted data streaming, and improved quote speeds with faster transaction confirmations."

### Use Cases

- **Detect trading opportunities** before competitors by monitoring account state changes at shred level
- **Track order fills** in real time by subscribing to relevant program accounts
- **Monitor liquidity changes** across DEXs for better routing decisions
- **Confirm your own trades** at the lowest possible latency

### Architecture

```
LaserStream (gRPC) ──> Your Bot ──> DFlow /order ──> Helius Sender
     │                    │
     │  shred-level       │  market signals
     │  account data      │  trigger trades
     │                    │
     └──> Fill detection  └──> Order submission
```

### TypeScript Example

```typescript
import { subscribe, CommitmentLevel } from 'helius-laserstream';

const config = {
  apiKey: process.env.HELIUS_API_KEY,
  endpoint: 'https://laserstream-mainnet-ewr.helius-rpc.com', // choose closest region
};

// Monitor token program for relevant account changes
const request = {
  accounts: {
    'token-accounts': {
      owner: ['TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA'],
      filters: {
        token_account_state: true,
      },
      nonempty_txn_signature: true,
    }
  },
  commitment: CommitmentLevel.CONFIRMED,
};

await subscribe(
  config,
  request,
  async (data) => {
    // Analyze account change for trading signal
    const signal = analyzeAccountChange(data);
    if (signal) {
      // Execute trade via DFlow + Sender (Pattern 1)
      await swapViaDFlowAndSender(keypair, signal.inputMint, signal.outputMint, signal.amount);
    }
  },
  (error) => {
    console.error('LaserStream error:', error);
  }
);
```

### LaserStream vs DFlow WebSockets

| | LaserStream | DFlow WebSockets |
|---|---|---|
| Data | Raw on-chain (transactions, accounts) | Market-level (prices, orderbook, trades) |
| Latency | Shred-level (lowest possible) | Market-level |
| Use case | Detecting on-chain events, HFT, bots | Price feeds, trading UIs |
| Plan required | Professional ($999/mo) | DFlow API key |

**Use both together** for the most competitive trading systems: LaserStream for on-chain signals and fill detection, DFlow WebSockets for market data and orderbook state.

---

## Pattern 5: Portfolio + Trading Dashboard

Combine Helius wallet intelligence with DFlow trading for a unified dashboard.

### Architecture

1. **Holdings**: Helius `getWalletBalances` for portfolio overview
2. **Token metadata**: Helius DAS `getAssetsByOwner` with `showFungible: true` for token details, icons, and prices
3. **Live prices**: DFlow WebSockets for real-time price updates on prediction market positions
4. **Trading**: DFlow `/order` + Helius Sender for executing swaps
5. **History**: Helius `parseTransactions` for human-readable trade history

### Flow

```
Helius Wallet API ──> Portfolio Display
Helius DAS API ────> Token Metadata + Prices
DFlow WebSockets ──> Live Market Prices
DFlow /order ──────> Trade Execution ──> Helius Sender
Helius parseTransactions ──> Trade History
```

---

## Pattern 6: Trading Bot with Price Signals

Build an automated trading bot that reacts to DFlow WebSocket price signals and executes via Helius Sender.

### Architecture

```
DFlow WebSockets ──> Price Signal Detection ──> DFlow /order ──> Helius Sender
                                                                      │
LaserStream ────────> Fill Confirmation ────────────────────────────────
```

### Flow

1. Connect to DFlow WebSockets for real-time prediction market prices
2. Implement signal detection logic (price thresholds, momentum, etc.)
3. On signal: get quote from DFlow, submit via Helius Sender
4. Monitor fill via LaserStream (fastest) or poll `/order-status`
5. Update portfolio state

### Key Considerations

- Use DFlow WebSocket `prices` channel for market data
- Use LaserStream for fill detection (shred-level latency) or `/order-status` polling (simpler)
- Always check market `status === 'active'` before submitting orders
- For prediction markets, ensure Proof KYC is completed before first trade
- Implement circuit breakers (max loss, max trades per period)
- Handle the Thursday 3-5 AM ET maintenance window for prediction markets

---

## Common Mistakes Across All Patterns

- Submitting DFlow transactions to raw RPC instead of Helius Sender
- Not using `skipPreflight: true` with Sender (transactions get rejected)
- Forgetting to poll `/order-status` for async trades (trade appears to hang)
- Using LaserStream for simple UI features that Enhanced WebSockets can handle (unnecessary cost)
- Confusing DFlow WebSockets (market data) with Helius WebSockets (on-chain data)
- Not implementing retry logic for Sender submissions
- Hardcoding priority fees instead of using DFlow's `priorityLevel` parameter or Helius `getPriorityFeeEstimate`

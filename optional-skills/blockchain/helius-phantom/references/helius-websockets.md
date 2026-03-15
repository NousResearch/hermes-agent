# WebSockets — Real-Time Solana Streaming

## Two WebSocket Tiers

Helius provides two WebSocket tiers on the same endpoint:

| | Standard WebSockets | Enhanced WebSockets |
|---|---|---|
| Methods | Solana native: `accountSubscribe`, `logsSubscribe`, `programSubscribe`, `signatureSubscribe`, `slotSubscribe`, `rootSubscribe` | `transactionSubscribe`, `accountSubscribe` with advanced filtering and auto-parsing |
| Plan required | Free+ (all plans) | Business+ |
| Filtering | Basic (single account or program) | Up to 50,000 addresses per filter, include/exclude/required logic |
| Parsing | Raw Solana data | Automatic transaction parsing (type, description, tokenTransfers) |
| Latency | Good | Faster (powered by LaserStream infrastructure) |
| Credits | 3 credits per 0.1 MB streamed | 3 credits per 0.1 MB streamed |
| Max connections | Plan-dependent | 250 concurrent (Business/Professional) |

Both tiers use the same endpoints:
- **Mainnet**: `wss://mainnet.helius-rpc.com/?api-key=YOUR_API_KEY`
- **Devnet**: `wss://devnet.helius-rpc.com/?api-key=YOUR_API_KEY`

**10-minute inactivity timeout** — send pings every 30 seconds to keep connections alive.

## MCP Tools

Enhanced WebSocket operations have MCP tools. Like LaserStream, these are config generators — WebSocket connections can't run over MCP stdio. The workflow is: generate config via MCP tool, then embed the code in the user's application.

| MCP Tool | What It Does |
|---|---|
| `transactionSubscribe` | Generates Enhanced WS subscription config + code for transaction streaming with filters |
| `accountSubscribe` | Generates Enhanced WS subscription config + code for account monitoring |
| `getEnhancedWebSocketInfo` | Returns endpoint, capabilities, plan requirements |

ALWAYS use these MCP tools first when the user needs Enhanced WebSocket subscriptions — they validate parameters, warn about config issues, and produce correct code.

Standard WebSocket subscriptions do not have MCP tools — generate the code directly using the patterns in this file.

## Choosing the Right Approach

| You want to... | Use |
|---|---|
| Monitor a specific account for changes | Standard `accountSubscribe` (Free+) or Enhanced `accountSubscribe` (Business+) |
| Stream transactions for specific accounts/programs | Enhanced `transactionSubscribe` (Business+) |
| Monitor program account changes | Standard `programSubscribe` (Free+) |
| Watch for transaction confirmation | Standard `signatureSubscribe` (Free+) |
| Track slot/root progression | Standard `slotSubscribe` / `rootSubscribe` (Free+) |
| Monitor transaction logs | Standard `logsSubscribe` (Free+) |
| Stream with advanced filtering (50K addresses) | Enhanced `transactionSubscribe` (Business+) |
| Need historical replay or 10M+ addresses | LaserStream (see Helius docs at `docs.helius.dev`) |
| Need push notifications without persistent connection | Webhooks (see Helius docs at `docs.helius.dev`) |

## Connection Pattern

All WebSocket code follows the same structure. ALWAYS include ping keepalive:

```typescript
const WebSocket = require('ws');

const ws = new WebSocket('wss://mainnet.helius-rpc.com/?api-key=YOUR_API_KEY');

ws.on('open', () => {
  console.log('Connected');

  // Send subscription request
  ws.send(JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'SUBSCRIPTION_METHOD',
    params: [/* ... */]
  }));

  // Keep connection alive — 10-minute inactivity timeout
  setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.ping();
  }, 30000);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());

  // First message is subscription confirmation
  if (msg.result !== undefined) {
    console.log('Subscribed, ID:', msg.result);
    return;
  }

  // Subsequent messages are notifications
  if (msg.method) {
    console.log('Notification:', msg.params);
  }
});

ws.on('close', () => console.log('Disconnected'));
ws.on('error', (err) => console.error('Error:', err));
```

## Enhanced WebSockets

### transactionSubscribe

Stream real-time transactions with advanced filtering. Use the `transactionSubscribe` MCP tool to generate the config, or build manually:

**Filter parameters:**
- `accountInclude`: transactions involving ANY of these addresses (OR logic, up to 50K)
- `accountExclude`: exclude transactions with these addresses (up to 50K)
- `accountRequired`: transactions must involve ALL of these addresses (AND logic, up to 50K)
- `vote`: include vote transactions (default: false)
- `failed`: include failed transactions (default: false)
- `signature`: filter to a specific transaction signature

**Options:**
- `commitment`: `processed`, `confirmed`, `finalized`
- `encoding`: `base58`, `base64`, `jsonParsed`
- `transactionDetails`: `full`, `signatures`, `accounts`, `none`
- `showRewards`: include reward data
- `maxSupportedTransactionVersion`: set to `0` to receive both legacy and versioned transactions (required when `transactionDetails` is `accounts` or `full`)

```typescript
ws.on('open', () => {
  ws.send(JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'transactionSubscribe',
    params: [
      {
        accountInclude: ['JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4'],
        vote: false,
        failed: false
      },
      {
        commitment: 'confirmed',
        encoding: 'jsonParsed',
        transactionDetails: 'full',
        maxSupportedTransactionVersion: 0
      }
    ]
  }));

  setInterval(() => ws.ping(), 30000);
});

ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());
  if (msg.method === 'transactionNotification') {
    const tx = msg.params.result;
    console.log('Signature:', tx.signature);
    console.log('Slot:', tx.slot);
    // tx.transaction contains full parsed transaction data
  }
});
```

**Notification payload:**

```json
{
  "method": "transactionNotification",
  "params": {
    "subscription": 4743323479349712,
    "result": {
      "transaction": {
        "transaction": ["base64data...", "base64"],
        "meta": {
          "err": null,
          "fee": 5000,
          "preBalances": [28279852264, 158122684, 1],
          "postBalances": [28279747264, 158222684, 1],
          "innerInstructions": [],
          "logMessages": ["Program 111... invoke [1]", "Program 111... success"],
          "preTokenBalances": [],
          "postTokenBalances": [],
          "computeUnitsConsumed": 0
        }
      },
      "signature": "5moMXe6VW7L7...",
      "slot": 224341380
    }
  }
}
```

### accountSubscribe (Enhanced)

Monitor account data/balance changes with enhanced performance:

```typescript
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 1,
  method: 'accountSubscribe',
  params: [
    'ACCOUNT_ADDRESS',
    { encoding: 'jsonParsed', commitment: 'confirmed' }
  ]
}));

ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());
  if (msg.method === 'accountNotification') {
    const value = msg.params.result.value;
    console.log('Lamports:', value.lamports);
    console.log('Owner:', value.owner);
    console.log('Data:', value.data);
  }
});
```

## Standard WebSockets

Available on all plans. These are standard Solana RPC WebSocket methods.

### Supported Methods

| Method | What It Does |
|---|---|
| `accountSubscribe` | Notifications when an account's lamports or data change |
| `logsSubscribe` | Transaction log messages (filter by address or `all`) |
| `programSubscribe` | Notifications when accounts owned by a program change |
| `signatureSubscribe` | Notification when a specific transaction is confirmed |
| `slotSubscribe` | Notifications on slot progression |
| `rootSubscribe` | Notifications when a new root is set |

Each has a corresponding `*Unsubscribe` method (e.g., `accountUnsubscribe`).

### Unsupported (Unstable) Methods

These are unstable in the Solana spec and NOT supported on Helius:
- `blockSubscribe` / `blockUnsubscribe`
- `slotsUpdatesSubscribe` / `slotsUpdatesUnsubscribe`
- `voteSubscribe` / `voteUnsubscribe`

### accountSubscribe (Standard)

```typescript
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 1,
  method: 'accountSubscribe',
  params: [
    'ACCOUNT_ADDRESS',
    {
      encoding: 'jsonParsed', // base58, base64, base64+zstd, jsonParsed
      commitment: 'confirmed' // finalized (default), confirmed, processed
    }
  ]
}));
```

### programSubscribe

Monitor all accounts owned by a program:

```typescript
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 1,
  method: 'programSubscribe',
  params: [
    'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA', // Token Program
    {
      encoding: 'jsonParsed',
      commitment: 'confirmed'
    }
  ]
}));
```

### logsSubscribe

Subscribe to transaction logs:

```typescript
// All logs
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 1,
  method: 'logsSubscribe',
  params: ['all', { commitment: 'confirmed' }]
}));

// Logs mentioning a specific address
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 1,
  method: 'logsSubscribe',
  params: [
    { mentions: ['PROGRAM_OR_ACCOUNT_ADDRESS'] },
    { commitment: 'confirmed' }
  ]
}));
```

### signatureSubscribe

Watch for a specific transaction to confirm:

```typescript
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 1,
  method: 'signatureSubscribe',
  params: [
    'TRANSACTION_SIGNATURE',
    { commitment: 'confirmed' }
  ]
}));

// Auto-unsubscribes after first notification
```

### slotSubscribe

```typescript
ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 1,
  method: 'slotSubscribe',
  params: []
}));
```

## Reconnection Pattern

WebSocket connections can drop. ALWAYS implement auto-reconnection with exponential backoff:

- On `close`: clear ping timer, wait `reconnectDelay` (start 1s, double each attempt, cap at 30s), then reconnect
- On successful `open`: reset delay to 1s, restart 30s ping timer, re-send subscription
- On `error`: log and let `close` handler trigger reconnect

## Common Patterns

All Enhanced `transactionSubscribe` patterns use the same shape — vary the filter addresses. Use the `transactionSubscribe` MCP tool to generate correct configs:

| Use Case | Filter | Key Addresses |
|---|---|---|
| Jupiter swaps | `accountInclude` | `JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4` |
| Magic Eden NFT sales | `accountInclude` | `M2mx93ekt1fmXSVkTrUL9xVFHkmME8HTUi5Cyc5aF7K` |
| Pump AMM data | `accountInclude` | `pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA` |
| Wallet activity (Enhanced) | `accountInclude` | `[WALLET_ADDRESS]` |
| Txs between two wallets | `accountRequired` (AND logic) | `[WALLET_A, WALLET_B]` |

For Standard WebSockets:
- **Wallet balance/data changes**: `accountSubscribe` with `[address, { encoding: 'jsonParsed', commitment: 'confirmed' }]`
- **Token program activity**: `programSubscribe` with `[TOKEN_PROGRAM_ID, { encoding: 'jsonParsed', commitment: 'confirmed' }]`

## WebSockets vs LaserStream vs Webhooks

| Feature | Standard WS | Enhanced WS | LaserStream | Webhooks |
|---|---|---|---|---|
| Plan | Free+ | Business+ | Professional+ | Free+ |
| Protocol | WebSocket | WebSocket | gRPC | HTTP POST |
| Latency | Good | Faster | Fastest (shred-level) | Variable |
| Max addresses | 1 per subscription | 50K per filter | 10M | 100K per webhook |
| Historical replay | No | No | Yes (24 hours) | No |
| Auto-reconnect | Manual | Manual | Built-in via SDK | N/A |
| Transaction parsing | No | Yes (auto) | No (raw data) | Yes (enhanced type) |
| Requires public endpoint | No | No | No | Yes |

**Use Standard WebSockets when**: you're on a Free/Developer plan, need basic account/program monitoring, or are using existing Solana WebSocket code.

**Use Enhanced WebSockets when**: you need transaction filtering with multiple addresses, auto-parsed transaction data, or monitoring DEX/NFT activity on Business+ plan.

**Use LaserStream when**: you need the lowest latency, historical replay, or are processing high data volumes. See Helius docs at `docs.helius.dev`.

**Use Webhooks when**: you want push notifications without maintaining a connection. See Helius docs at `docs.helius.dev`.

## Best Practices

- ALWAYS send pings every 30 seconds — 10-minute inactivity timeout disconnects silently
- ALWAYS implement auto-reconnection with exponential backoff
- Use `accountRequired` for stricter matching (AND logic) vs `accountInclude` (OR logic)
- Set `vote: false` and `failed: false` to reduce noise unless you specifically need those
- Set `maxSupportedTransactionVersion: 0` to receive both legacy and versioned transactions
- Use `jsonParsed` encoding for human-readable data; `base64` for raw processing
- Use the MCP tools (`transactionSubscribe`, `accountSubscribe`) to generate correct configs before embedding in user code
- For standard WebSockets, use `confirmed` commitment for most use cases

## Common Mistakes

- Not implementing ping keepalive — connection silently drops after 10 minutes of inactivity
- Not implementing auto-reconnection — WebSocket disconnects are normal and expected
- Confusing `accountInclude` (OR — any match) with `accountRequired` (AND — all must match)
- Not setting `maxSupportedTransactionVersion: 0` — misses versioned transactions
- Using Enhanced WebSocket features on Free/Developer plans — requires Business+
- Subscribing without filters on `transactionSubscribe` — streams ALL network transactions, extreme volume
- Using `blockSubscribe`, `slotsUpdatesSubscribe`, or `voteSubscribe` — these are unstable and not supported on Helius
- Not handling the subscription confirmation message (first message has `result` field, not notification data)

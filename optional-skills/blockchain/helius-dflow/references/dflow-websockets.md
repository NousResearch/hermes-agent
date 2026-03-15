# DFlow WebSockets — Real-Time Market Data

## What This Covers

Real-time streaming of prediction market data from DFlow. Use for live price tickers, trade feeds, orderbook depth, and market monitoring.

This is different from Helius WebSockets (see `references/helius-websockets.md`), which stream on-chain data like transaction confirmations and account changes. DFlow WebSockets stream market-level data — prices, trades, and orderbooks — specific to DFlow's prediction markets.

For the lowest-latency on-chain data (shred-level), see `references/helius-laserstream.md`.

## Connection

* WebSocket URL: `wss://prediction-markets-api.dflow.net`
* A valid API key is required via the `x-api-key` header. Unlike the REST Trade API and Metadata API (which have keyless dev endpoints), WebSockets always require a key. Apply for one at `https://pond.dflow.net/build/api-key`.

```typescript
const ws = new WebSocket('wss://prediction-markets-api.dflow.net', {
  headers: { 'x-api-key': process.env.DFLOW_API_KEY }
});
```

## Channels

| Channel | Description |
|---|---|
| `prices` | Real-time bid/ask price updates for markets |
| `trades` | Real-time trade execution updates |
| `orderbook` | Real-time orderbook depth updates for markets |

## Subscription Management

Send JSON messages to subscribe or unsubscribe.

### Subscribe to all markets

```json
{ "type": "subscribe", "channel": "prices", "all": true }
```

### Subscribe to specific markets

```json
{ "type": "subscribe", "channel": "prices", "tickers": ["MARKET_TICKER_1", "MARKET_TICKER_2"] }
```

### Unsubscribe

```json
{ "type": "unsubscribe", "channel": "prices", "all": true }
```

```json
{ "type": "unsubscribe", "channel": "prices", "tickers": ["MARKET_TICKER_1"] }
```

### Subscription Rules

* `"all": true` clears specific ticker subscriptions for that channel.
* Specific tickers disable "all" mode for that channel.
* Each channel maintains independent subscription state.
* Unsubscribing from specific tickers has no effect under "all" mode. Unsubscribe from "all" first.

## Implementation Pattern

```typescript
import WebSocket from 'ws';

function connectDFlowWebSocket(apiKey: string) {
  const ws = new WebSocket('wss://prediction-markets-api.dflow.net', {
    headers: { 'x-api-key': apiKey }
  });

  ws.on('open', () => {
    console.log('Connected to DFlow WebSocket');

    // Subscribe to price updates for specific markets
    ws.send(JSON.stringify({
      type: 'subscribe',
      channel: 'prices',
      tickers: ['MARKET_TICKER_1', 'MARKET_TICKER_2']
    }));
  });

  ws.on('message', (data) => {
    const message = JSON.parse(data.toString());
    console.log('Update:', message);
  });

  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
  });

  ws.on('close', () => {
    console.log('Disconnected, reconnecting...');
    setTimeout(() => connectDFlowWebSocket(apiKey), 1000);
  });

  return ws;
}
```

## Best Practices

* Implement reconnection with exponential backoff.
* Subscribe only to needed markets using specific tickers when possible.
* Process messages asynchronously to avoid blocking during high-volume periods.
* Always implement `onerror` and `onclose` handlers.
* Use `"all": true` only when you genuinely need every market — it generates high message volume.

## DFlow WebSockets vs Helius Streaming

| Feature | DFlow WebSockets | Helius Enhanced WebSockets | Helius LaserStream |
|---|---|---|---|
| Data type | Market prices, trades, orderbooks | On-chain tx/account changes | On-chain tx/account changes |
| Latency | Market-level (fast) | Low (1.5-2x faster than standard WS) | Lowest (shred-level) |
| Use case | Price feeds, trading UIs | Tx confirmations, account monitoring | HFT, bots, indexers |
| Protocol | WebSocket | WebSocket | gRPC |
| Auth | DFlow API key | Helius API key | Helius API key |

**Use DFlow WebSockets when**: you need market-level data (prices, orderbooks, trade feeds) for prediction market UIs.

**Use Helius WebSockets when**: you need to monitor on-chain events (transaction confirmations, account changes) in real time.

**Use both together when**: building a full trading interface — DFlow WS for market data, Helius WS for transaction confirmations.

## Common Mistakes

- Not implementing reconnection logic (WebSocket connections drop)
- Subscribing to all markets when only a few are needed (unnecessary bandwidth)
- Blocking the event loop with synchronous processing of high-volume messages
- Not handling the `x-api-key` header requirement (connection will be rejected)
- Confusing DFlow WebSockets (market data) with Helius WebSockets (on-chain data)

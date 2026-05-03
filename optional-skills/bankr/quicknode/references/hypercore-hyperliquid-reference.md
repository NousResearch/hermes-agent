# HyperCore & Hyperliquid Reference

Quicknode provides infrastructure for the Hyperliquid L1 chain through HyperCore, delivering gRPC, JSON-RPC, WebSocket, and Info API access to exchange and trading data, plus HyperEVM RPC for smart contract execution.

## Overview

| Property | Value |
|----------|-------|
| **Chain** | Hyperliquid L1 |
| **Consensus** | HyperBFT (based on HotStuff) |
| **Native Token** | HYPE |
| **Mainnet Chain ID** | 999 |
| **Testnet Chain ID** | 998 |
| **Block Rate** | ~12 blocks/sec |
| **Status** | Public Beta |
| **gRPC Compression** | zstd (~70% bandwidth reduction) |
| **Architecture** | HyperCore (exchange/trading) + HyperEVM (smart contracts) |

## Network Configuration

| Network | Endpoint Pattern | Chain ID |
|---------|-----------------|----------|
| **Mainnet** | `https://[name].hype-mainnet.quiknode.pro/[token]/` | 999 (0x3E7) |
| **Testnet** | `https://[name].hype-testnet.quiknode.pro/[token]/` | 998 |

Testnet is pruned to the last 250 blocks.

## HyperCore Access Methods

| Method | Path / Port | Protocol | Description |
|--------|-------------|----------|-------------|
| **Info** | `/info` | HTTP POST | 50+ specialized methods for market data, positions, orders |
| **JSON-RPC** | `/hypercore` | HTTP POST | Block queries: `hl_getLatestBlocks`, `hl_getBlock`, `hl_getBatchBlocks` |
| **WebSocket** | `/hypercore/ws` | WebSocket | Real-time subscriptions: `hl_subscribe`, `hl_unsubscribe` |
| **gRPC** | Port 10000 | gRPC (HTTP/2) | Lowest latency streaming: `Ping`, `StreamBlocks`, `StreamData` |

## Authentication

### URL Token (default)

```
https://your-endpoint.hype-mainnet.quiknode.pro/your-auth-token/
```

### Header-Based

```bash
curl -H "x-token: your-auth-token" \
  https://your-endpoint.hype-mainnet.quiknode.pro/evm
```

### gRPC Authentication

```javascript
const grpc = require("@grpc/grpc-js");

const metadata = new grpc.Metadata();
metadata.add("x-token", "your-auth-token");
// Pass metadata to all gRPC calls
```

Endpoint for gRPC: `your-endpoint.hype-mainnet.quiknode.pro:10000` (TLS required).

## Info Endpoint

The Info API provides 50+ methods for querying Hyperliquid exchange data. All requests are `POST` to `/info` with a `type` field.

> **Note:** Some Info methods (e.g., `allMids`, `l2Book`, `meta`) are also available via Hyperliquid's public endpoints without a Quicknode subscription. Check https://www.quicknode.com/docs/hyperliquid/llms.txt for details on which methods require a Quicknode endpoint vs. public access.

### Base URL

```
https://[endpoint].hype-mainnet.quiknode.pro/[token]/info
```

### Quick Example

```typescript
const response = await fetch(
  `${process.env.QUICKNODE_RPC_URL}info`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ type: "allMids" }),
  }
);
const midPrices = await response.json();
// { "BTC": "92385.0", "ETH": "3167.4", ... }
```

### Key Methods

| Method (`type`) | Parameters | Description |
|-----------------|------------|-------------|
| `allMids` | — | Real-time mid-market prices for all pairs |
| `l2Book` | `coin` | Level 2 order book (up to 20 levels per side) |
| `recentTrades` | `coin` | Recent executed trades |
| `candleSnapshot` | `coin`, `interval`, `startTime`, `endTime` | OHLCV candlestick data |
| `meta` | — | Exchange metadata: trading pairs, leverage limits |
| `metaAndAssetCtxs` | — | Market data with funding, OI, oracle prices |
| `spotMeta` | — | Spot market metadata |
| `spotMetaAndAssetCtxs` | — | Spot metadata with prices |
| `clearinghouseState` | `user` | Account positions, margin, P&L |
| `spotClearinghouseState` | `user` | Spot token balances |
| `openOrders` | `user` | All open orders for a user |
| `frontendOpenOrders` | `user` | Open orders (frontend format) |
| `historicalOrders` | `user` | Up to 2,000 recent historical orders |
| `orderStatus` | `user`, `oid` | Status of a specific order |
| `userFills` | `user` | Up to 2,000 recent trade executions |
| `userFillsByTime` | `user`, `startTime` | Fills within a time range |
| `fundingHistory` | `coin`, `startTime` | Historical funding rates |
| `predictedFundings` | — | Forecasted funding rates |
| `activeAssetData` | `user`, `coin` | Active trading data for user/asset |
| `portfolio` | `user` | Account value and P&L history |
| `vaultDetails` | `vaultAddress` | Vault analytics |
| `exchangeStatus` | — | Exchange status and maintenance info |

## JSON-RPC Methods

POST requests to `/hypercore`.

### hl_getLatestBlocks

```typescript
const response = await fetch(
  `${process.env.QUICKNODE_RPC_URL}hypercore`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "hl_getLatestBlocks",
      params: { stream: "trades", count: 10 },
      id: 1,
    }),
  }
);
const { result } = await response.json();
// result.blocks: [{ local_time, block_time, block_number, events }]
```

### hl_getBlock

```typescript
const response = await fetch(
  `${process.env.QUICKNODE_RPC_URL}hypercore`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "hl_getBlock",
      params: ["trades", 817824084],
      id: 1,
    }),
  }
);
```

### hl_getBatchBlocks

```typescript
const response = await fetch(
  `${process.env.QUICKNODE_RPC_URL}hypercore`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "hl_getBatchBlocks",
      params: { stream: "trades", from: 817824078, to: 817824090 },
      id: 1,
    }),
  }
);
```

### hl_getLatestBlockNumber

```typescript
const response = await fetch(
  `${process.env.QUICKNODE_RPC_URL}hypercore`,
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      jsonrpc: "2.0",
      method: "hl_getLatestBlockNumber",
      params: ["events"],
      id: 1,
    }),
  }
);
```

## WebSocket

Connect to `/hypercore/ws` for real-time subscriptions.

### Subscribe

```typescript
import WebSocket from "ws";

const ws = new WebSocket(
  `${process.env.QUICKNODE_WSS_URL}hypercore/ws`
);

ws.on("open", () => {
  // Subscribe to trades
  ws.send(
    JSON.stringify({
      jsonrpc: "2.0",
      method: "hl_subscribe",
      params: { streamType: "trades" },
      id: 1,
    })
  );
});

ws.on("message", (data) => {
  const message = JSON.parse(data.toString());
  if (message.params) {
    console.log("Trade event:", message.params);
  }
});

// Unsubscribe
ws.send(
  JSON.stringify({
    jsonrpc: "2.0",
    method: "hl_unsubscribe",
    params: { streamType: "trades" },
    id: 2,
  })
);
```

## gRPC Streaming

Port 10000 provides the lowest-latency access to HyperCore data via three RPC methods.

### Connection Setup

```typescript
const grpc = require("@grpc/grpc-js");
const protoLoader = require("@grpc/proto-loader");

const ENDPOINT = "your-endpoint.hype-mainnet.quiknode.pro:10000";
const TOKEN = "your-auth-token";

const packageDefinition = protoLoader.loadSync("streaming.proto", {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

const proto = grpc.loadPackageDefinition(packageDefinition);
const channelCredentials = grpc.credentials.createSsl();
const client = new proto.streaming.Streaming(ENDPOINT, channelCredentials, {
  "grpc.max_receive_message_length": 100 * 1024 * 1024, // 100MB
});

const metadata = new grpc.Metadata();
metadata.add("x-token", TOKEN);
```

### gRPC Methods

| Method | Type | Description |
|--------|------|-------------|
| `Ping` | Unary | Connection health check |
| `StreamBlocks` | Server streaming | Stream blocks from a timestamp |
| `StreamData` | Bidirectional streaming | Subscribe to filtered data streams |

### Ping

```typescript
client.Ping({ count: 1 }, metadata, (error, response) => {
  if (error) console.error("Ping failed:", error);
  else console.log("Ping response:", response);
});
```

### StreamData (Bidirectional)

```typescript
const stream = client.StreamData(metadata);

// Subscribe to trades for specific coins
stream.write({
  subscribe: {
    stream_type: "TRADES",
    coins: ["BTC", "ETH"],
  },
});

// Send keepalive pings every 30 seconds
const pingInterval = setInterval(() => {
  stream.write({ ping: { timestamp: Date.now() } });
}, 30000);

stream.on("data", (response) => {
  if (response.data) {
    const block = JSON.parse(response.data.data);
    console.log("Block:", response.data.block_number, block);
  }
  if (response.pong) {
    console.log("Pong:", response.pong.timestamp);
  }
});

stream.on("error", (error) => {
  console.error("Stream error:", error);
  clearInterval(pingInterval);
});

stream.on("end", () => {
  clearInterval(pingInterval);
});
```

## gRPC Stream Types

| Stream Type | Volume | Available Via |
|-------------|--------|---------------|
| **TRADES** | High | gRPC, JSON-RPC, WebSocket |
| **ORDERS** | Very High | gRPC, JSON-RPC, WebSocket |
| **BOOK_UPDATES** | Very High | gRPC, JSON-RPC, WebSocket |
| **TWAP** | Low | gRPC, JSON-RPC, WebSocket |
| **EVENTS** | High | gRPC, JSON-RPC, WebSocket |
| **BLOCKS** | Extreme | gRPC only |
| **WRITER_ACTIONS** | Low | gRPC, JSON-RPC, WebSocket |

### Stream Data Details

- **TRADES**: Execution data — coin, price, size, side, fees, liquidation info
- **ORDERS**: Order lifecycle — 18+ status types (open, filled, canceled, rejected variants)
- **BOOK_UPDATES**: Level-2 order book diffs — individual order adds/removes
- **TWAP**: Time-weighted average price order updates — activated, finished, terminated
- **EVENTS**: Ledger updates, funding payments, deposits, withdrawals, delegations
- **BLOCKS**: Raw HyperCore blocks with all 34 action types (gRPC only)
- **WRITER_ACTIONS**: System-level spot token transfers (HyperCore to HyperEVM)

## gRPC Filtering

Filter streams by coin, user, side, and other fields depending on stream type.

### Filter Fields by Stream

| Stream | Available Filters |
|--------|-------------------|
| **TRADES** | `coin`, `user`, `side`, `liquidation`, `builder` |
| **ORDERS** | `coin`, `user`, `status`, `builder` |
| **BOOK_UPDATES** | `coin`, `side` |
| **TWAP** | `coin`, `user`, `status` |
| **EVENTS** | `user`, `type` |
| **WRITER_ACTIONS** | `user`, `action.type`, `action.token` |

### Filter Logic

- **AND across fields** — When multiple filter fields are specified (e.g., `coin` and `side`), all conditions must match (AND logic).
- **OR within values** — When a field has multiple values (e.g., `coin: { values: ["BTC", "ETH"] }`), any value can match (OR logic).
- **Special value `"*"`** — Matches any event where the field exists (non-null).
- **Special value `"null"`** — Matches events where the field is explicitly null.
- **Recursive matching** — Filters match recursively into nested JSON structures, so top-level field filters also apply to nested objects.

### Filter Limits

| Limit | Maximum |
|-------|---------|
| Values per `user` / `address` filter | 100 |
| Values per `coin` filter | 50 |
| Values per `type` / `status` filter | 20 |
| Total filter values across all fields | 500 |
| Named filters per stream | 10 |

### Filtering Examples

```typescript
const stream = client.StreamData(metadata);

// Subscribe to BTC and ETH buy trades only
stream.write({
  subscribe: {
    stream_type: "TRADES",
    filters: {
      coin: { values: ["BTC", "ETH"] },
      side: { values: ["B"] },
    },
  },
});

// Subscribe to order status changes for a specific user
stream.write({
  subscribe: {
    stream_type: "ORDERS",
    filters: {
      user: { values: ["0x2ba553d9f990a3b66b03b2dc0d030dfc1c061036"] },
      status: { values: ["filled", "canceled"] },
    },
  },
});

// Subscribe to all events where the user field exists
stream.write({
  subscribe: {
    stream_type: "EVENTS",
    filters: {
      user: { values: ["*"] },
    },
  },
});

// Subscribe to liquidation trades only
stream.write({
  subscribe: {
    stream_type: "TRADES",
    filters: {
      liquidation: { values: ["*"] },
    },
  },
});
```

## HyperEVM

HyperEVM provides EVM-compatible smart contract execution on Hyperliquid. Two RPC paths are available:

| Path | Protocol | Archive | Debug/Trace | Use Case |
|------|----------|---------|-------------|----------|
| `/evm` | HTTP | Partial | No | Standard blockchain operations |
| `/nanoreth` | HTTP + WebSocket | Extended | Yes (`debug_*`, `trace_*`) | Advanced debugging, tracing, subscriptions |

### Standard EVM Example (`/evm`)

```typescript
import { JsonRpcProvider } from "ethers";

const provider = new JsonRpcProvider(
  `${process.env.QUICKNODE_RPC_URL}evm`
);

const blockNumber = await provider.getBlockNumber();
const balance = await provider.getBalance("0x...");
```

### Debug/Trace Example (`/nanoreth`)

```typescript
import { JsonRpcProvider } from "ethers";

const provider = new JsonRpcProvider(
  `${process.env.QUICKNODE_RPC_URL}nanoreth`
);

// Standard methods work on nanoreth too
const blockNumber = await provider.getBlockNumber();

// Debug and trace methods only available on /nanoreth
const trace = await provider.send("debug_traceTransaction", [
  "0xTransactionHash...",
  { tracer: "callTracer" },
]);
```

### WebSocket Subscriptions (`/nanoreth`)

```typescript
import { WebSocketProvider } from "ethers";

const wsProvider = new WebSocketProvider(
  `${process.env.QUICKNODE_WSS_URL}nanoreth`
);

wsProvider.on("block", (blockNumber) => {
  console.log("New block:", blockNumber);
});
```

### Hyperliquid-Specific EVM Methods

| Method | Description |
|--------|-------------|
| `eth_getSystemTxsByBlockNumber` | Internal system transactions (HyperCore-to-HyperEVM) |
| `eth_getSystemTxsByBlockHash` | System transactions by block hash |
| `eth_usingBigBlocks` | Check if address uses big blocks |
| `eth_bigBlockGasPrice` | Gas price for big blocks |

## Best Practices

1. **Use gRPC for lowest latency** — Port 10000 gRPC streaming provides sub-millisecond data delivery, ideal for trading applications.
2. **Enable zstd compression** — Reduces bandwidth by ~70%, critical for high-volume streams like ORDERS and BOOK_UPDATES.
3. **Use `/nanoreth` for debugging** — Extended archive and trace/debug methods are only available on `/nanoreth`, not `/evm`.
4. **Handle ~12 blocks/sec throughput** — Hyperliquid produces blocks rapidly. Ensure your consumer can process events at this rate.
5. **Send gRPC keepalive pings** — Send pings every 30 seconds to maintain the connection.
6. **Note public beta status** — HyperCore on Quicknode is in public beta. APIs and behavior may change.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| gRPC connection refused on port 10000 | Wrong endpoint or port | Use `endpoint.hype-mainnet.quiknode.pro:10000` with TLS |
| Auth failed on gRPC | Missing or wrong `x-token` metadata | Add `metadata.add('x-token', TOKEN)` to all gRPC calls |
| No data from `/info` | Wrong path or missing `type` field | POST to `/info` with `{"type": "methodName"}` |
| WebSocket disconnects | No ping/pong or server maintenance | Implement reconnection logic with backoff |
| `/evm` missing debug methods | Debug methods not available on `/evm` | Switch to `/nanoreth` for `debug_*` and `trace_*` methods |
| Testnet data missing | Testnet pruned to last 250 blocks | Use mainnet for historical data; testnet is for testing only |
| High bandwidth usage | Unfiltered high-volume streams | Apply coin/user/side filters and enable zstd compression |

## Documentation

- **Hyperliquid Overview**: https://www.quicknode.com/docs/hyperliquid
- **Hyperliquid Overview (llms.txt)**: : https://www.quicknode.com/docs/hyperliquid/llms.txt
- **Hyperliquid gRPC API**: https://www.quicknode.com/docs/hyperliquid/grpc-api
- **HyperCore Filtering**: https://www.quicknode.com/docs/hyperliquid/filtering
- **Hyperliquid llms.txt**: https://www.quicknode.com/docs/hyperliquid/llms.txt
- **HyperCore Info Methods**: https://www.quicknode.com/docs/hyperliquid (Info endpoint section)
- **HyperEVM**: https://www.quicknode.com/docs/hyperliquid (HyperEVM section)
- **Guides**: https://www.quicknode.com/guides/tags/hyperliquid

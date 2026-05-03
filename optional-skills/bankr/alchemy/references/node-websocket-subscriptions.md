---
id: references/node-websocket-subscriptions.md
name: 'WebSocket Subscriptions'
description: 'Use WebSockets for real-time blockchain events without polling. Best for pending transactions, new blocks, and logs.'
tags:
  - alchemy
  - node-apis
  - evm
  - rpc
related:
  - node-json-rpc.md
  - webhooks-details.md
updated: 2026-02-23
---
# WebSocket Subscriptions

Real-time blockchain events via WebSocket. No polling required.

**Base URL**: `wss://<network>.g.alchemy.com/v2/$ALCHEMY_API_KEY`

---

## `eth_subscribe`

Creates a subscription for real-time events.

### Subscription Types

| Type | Description |
|------|-------------|
| `newHeads` | New block headers as they are mined |
| `logs` | Event logs matching a filter |
| `newPendingTransactions` | Pending transaction hashes (high volume) |
| `alchemy_minedTransactions` | Mined transactions matching a filter (Alchemy-specific) |

---

### `newHeads`

Subscribe to new block headers.

#### Request

```json
{ "jsonrpc": "2.0", "id": 1, "method": "eth_subscribe", "params": ["newHeads"] }
```

#### Notification

```json
{
  "jsonrpc": "2.0",
  "method": "eth_subscription",
  "params": {
    "subscription": "0x1234...",
    "result": {
      "number": "0x1312D00",
      "hash": "0x...",
      "parentHash": "0x...",
      "timestamp": "0x...",
      "gasLimit": "0x...",
      "gasUsed": "0x...",
      "baseFeePerGas": "0x...",
      "miner": "0x..."
    }
  }
}
```

---

### `logs`

Subscribe to event logs matching a filter.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `address` | string or string[] | No | Contract address(es) to filter |
| `topics` | array | No | Up to 4 topic filters (`null` = any) |

#### Request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "eth_subscribe",
  "params": [
    "logs",
    {
      "address": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"]
    }
  ]
}
```

#### Notification

```json
{
  "jsonrpc": "2.0",
  "method": "eth_subscription",
  "params": {
    "subscription": "0x5678...",
    "result": {
      "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "topics": ["0xddf252ad...", "0x000...from...", "0x000...to..."],
      "data": "0x00000000000000000000000000000000000000000000000000000000000f4240",
      "blockNumber": "0x1312D01",
      "transactionHash": "0x...",
      "logIndex": "0x0",
      "removed": false
    }
  }
}
```

---

### `newPendingTransactions`

Subscribe to pending transaction hashes.

#### Request

```json
{ "jsonrpc": "2.0", "id": 1, "method": "eth_subscribe", "params": ["newPendingTransactions"] }
```

#### Notification

```json
{
  "jsonrpc": "2.0",
  "method": "eth_subscription",
  "params": {
    "subscription": "0x9abc...",
    "result": "0x3847245c01829b043431067fb2bfa95f7b5bdc7e..."
  }
}
```

---

## `eth_unsubscribe`

Cancels a subscription.

### Request

```json
{ "jsonrpc": "2.0", "id": 1, "method": "eth_unsubscribe", "params": ["0x1234...subscription_id..."] }
```

### Response

```json
{ "jsonrpc": "2.0", "id": 1, "result": true }
```

---

## Example (Node.js)

```ts
import WebSocket from "ws";

const ws = new WebSocket(
  `wss://eth-mainnet.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`
);

ws.on("open", () => {
  ws.send(JSON.stringify({
    jsonrpc: "2.0", id: 1, method: "eth_subscribe", params: ["newHeads"]
  }));
});

ws.on("message", (data) => {
  const msg = JSON.parse(data.toString());
  if (msg.method === "eth_subscription") {
    console.log("New block:", msg.params.result.number);
  }
});
```

---

## Notes

- Subscriptions are stateful. Handle reconnects and resubscribe after reconnect.
- You may receive duplicate events on reconnect. De-duplicate by block hash or log index.
- `newPendingTransactions` is very high volume. Use tight filters if available.
- If WebSockets are unavailable, fall back to HTTP polling with coarse intervals and backoff.

## Official Docs
- [Subscription API Overview](https://www.alchemy.com/docs/reference/subscription-api)
- [eth_subscribe](https://www.alchemy.com/docs/chains/ethereum/ethereum-api-endpoints/eth-subscribe)

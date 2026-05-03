---
id: references/recipes-subscribe-pending-txs.md
name: 'Recipe: Subscribe to Pending Transactions'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
related:
  - node-websocket-subscriptions.md
updated: 2026-02-05
---
# Recipe: Subscribe to Pending Transactions

## Goal
Stream pending transactions in real time and optionally enrich them with transaction details.

## Inputs
- `network`
- `ALCHEMY_API_KEY`

## Steps
1. Connect to WebSocket endpoint.
2. Subscribe to `newPendingTransactions`.
3. Optionally call `eth_getTransactionByHash` to enrich data.

## Example (TypeScript)
```ts
import WebSocket from "ws";

const ws = new WebSocket(
  `wss://eth-mainnet.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`
);

ws.on("open", () => {
  ws.send(
    JSON.stringify({
      jsonrpc: "2.0",
      id: 1,
      method: "eth_subscribe",
      params: ["newPendingTransactions"],
    })
  );
});

ws.on("message", async (data) => {
  const msg = JSON.parse(data.toString());
  const txHash = msg?.params?.result;
  if (!txHash) return;

  // Optional: fetch tx details
  const res = await fetch(
    `https://eth-mainnet.g.alchemy.com/v2/${process.env.ALCHEMY_API_KEY}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 2,
        method: "eth_getTransactionByHash",
        params: [txHash],
      }),
    }
  );
  const json = await res.json();
  console.log(json.result);
});
```

## Output
- Stream of pending transaction hashes and optional enriched tx data.

## Related Files
- `node-websocket-subscriptions.md`

## Official Docs
- [Subscription API Overview](https://www.alchemy.com/docs/reference/subscription-api)

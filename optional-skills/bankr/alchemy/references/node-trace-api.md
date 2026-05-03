---
id: references/node-trace-api.md
name: 'Trace API'
description: 'Trace APIs expose internal call data and state changes for transactions and blocks. Useful for analytics and auditing.'
tags:
  - alchemy
  - node-apis
  - evm
  - rpc
related:
  - node-debug-api.md
  - data-transfers-api.md
updated: 2026-02-23
---
# Trace API

Trace APIs expose internal call data and state changes for transactions and blocks. JSON-RPC POST requests.

**Base URL**: `https://<network>.g.alchemy.com/v2/$ALCHEMY_API_KEY`

**Supported chains**: Ethereum Mainnet (full support). Partial support on other networks — verify per chain.

---

## `trace_transaction`

Returns all traces for a mined transaction.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `transactionHash` | string | Yes | Transaction hash (32-byte hex) |

### Request

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "trace_transaction",
    "params": ["0x3847245c01829b043431067fb2bfa95f7b5bdc7e..."]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": [
    {
      "action": {
        "callType": "call",
        "from": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "to": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "gas": "0x...",
        "input": "0xa9059cbb...",
        "value": "0x0"
      },
      "result": {
        "gasUsed": "0x...",
        "output": "0x0000...0001"
      },
      "subtraces": 1,
      "traceAddress": [],
      "type": "call",
      "blockHash": "0x...",
      "blockNumber": 20000000,
      "transactionHash": "0x3847245c...",
      "transactionPosition": 5
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `action.callType` | string | `"call"`, `"delegatecall"`, `"staticcall"`, `"create"`, `"create2"` |
| `action.from` | string | Caller address |
| `action.to` | string | Target address |
| `action.gas` | string | Gas provided (hex) |
| `action.input` | string | Calldata (hex) |
| `action.value` | string | ETH value (hex) |
| `result.gasUsed` | string | Gas used (hex) |
| `result.output` | string | Return data (hex) |
| `subtraces` | integer | Number of sub-traces |
| `traceAddress` | integer[] | Position in call tree (empty for top-level) |
| `type` | string | Trace type: `"call"`, `"create"`, `"suicide"`, `"reward"` |

---

## `trace_block`

Returns all traces for all transactions in a block.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `blockNumber` | string | Yes | Block number (hex) or `"latest"` |

### Request

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "trace_block",
    "params": ["0x1312D00"]
  }'
```

### Response

Array of trace objects (same schema as `trace_transaction`), one per trace across all transactions.

---

## `trace_call`

Traces a call without submitting it (like `eth_call` with trace output).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `transaction` | object | Yes | — | Transaction object (`from`, `to`, `data`, `value`, `gas`) |
| `traceTypes` | string[] | Yes | — | `["trace"]`, `["vmTrace"]`, `["stateDiff"]`, or combinations |
| `blockTag` | string | No | `"latest"` | Block tag or hex number |

### Request

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "trace_call",
    "params": [
      {
        "from": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "to": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "data": "0x70a08231..."
      },
      ["trace"],
      "latest"
    ]
  }'
```

---

## `trace_filter`

Returns traces matching a filter across a block range.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filter.fromBlock` | string | No | Start block (hex) |
| `filter.toBlock` | string | No | End block (hex) |
| `filter.fromAddress[]` | string[] | No | Filter by sender addresses |
| `filter.toAddress[]` | string[] | No | Filter by recipient addresses |
| `filter.after` | integer | No | Skip this many traces |
| `filter.count` | integer | No | Max traces to return |

### Request

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "trace_filter",
    "params": [{
      "fromBlock": "0x1312D00",
      "toBlock": "0x1312D05",
      "toAddress": ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],
      "count": 10
    }]
  }'
```

### Response

Array of trace objects (same schema as `trace_transaction`).

---

## Notes

- Trace data can be very large. Use `count` and block range limits to control response size.
- `trace_filter` over wide block ranges is expensive. Use narrow ranges.
- Different trace endpoints return the same field schema but at different scopes (per-tx vs per-block).
- `vmTrace` and `stateDiff` trace types produce significantly larger responses.

## Official Docs
- [trace_transaction](https://www.alchemy.com/docs/chains/ethereum/ethereum-api-endpoints/trace-transaction)
- [trace_block](https://www.alchemy.com/docs/chains/ethereum/ethereum-api-endpoints/trace-block)
- [trace_call](https://www.alchemy.com/docs/chains/ethereum/ethereum-api-endpoints/trace-call)
- [trace_filter](https://www.alchemy.com/docs/chains/ethereum/ethereum-api-endpoints/trace-filter)

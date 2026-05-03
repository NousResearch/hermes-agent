---
id: references/node-debug-api.md
name: 'Debug API'
description: 'Debug methods provide execution-level traces for a transaction or call. Use these for simulation, gas profiling, and internal call inspection.'
tags:
  - alchemy
  - node-apis
  - evm
  - rpc
related:
  - node-trace-api.md
  - data-simulation-api.md
updated: 2026-02-23
---
# Debug API

Debug methods provide execution-level traces for transactions and calls. JSON-RPC POST requests.

**Base URL**: `https://<network>.g.alchemy.com/v2/$ALCHEMY_API_KEY`

**Supported chains**: Ethereum Mainnet, Sepolia, and select L2s (verify per network).

---

## `debug_traceTransaction`

Replays a mined transaction and returns its execution trace.

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `transactionHash` | string | Yes | — | Transaction hash (32-byte hex) |
| `options.tracer` | string | No | — | `"callTracer"` (call tree) or `"prestateTracer"` (pre-execution state) |
| `options.tracerConfig.onlyTopCall` | boolean | No | `false` | Only trace the top-level call (skip sub-calls) |
| `options.timeout` | string | No | — | Trace timeout (e.g., `"10s"`, `"30s"`) |

### Request

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "debug_traceTransaction",
    "params": [
      "0x3847245c01829b043431067fb2bfa95f7b5bdc7e...",
      { "tracer": "callTracer" }
    ]
  }'
```

### Response (callTracer)

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "type": "CALL",
    "from": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
    "to": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "value": "0x0",
    "gas": "0x...",
    "gasUsed": "0x...",
    "input": "0xa9059cbb...",
    "output": "0x0000...0001",
    "calls": [
      {
        "type": "DELEGATECALL",
        "from": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "to": "0xa2327a938febf5fec13bacfb16ae10ecbc4cc26d",
        "gas": "0x...",
        "gasUsed": "0x...",
        "input": "0xa9059cbb...",
        "output": "0x0000...0001"
      }
    ]
  }
}
```

### Response Fields (callTracer)

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Call type: `"CALL"`, `"DELEGATECALL"`, `"STATICCALL"`, `"CREATE"`, `"CREATE2"` |
| `from` | string | Caller address |
| `to` | string | Target address |
| `value` | string | ETH value (hex) |
| `gas` | string | Gas allocated (hex) |
| `gasUsed` | string | Gas consumed (hex) |
| `input` | string | Calldata (hex) |
| `output` | string | Return data (hex) |
| `error` | string | Revert reason (if call failed) |
| `calls` | array | Nested sub-calls (recursive same structure) |

---

## `debug_traceCall`

Traces a call without submitting it (like `eth_call` but with trace output).

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `transaction` | object | Yes | — | Transaction object (`from`, `to`, `data`, `value`, `gas`) |
| `blockTag` | string | No | `"latest"` | Block tag or hex number |
| `options` | object | No | — | Same tracer options as `debug_traceTransaction` |

### Request

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "debug_traceCall",
    "params": [
      {
        "from": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "to": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "data": "0x70a08231000000000000000000000000d8da6bf26964af9d7eed9e03e53415d37aa96045"
      },
      "latest",
      { "tracer": "callTracer" }
    ]
  }'
```

### Response

Same format as `debug_traceTransaction` (depends on tracer selected).

---

## `debug_traceBlockByNumber`

Traces all transactions in a block.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `blockNumber` | string | Yes | Block number (hex) or `"latest"` |
| `options` | object | No | Same tracer options |

### Request

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "debug_traceBlockByNumber",
    "params": ["0x1312D00", { "tracer": "callTracer" }]
  }'
```

### Response

Returns an array of trace results, one per transaction in the block.

---

## Notes

- Debug traces are expensive (high compute units). Cache results where possible.
- Trace output varies by tracer. `callTracer` gives a call tree; `prestateTracer` shows pre-execution account state.
- Some networks may not support all debug methods.
- Use `onlyTopCall: true` to reduce output size when you only need the top-level call.

## Official Docs
- [debug_traceTransaction](https://www.alchemy.com/docs/chains/ethereum/ethereum-api-endpoints/debug-tracetransaction)
- [debug_traceCall](https://www.alchemy.com/docs/chains/ethereum/ethereum-api-endpoints/debug-tracecall)

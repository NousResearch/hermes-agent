---
id: references/node-utility-api.md
name: 'Utility API'
description: 'Convenience RPC methods that reduce round trips for common tasks like bulk transaction receipt retrieval.'
tags:
  - alchemy
  - node-apis
  - evm
  - rpc
related:
  - node-json-rpc.md
  - operational-rate-limits-and-compute-units.md
updated: 2026-02-23
---
# Utility API

Alchemy-specific convenience methods that reduce round trips. JSON-RPC POST requests.

**Base URL**: `https://<network>.g.alchemy.com/v2/$ALCHEMY_API_KEY`

---

## `alchemy_getTransactionReceipts`

Returns all transaction receipts for a given block in a single call. Replaces looping over `eth_getTransactionReceipt`.

### Parameters

Accepts a single object with one of:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `blockNumber` | string | Conditional | Block number (hex). Use this OR `blockHash`. |
| `blockHash` | string | Conditional | Block hash (32-byte hex). Use this OR `blockNumber`. |

### Request

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_getTransactionReceipts",
    "params": [{ "blockNumber": "0x1312D00" }]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "receipts": [
      {
        "transactionHash": "0x3847245c...",
        "transactionIndex": "0x0",
        "blockHash": "0x...",
        "blockNumber": "0x1312D00",
        "from": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "to": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "cumulativeGasUsed": "0x...",
        "gasUsed": "0xb4c8",
        "effectiveGasPrice": "0x...",
        "contractAddress": null,
        "logs": [],
        "logsBloom": "0x...",
        "status": "0x1",
        "type": "0x2"
      }
    ]
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `receipts` | array | Array of transaction receipt objects |

Each receipt has the same fields as `eth_getTransactionReceipt`:

| Field | Type | Description |
|-------|------|-------------|
| `transactionHash` | string | Transaction hash |
| `transactionIndex` | string | Index in block (hex) |
| `blockHash` | string | Block hash |
| `blockNumber` | string | Block number (hex) |
| `from` | string | Sender address |
| `to` | string | Recipient address (null for contract creation) |
| `cumulativeGasUsed` | string | Cumulative gas used (hex) |
| `gasUsed` | string | Gas used by this tx (hex) |
| `effectiveGasPrice` | string | Actual gas price (hex) |
| `contractAddress` | string | Created contract address (null if not deployment) |
| `logs` | array | Event logs |
| `status` | string | `"0x1"` (success) or `"0x0"` (failure) |
| `type` | string | Tx type (hex) |

---

## Notes

- Response can be very large for blocks with many transactions. Watch response sizes.
- Some networks may not support this method.

## Official Docs
- [alchemy_getTransactionReceipts](https://www.alchemy.com/docs/chains/ethereum/ethereum-api-endpoints/alchemy-gettransactionreceipts)

---
id: agentic-gateway/references/data-simulation-api.md
name: 'Simulation API'
description: 'Simulate transactions before submitting them on-chain via the Agentic Gateway. Use this for safety checks and user previews.'
tags:
  - agentic-gateway
  - data-apis
  - data
updated: 2026-02-23
---
# Simulation API

Simulate transactions before submitting them on-chain. Preview asset changes and execution traces. JSON-RPC POST requests.

**Base URL**: `https://x402.alchemy.com/{chainNetwork}/v2`

**Supported chains**: Ethereum, Base, Polygon, Arbitrum, Optimism (and testnets).

---

## `alchemy_simulateAssetChanges`

Simulates a transaction and returns the predicted asset changes (tokens gained/lost).

### Parameters

Single parameter: a `transaction` object.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `transaction.from` | string | No | Sender address (hex) |
| `transaction.to` | string | Yes | Recipient/contract address (hex) |
| `transaction.value` | string | No | ETH value to send (hex, in wei) |
| `transaction.data` | string | No | Calldata (hex) |
| `transaction.gas` | string | No | Gas limit (hex) |
| `transaction.gasPrice` | string | No | Gas price (hex, legacy) |
| `transaction.maxFeePerGas` | string | No | Max fee per gas (hex, EIP-1559) |
| `transaction.maxPriorityFeePerGas` | string | No | Max priority fee (hex, EIP-1559) |

### Request

```bash
curl -s -X POST https://x402.alchemy.com/eth-mainnet/v2 \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_simulateAssetChanges",
    "params": [{
      "from": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
      "to": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
      "value": "0x0",
      "data": "0xa9059cbb000000000000000000000000ef4396d9ff8107086d215a1c9f8866c54795d7c700000000000000000000000000000000000000000000000000000000000f4240"
    }]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "changes": [
      {
        "assetType": "ERC20",
        "changeType": "TRANSFER",
        "from": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "to": "0xef4396d9ff8107086d215a1c9f8866c54795d7c7",
        "rawAmount": "1000000",
        "contractAddress": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "tokenId": null,
        "decimals": 6,
        "symbol": "USDC",
        "name": "USD Coin",
        "logo": "https://static.alchemyapi.io/images/assets/3408.png",
        "amount": "1.0"
      }
    ],
    "gasUsed": "0xb4c8",
    "error": null
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `changes` | array | List of asset change objects |
| `changes[].assetType` | string | `"NATIVE"`, `"ERC20"`, `"ERC721"`, `"ERC1155"`, `"SPECIAL_NFT"` |
| `changes[].changeType` | string | `"TRANSFER"`, `"APPROVE"` |
| `changes[].from` | string | Sender address |
| `changes[].to` | string | Recipient address |
| `changes[].rawAmount` | string | Raw token amount (smallest unit) |
| `changes[].amount` | string | Human-readable amount (decimal-adjusted) |
| `changes[].contractAddress` | string | Token contract address (null for native) |
| `changes[].tokenId` | string | Token ID (for NFTs) |
| `changes[].decimals` | integer | Token decimals |
| `changes[].symbol` | string | Token symbol |
| `changes[].name` | string | Token name |
| `changes[].logo` | string | Token logo URL |
| `gasUsed` | string | Gas used (hex) |
| `error` | object | Error details if simulation reverted (null on success) |

---

## `alchemy_simulateExecution`

Simulates a transaction and returns detailed execution traces (call traces, logs, state overrides).

### Parameters

Ordered parameters: `transaction`, `blockTag`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `transaction` | object | Yes | — | Same transaction object as `simulateAssetChanges` |
| `blockTag` | string | No | `"latest"` | `"latest"`, `"safe"`, `"finalized"`, `"earliest"`, or hex block number |

### Request

```bash
curl -s -X POST https://x402.alchemy.com/eth-mainnet/v2 \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_simulateExecution",
    "params": [
      {
        "from": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "to": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "data": "0xa9059cbb000000000000000000000000ef4396d9ff8107086d215a1c9f8866c54795d7c700000000000000000000000000000000000000000000000000000000000f4240"
      },
      "latest"
    ]
  }'
```

### Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "calls": [
      {
        "type": "CALL",
        "from": "0xd8da6bf26964af9d7eed9e03e53415d37aa96045",
        "to": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "value": "0x0",
        "gas": "0x...",
        "gasUsed": "0x...",
        "input": "0xa9059cbb...",
        "output": "0x0000...0001",
        "decoded": {
          "methodName": "transfer",
          "inputs": [
            { "name": "to", "value": "0xef4396d9ff8107086d215a1c9f8866c54795d7c7", "type": "address" },
            { "name": "value", "value": "1000000", "type": "uint256" }
          ],
          "outputs": [
            { "name": "", "value": "true", "type": "bool" }
          ]
        }
      }
    ],
    "logs": [
      {
        "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "topics": ["0xddf252ad...", "0x000...d8da...", "0x000...ef43..."],
        "data": "0x00000000000000000000000000000000000000000000000000000000000f4240",
        "decoded": {
          "eventName": "Transfer",
          "inputs": [
            { "name": "from", "value": "0xd8da...", "type": "address", "indexed": true },
            { "name": "to", "value": "0xef43...", "type": "address", "indexed": true },
            { "name": "value", "value": "1000000", "type": "uint256", "indexed": false }
          ]
        }
      }
    ]
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `calls` | array | Execution trace entries |
| `calls[].type` | string | Call type: `"CALL"`, `"DELEGATECALL"`, `"STATICCALL"`, `"CREATE"`, `"CREATE2"` |
| `calls[].from` | string | Caller address |
| `calls[].to` | string | Target address |
| `calls[].value` | string | ETH value (hex) |
| `calls[].gas` | string | Gas allocated (hex) |
| `calls[].gasUsed` | string | Gas consumed (hex) |
| `calls[].input` | string | Input calldata (hex) |
| `calls[].output` | string | Return data (hex) |
| `calls[].decoded` | object | ABI-decoded method name, inputs, outputs, and authority (e.g., `"ETHERSCAN"`) |
| `calls[].calls` | array | Nested sub-calls (only in `NESTED` format) |
| `logs` | array | Event logs emitted |
| `logs[].address` | string | Emitting contract |
| `logs[].topics` | string[] | Log topics |
| `logs[].data` | string | Log data (hex) |
| `logs[].decoded` | object | ABI-decoded event name and inputs |

---

## `alchemy_simulateAssetChangesBundle`

Simulates a bundle of transactions sequentially and returns asset changes for each.

### Parameters

Single parameter: `transactions` array (same transaction object format). **Max 3 transactions per bundle.**

### Request

```bash
curl -s -X POST https://x402.alchemy.com/eth-mainnet/v2 \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_simulateAssetChangesBundle",
    "params": [[
      {
        "from": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "to": "0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
        "data": "0x095ea7b3..."
      },
      {
        "from": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
        "to": "0xdef1c0ded9bec7f1a1670819833240f027b25eff",
        "data": "0xd9627aa4..."
      }
    ]]
  }'
```

### Response

Returns an array of results, one per transaction. Each result has the same `changes`, `gasUsed`, and `error` fields as `simulateAssetChanges`.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": [
    { "changes": [...], "gasUsed": "0x...", "error": null },
    { "changes": [...], "gasUsed": "0x...", "error": null }
  ]
}
```

---

## `alchemy_simulateExecutionBundle`

Simulates a bundle of transactions and returns execution traces for each.

### Parameters

Ordered parameters: `transactions`, `blockTag`. Same as `simulateExecution` but with a transactions array. **Max 3 transactions per bundle.**

### Request

```bash
curl -s -X POST https://x402.alchemy.com/eth-mainnet/v2 \
  -H "Content-Type: application/json" \
  -H "Authorization: SIWE $TOKEN" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_simulateExecutionBundle",
    "params": [
      [
        { "from": "0xd8dA...", "to": "0xA0b8...", "data": "0x095ea7b3..." },
        { "from": "0xd8dA...", "to": "0xdef1...", "data": "0xd9627aa4..." }
      ],
      "latest"
    ]
  }'
```

### Response

Returns an array of results, one per transaction. Each result has `calls` and `logs` fields as in `simulateExecution`.

---

## Notes

- Simulation does not guarantee real execution outcome if on-chain state changes between simulation and submission.
- Bundle transactions execute sequentially — earlier transactions affect the state for later ones.
- Simulation is more compute-intensive than standard reads. Cache results where possible.
- Reverted simulations return an `error` object with the revert reason.

## Official Docs
- [Simulation API Overview](https://www.alchemy.com/docs/data/simulation-api)
- [alchemy_simulateAssetChanges](https://www.alchemy.com/docs/data/simulation-api/simulation-api-endpoints/alchemy-simulate-asset-changes)
- [alchemy_simulateExecution](https://www.alchemy.com/docs/data/simulation-api/simulation-api-endpoints/alchemy-simulate-execution)
- [alchemy_simulateAssetChangesBundle](https://www.alchemy.com/docs/data/simulation-api/simulation-api-endpoints/alchemy-simulate-asset-changes-bundle)
- [alchemy_simulateExecutionBundle](https://www.alchemy.com/docs/data/simulation-api/simulation-api-endpoints/alchemy-simulate-execution-bundle)

---
id: references/node-enhanced-apis.md
name: 'Enhanced APIs (Alchemy RPC Extensions)'
description: 'Alchemy provides enhanced JSON-RPC methods (prefixed with `alchemy_`) that offer indexed, higher-level data without manual log scanning.'
tags:
  - alchemy
  - node-apis
  - evm
  - rpc
related:
  - data-token-api.md
  - data-transfers-api.md
  - data-simulation-api.md
updated: 2026-02-23
---
# Enhanced APIs (Alchemy RPC Extensions)

Alchemy-specific JSON-RPC methods (`alchemy_*` prefix) that provide indexed, higher-level data. These are documented in detail in their dedicated reference files.

**Base URL**: `https://<network>.g.alchemy.com/v2/$ALCHEMY_API_KEY`

---

## Method Index

For detailed parameters, request/response examples, and response schemas, see the dedicated reference files:

| Method | Description | Reference |
|--------|-------------|-----------|
| `alchemy_getTokenBalances` | ERC-20 token balances for an address | [data-token-api.md](data-token-api.md) |
| `alchemy_getTokenMetadata` | Token name, symbol, decimals, logo | [data-token-api.md](data-token-api.md) |
| `alchemy_getTokenAllowance` | Spender allowance for a token | [data-token-api.md](data-token-api.md) |
| `alchemy_getAssetTransfers` | Historical transfer history | [data-transfers-api.md](data-transfers-api.md) |
| `alchemy_simulateAssetChanges` | Simulate transaction asset changes | [data-simulation-api.md](data-simulation-api.md) |
| `alchemy_simulateExecution` | Simulate with execution traces | [data-simulation-api.md](data-simulation-api.md) |
| `alchemy_simulateAssetChangesBundle` | Simulate bundle asset changes | [data-simulation-api.md](data-simulation-api.md) |
| `alchemy_simulateExecutionBundle` | Simulate bundle with traces | [data-simulation-api.md](data-simulation-api.md) |
| `alchemy_getTransactionReceipts` | Bulk receipts for a block | [node-utility-api.md](node-utility-api.md) |

---

## Quick Example

```bash
curl -s -X POST https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "alchemy_getTokenBalances",
    "params": ["0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045", "erc20"]
  }'
```

---

## Notes

- All enhanced methods use the same JSON-RPC endpoint as standard node API.
- Enhanced APIs are efficient but compute-unit metered. Prefer filters over large ranges.
- Availability varies by network. Check per-chain support.

## Agentic Gateway

These enhanced APIs are also available via `https://x402.alchemy.com/{chainNetwork}/v2` without an API key.
See the `agentic-gateway` skill for SIWE authentication and x402 payment setup.

## Official Docs
- [Enhanced APIs Overview](https://www.alchemy.com/docs/reference/enhanced-apis-overview)

---
id: references/node-overview.md
name: node-apis
description: Core JSON-RPC and WebSocket APIs for EVM chains via Alchemy node endpoints, plus Debug/Trace and utility methods. Use when building EVM integrations that need standard RPC calls, real-time subscriptions, enhanced Alchemy methods, or execution-level tracing.
tags: []
related: []
updated: 2026-02-14
metadata:
  author: alchemyplatform
  version: "1.0"
---
# Node APIs (EVM)

## Summary
Core JSON-RPC and WebSocket APIs for EVM chains via Alchemy node endpoints, plus Debug/Trace and utility methods.

## References (Recommended Order)
1. [node-json-rpc.md](node-json-rpc.md) - Standard JSON-RPC methods and endpoint patterns.
2. [node-websocket-subscriptions.md](node-websocket-subscriptions.md) - Real-time subscriptions (pending txs, logs, new heads).
3. [node-enhanced-apis.md](node-enhanced-apis.md) - Alchemy-enhanced RPC methods that reduce RPC call count.
4. [node-utility-api.md](node-utility-api.md) - Convenience endpoints like bulk transaction receipts.
5. [node-debug-api.md](node-debug-api.md) - Debug tracing for transaction simulation and execution insight.
6. [node-trace-api.md](node-trace-api.md) - Trace-level details for internal calls and state diffs.

## How to Use This Skill
- Start with `node-json-rpc.md` for base connectivity and request patterns.
- Use `node-enhanced-apis.md` for wallet/asset analytics on EVM without scanning logs.
- Use Debug/Trace when you need internal call trees or detailed execution flow.

## Agentic Gateway
Node JSON-RPC and enhanced APIs are also available via the Agentic Gateway (`https://x402.alchemy.com/{chainNetwork}/v2`) without an API key.
See the `agentic-gateway` skill for SIWE authentication and x402 payment setup.

## Cross-References
- `data-apis` skill for higher-level asset analytics.
- `webhooks` skill for event-driven flows.
- `operational` skill for auth, limits, and reliability.
- `agentic-gateway` skill for easy agent access to Alchemy's developer platform.

## Official Docs
- [Chain APIs Overview](https://www.alchemy.com/docs/reference/chain-apis-overview)
- [Enhanced APIs Overview](https://www.alchemy.com/docs/reference/enhanced-apis-overview)

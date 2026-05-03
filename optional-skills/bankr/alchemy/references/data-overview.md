---
id: references/data-overview.md
name: data-apis
description: Higher-level Alchemy APIs for asset discovery, wallet analytics, transfer history, NFT data, and token pricing. Use when you need indexed blockchain data without raw RPC log scanning, including token balances, NFT ownership, portfolio views, price feeds, and transaction simulation.
tags: []
related: []
updated: 2026-02-14
metadata:
  author: alchemyplatform
  version: "1.0"
---
# Data APIs

## Summary
Higher-level APIs for asset discovery, wallet analytics, transfer history, and pricing. These are optimized for analytics use cases and reduce the need for raw RPC log scanning.

## References (Recommended Order)
1. [data-token-api.md](data-token-api.md) - Token balances and token metadata for wallets and contracts.
2. [data-portfolio-apis.md](data-portfolio-apis.md) - Consolidated wallet views (tokens/NFTs/summary).
3. [data-transfers-api.md](data-transfers-api.md) - Transfer history and indexed movement data.
4. [data-nft-api.md](data-nft-api.md) - NFT ownership, metadata, and collection queries.
5. [data-prices-api.md](data-prices-api.md) - Token price data for current and historical pricing.
6. [data-simulation-api.md](data-simulation-api.md) - Pre-execution simulation for risk checks.

## How to Use This Skill
- Prefer these APIs when you want asset analytics or historical data without maintaining a custom indexer.
- If you need real-time updates, pair with the `webhooks` skill.

## Agentic Gateway
Most Data APIs are also available via the Agentic Gateway (`https://x402.alchemy.com/...`) without an API key.
See the `agentic-gateway` skill for SIWE authentication and x402 payment setup.

## Cross-References
- `node-apis` skill → `node-enhanced-apis.md` for related RPC-style endpoints.
- `recipes` skill for end-to-end workflows.
- `agentic-gateway` skill for easy agent access to Alchemy's developer platform.

## Official Docs
- [Data APIs Overview](https://www.alchemy.com/docs/reference/data-overview)

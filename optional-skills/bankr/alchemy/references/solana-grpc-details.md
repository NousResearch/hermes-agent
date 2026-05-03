---
id: references/solana-grpc-details.md
name: 'Yellowstone gRPC Overview'
description: 'Yellowstone gRPC provides high-throughput Solana data streams for blocks, transactions, accounts, and slots. Use this for real-time indexing at scale.'
tags:
  - alchemy
  - solana
related:
  - solana-grpc-subscribe-request.md
  - solana-grpc-best-practices.md
updated: 2026-02-05
---
# Yellowstone gRPC Overview

## Summary
Yellowstone gRPC provides high-throughput Solana data streams for blocks, transactions, accounts, and slots. Use this for real-time indexing at scale.

## Primary Use Cases
- Build custom Solana indexers.
- Stream account changes with low latency.
- High-frequency monitoring and analytics.

## When To Use / Not Use
- Use for streaming at scale or when polling is too expensive.
- Avoid if basic JSON-RPC polling is sufficient.

## Related Files
- `solana-grpc-subscribe-request.md`
- `solana-grpc-best-practices.md`

## Official Docs
- [Yellowstone gRPC Overview](https://www.alchemy.com/docs/reference/yellowstone-grpc-overview)

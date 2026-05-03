---
id: references/solana-grpc-subscribe-blocks.md
name: 'Subscribe Blocks'
description: 'Block streams provide full block data with transactions and metadata.'
tags:
  - alchemy
  - solana
related:
  - solana-grpc-best-practices.md
updated: 2026-02-05
---
# Subscribe Blocks

## Summary
Block streams provide full block data with transactions and metadata.

## Use Cases
- Full block ingestion for indexers.
- Cross-check block data against RPC.

## Guidance
- Ensure backpressure handling for large blocks.
- Guard against large message sizes and slow consumers.

## Related Files
- `solana-grpc-best-practices.md`

## Official Docs
- [Yellowstone gRPC API Overview](https://www.alchemy.com/docs/reference/yellowstone-grpc-api-overview)

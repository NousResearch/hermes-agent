---
id: references/solana-grpc-subscribe-transactions.md
name: 'Subscribe Transactions'
description: 'Transaction streams deliver raw or decoded transaction data in near real-time.'
tags:
  - alchemy
  - solana
related:
  - solana-grpc-subscribe-request.md
  - solana-grpc-best-practices.md
updated: 2026-02-05
---
# Subscribe Transactions

## Summary
Transaction streams deliver raw or decoded transaction data in near real-time.

## Use Cases
- Real-time analytics pipelines.
- On-chain alerting and monitoring.

## Guidance
- Filter by program IDs when possible.
- Expect high volume on mainnet.
- Implement backpressure and bounded queues; transactions can arrive in bursts.

## Related Files
- `solana-grpc-subscribe-request.md`
- `solana-grpc-best-practices.md`

## Official Docs
- [Yellowstone gRPC API Overview](https://www.alchemy.com/docs/reference/yellowstone-grpc-api-overview)

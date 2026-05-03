---
id: references/solana-grpc-subscribe-accounts.md
name: 'Subscribe Accounts'
description: 'Account streams deliver updates when account state changes.'
tags:
  - alchemy
  - solana
related:
  - solana-grpc-subscribe-request.md
updated: 2026-02-05
---
# Subscribe Accounts

## Summary
Account streams deliver updates when account state changes.

## Use Cases
- Track token accounts and balances.
- React to program state changes.

## Guidance
- Use account owner or data filters to reduce noise.
- Use backpressure and batching if account updates spike.

## Related Files
- `solana-grpc-subscribe-request.md`

## Official Docs
- [Yellowstone gRPC API Overview](https://www.alchemy.com/docs/reference/yellowstone-grpc-api-overview)

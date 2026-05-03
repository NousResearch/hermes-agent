---
id: references/solana-grpc-best-practices.md
name: 'Yellowstone Best Practices'
description: 'Practical guidance to keep Yellowstone consumers reliable and efficient.'
tags:
  - alchemy
  - solana
related:
  - solana-grpc-examples.md
updated: 2026-02-05
---
# Yellowstone Best Practices

## Summary
Practical guidance to keep Yellowstone consumers reliable and efficient.

## Recommendations
- Use server-side filters to reduce bandwidth.
- Implement reconnect logic and resume from last processed slot.
- Persist checkpoints to avoid reprocessing.
- Apply backpressure (queue + worker model).
- Use gRPC flow control and limit in-flight messages to avoid OOM.
- Persist checkpoints to resume after reconnects without reprocessing.

## Related Files
- `solana-grpc-examples.md`

## Official Docs
- [Yellowstone gRPC Overview](https://www.alchemy.com/docs/reference/yellowstone-grpc-overview)

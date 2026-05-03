---
id: references/solana-wallets.md
name: 'Solana Wallet Integration'
description: 'High-level guidance for integrating Solana wallets and signing transactions.'
tags:
  - alchemy
  - solana
related:
  - solana-rpc.md
  - ecosystem-solana-web3js.md
updated: 2026-02-05
---
# Solana Wallet Integration

## Summary
High-level guidance for integrating Solana wallets and signing transactions.

## Primary Use Cases
- Wallet connect flows.
- Program interactions and signing.

## Integration Notes
- Use `@solana/web3.js` for client and server operations.
- Use `Transaction` or `VersionedTransaction` based on the program requirements.

## Gotchas
- Always specify the recent blockhash.
- Handle signature confirmation with appropriate `commitment`.

## Related Files
- `solana-rpc.md`
- `ecosystem-solana-web3js.md`

## Official Docs
- [Solana API Quickstart](https://www.alchemy.com/docs/reference/solana-api-quickstart)

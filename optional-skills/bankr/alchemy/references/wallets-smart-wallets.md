---
id: references/wallets-smart-wallets.md
name: 'Smart Wallets'
description: 'Smart wallets (account abstraction) enable programmable accounts with features like session keys, batched transactions, and gas sponsorship.'
tags:
  - alchemy
  - wallets
related:
  - wallets-bundler.md
  - wallets-gas-manager.md
updated: 2026-02-05
---
# Smart Wallets

## Summary
Smart wallets (account abstraction) enable programmable accounts with features like session keys, batched transactions, and gas sponsorship.

## Primary Use Cases
- Gasless onboarding.
- Transaction batching and automation.
- Safer UX via spend limits or session keys.

## Integration Notes
- Smart wallets typically require a bundler and paymaster.
- Pair with `wallets-bundler.md` and `wallets-gas-manager.md`.

## Gotchas & Edge Cases
- Account deployment costs can vary by chain.
- Some dapps require EOA-only signatures; handle fallbacks.

## Related Files
- `wallets-bundler.md`
- `wallets-gas-manager.md`

## Official Docs
- [Smart Wallets](https://www.alchemy.com/docs/wallets)
- [Intro to Account Kit](https://www.alchemy.com/docs/wallets/concepts/intro-to-account-kit)

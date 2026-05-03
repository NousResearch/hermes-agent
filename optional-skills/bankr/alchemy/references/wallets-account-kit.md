---
id: references/wallets-account-kit.md
name: 'Account Kit'
description: 'Account Kit is Alchemy''s wallet SDK for onboarding users and managing wallet UX. Use it for embedded wallet flows or seamless authentication.'
tags:
  - alchemy
  - wallets
related:
  - wallets-smart-wallets.md
  - wallets-gas-manager.md
  - operational-auth-and-keys.md
updated: 2026-02-05
---
# Account Kit

## Summary
Account Kit is Alchemy's wallet SDK for onboarding users and managing wallet UX. Use it for embedded wallet flows or seamless authentication.

## Primary Use Cases
- Email/social login wallet creation.
- Embedded wallet UX in web apps.
- Signing and sending transactions with minimal setup.

## Integration Notes
- Typically used client-side with a wallet UI layer.
- Pair with `wallets-gas-manager.md` for sponsored transactions.

## Gotchas & Edge Cases
- Ensure you store session/auth tokens securely.
- Keep wallet creation flows resilient to network errors.

## Related Files
- `wallets-smart-wallets.md`
- `wallets-gas-manager.md`
- `operational-auth-and-keys.md`

## Official Docs
- [Account Kit Core Reference](https://www.alchemy.com/docs/wallets/reference/account-kit/core)
- [Intro to Account Kit](https://www.alchemy.com/docs/wallets/concepts/intro-to-account-kit)

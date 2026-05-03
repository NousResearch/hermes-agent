---
id: references/webhooks-webhook-types.md
name: 'Webhook Types'
description: 'Webhook types determine what events you receive and how they''re filtered.'
tags:
  - alchemy
  - webhooks
  - webhook
related:
  - webhooks-address-activity.md
  - webhooks-nft-activity.md
  - webhooks-custom-webhooks.md
updated: 2026-02-05
---
# Webhook Types

## Summary
Webhook types determine what events you receive and how they're filtered.

## Spec-Derived Types
- `ADDRESS_ACTIVITY`
- `NFT_ACTIVITY`
- `GRAPHQL`

## How To Choose
- Use Address Activity for wallets and transfer tracking.
- Use NFT Activity for marketplace and collection events.
- Use Custom Webhooks (GRAPHQL) when you need custom log filters or multi-contract triggers.

## Notes
- Exact type names and availability vary by chain; verify in official docs.

## Related Files
- `webhooks-address-activity.md`
- `webhooks-nft-activity.md`
- `webhooks-custom-webhooks.md`

## Official Docs
- [Notify API Quickstart](https://www.alchemy.com/docs/reference/notify-api-quickstart)

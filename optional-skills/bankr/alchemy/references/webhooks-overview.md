---
id: references/webhooks-overview.md
name: webhooks
description: Push-based delivery of blockchain events via Alchemy Notify API. Use when you need real-time notifications for address activity, NFT transfers, or custom on-chain events instead of polling. Covers webhook creation, payload formats, signature verification, and filtering.
tags: []
related: []
updated: 2026-02-14
metadata:
  author: alchemyplatform
  version: "1.0"
---
# Webhooks (Notify)

## Summary
Push-based delivery of blockchain events. Use this instead of polling where possible.

## References (Recommended Order)
1. [webhooks-details.md](webhooks-details.md) - Webhook architecture and how to choose types.
2. [webhooks-webhook-types.md](webhooks-webhook-types.md) - Event types and when to use each.
3. [webhooks-webhook-payloads.md](webhooks-webhook-payloads.md) - Payload structure and validation patterns.
4. [webhooks-verify-signatures.md](webhooks-verify-signatures.md) - Signature verification and security guidance.
5. [webhooks-address-activity.md](webhooks-address-activity.md) - Address activity webhooks.
6. [webhooks-nft-activity.md](webhooks-nft-activity.md) - NFT-related webhooks.
7. [webhooks-custom-webhooks.md](webhooks-custom-webhooks.md) - Custom webhook setup and filters.

## Cross-References
- `data-apis` skill → `data-transfers-api.md` for historical data when a webhook is missed.
- `operational` skill → `operational-auth-and-keys.md` for security best practices.

## Official Docs
- [Notify API Quickstart](https://www.alchemy.com/docs/reference/notify-api-quickstart)

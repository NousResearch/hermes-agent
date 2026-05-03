---
id: references/webhooks-address-activity.md
name: 'Address Activity Webhooks'
description: 'Address activity webhooks notify you when specified addresses send or receive assets.'
tags:
  - alchemy
  - webhooks
  - webhook
related:
  - webhooks-webhook-payloads.md
  - webhooks-verify-signatures.md
  - data-transfers-api.md
updated: 2026-02-05
---
# Address Activity Webhooks

## Summary
Address activity webhooks notify you when specified addresses send or receive assets.

## Primary Use Cases
- Wallet notifications.
- Compliance monitoring.
- Trigger downstream indexers.

## Spec-Derived Fields
Create webhook (`POST /create-webhook`) with:
- `webhook_type: ADDRESS_ACTIVITY`
- `addresses[]` (required)

Update addresses (`PATCH /update-webhook-addresses`) with:
- `webhook_id` (required)
- `addresses_to_add[]` (required, empty array if none)
- `addresses_to_remove[]` (required, empty array if none)

## Configuration Notes
- Provide one or more addresses to watch.
- Choose event categories (native, ERC-20, ERC-721, ERC-1155).
- Decide if you need internal transfers.

## Payload Expectations
- Expect an array of activities with tx hash, from/to, value, token metadata.

## Operational Guidance
- De-dupe events by tx hash + log index.
- Store mapping of webhook ID to address list.

## Related Files
- `webhooks-webhook-payloads.md`
- `webhooks-verify-signatures.md`
- `data-transfers-api.md`

## Official Docs
- [Notify API Quickstart](https://www.alchemy.com/docs/reference/notify-api-quickstart)

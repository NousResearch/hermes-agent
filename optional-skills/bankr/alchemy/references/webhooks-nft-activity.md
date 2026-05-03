---
id: references/webhooks-nft-activity.md
name: 'NFT Activity Webhooks'
description: 'Receive notifications for NFT transfers, mints, and burns for specified contracts or collections.'
tags:
  - alchemy
  - webhooks
  - webhook
related:
  - webhooks-webhook-payloads.md
  - webhooks-verify-signatures.md
  - data-nft-api.md
updated: 2026-02-05
---
# NFT Activity Webhooks

## Summary
Receive notifications for NFT transfers, mints, and burns for specified contracts or collections.

## Primary Use Cases
- Marketplace feeds.
- Collection-specific alerts.
- Indexing NFT ownership changes.

## Spec-Derived Fields
Create webhook (`POST /create-webhook`) with:
- `webhook_type: NFT_ACTIVITY`
- `nft_filters[]` (required) where each filter can include `contract_address` and optional `token_id` (requires `contract_address` if provided)

Update filters (`PATCH /update-webhook-nft-filters`) with:
- `webhook_id` (required)
- `nft_filters_to_add[]` (required, empty array if none)
- `nft_filters_to_remove[]` (required, empty array if none)

## Configuration Notes
- Filter by collection/contract address.
- Decide which event types matter (mint, transfer, burn).

## Payload Expectations
- Expect token IDs, contract addresses, and from/to addresses.

## Related Files
- `webhooks-webhook-payloads.md`
- `webhooks-verify-signatures.md`
- `data-nft-api.md`

## Official Docs
- [Notify API Quickstart](https://www.alchemy.com/docs/reference/notify-api-quickstart)

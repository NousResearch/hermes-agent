---
id: references/webhooks-details.md
name: 'Webhooks Overview (Notify)'
description: 'Notify webhooks push blockchain events to your server so you don''t need to poll. They are ideal for near real-time pipelines.'
tags:
  - alchemy
  - webhooks
  - webhook
related:
  - webhooks-webhook-types.md
  - webhooks-verify-signatures.md
  - webhooks-address-activity.md
updated: 2026-02-05
---
# Webhooks Overview (Notify)

## Summary
Notify webhooks push blockchain events to your server so you don't need to poll. They are ideal for near real-time pipelines.

## Primary Use Cases
- Activity feeds and notifications.
- Indexer triggers for downstream processing.
- Security monitoring for wallet activity.

## When To Use / Not Use
- Use when you need low-latency updates.
- Avoid if you only need periodic snapshots.

## How It Works
1. Create a webhook (dashboard or API).
2. Configure filters (addresses, contract, event types).
3. Receive POST payloads in your webhook endpoint.
4. Verify the signature and process idempotently.

## Auth Token (Dashboard)
- The Notify API uses an **auth token** for managing webhooks.
- You must generate this token in the Alchemy dashboard.
- Placeholder format for examples: `ALCHEMY_NOTIFY_AUTH_TOKEN=<GENERATE_IN_DASHBOARD>`

## Notify API Endpoints (Spec-Derived)
- Base URL: `https://dashboard.alchemy.com/api`
- Required header for API management calls: `X-Alchemy-Token: <ALCHEMY_NOTIFY_AUTH_TOKEN>`

Common endpoints:
- `POST /create-webhook`
- `PUT /update-webhook`
- `PATCH /update-webhook-addresses`
- `PATCH /update-webhook-nft-filters`
- `GET /team-webhooks`
- `GET /webhook-addresses?webhook_id=...`
- `GET /webhook-nft-filters?webhook_id=...`
- `DELETE /delete-webhook?webhook_id=...`

## Spec-Derived Create Payload
`POST /create-webhook` requires:
- `network` (enum, e.g., `ETH_MAINNET`, `BASE_MAINNET`, `ARB_MAINNET`)
- `webhook_type` (enum: `GRAPHQL`, `ADDRESS_ACTIVITY`, `NFT_ACTIVITY`)
- `webhook_url`

Optional fields:
- `name`
- `addresses[]` (required for `ADDRESS_ACTIVITY`)
- `nft_filters[]` (required for `NFT_ACTIVITY`)
- `graphql_query` (required for `GRAPHQL`; string or `{ query, skip_empty_messages? }`)

## Operational Guidance
- Store webhook IDs for future updates or deletion.
- Implement retries and idempotency using event IDs.
- Persist raw payloads for audit/debugging.

## Related Files
- `webhooks-webhook-types.md`
- `webhooks-verify-signatures.md`
- `webhooks-address-activity.md`

## Official Docs
- [Notify API Quickstart](https://www.alchemy.com/docs/reference/notify-api-quickstart)

---
id: references/webhooks-custom-webhooks.md
name: 'Custom Webhooks (GraphQL)'
description: 'Custom webhooks allow flexible event filtering using a GraphQL query hosted by Alchemy.'
tags:
  - alchemy
  - webhooks
  - webhook
related:
  - webhooks-webhook-types.md
  - webhooks-webhook-payloads.md
updated: 2026-02-05
---
# Custom Webhooks (GraphQL)

## Summary
Custom webhooks allow flexible event filtering using a GraphQL query hosted by Alchemy.

## Primary Use Cases
- Multi-contract event triggers.
- Custom event signatures or ABI decoding.
- Fine-grained filters by topics or addresses.

## Spec-Derived Create Fields
When `webhook_type=GRAPHQL`, `graphql_query` is required and can be:
- A string containing the GraphQL query, or
- An object with fields `query` (string, required) and `skip_empty_messages` (boolean, optional)

## Configuration Notes
- Use the smallest filter that meets your needs to reduce noise.
- Test filters on testnet before mainnet.

## Operational Guidance
- Maintain a versioned set of filters to simplify migrations.

## Related Files
- `webhooks-webhook-types.md`
- `webhooks-webhook-payloads.md`

## Official Docs
- [Custom Webhooks Guide](https://www.alchemy.com/docs/how-to-use-custom-webhooks-for-crypto-token-alerts)

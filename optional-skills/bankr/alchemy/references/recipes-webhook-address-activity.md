---
id: references/recipes-webhook-address-activity.md
name: 'Recipe: Address Activity Webhook'
description: ''
tags:
  - alchemy
  - recipes
  - recipe
  - webhook
related:
  - webhooks-address-activity.md
  - webhooks-verify-signatures.md
updated: 2026-02-05
---
# Recipe: Address Activity Webhook

## Goal
Receive push notifications when a wallet address is involved in on-chain activity.

## Inputs
- `addresses` to watch
- Webhook endpoint URL
- `ALCHEMY_NOTIFY_AUTH_TOKEN=<GENERATE_IN_DASHBOARD>`
- `ALCHEMY_API_KEY` (for chain RPC/data lookups)

## Steps
1. Create an Address Activity webhook in Alchemy dashboard (or via Notify API using your auth token).
2. Configure addresses and categories.
3. Implement the webhook endpoint with signature verification.
4. Store webhook ID and address set for updates.

## Example (Payload Handling)
```ts
export async function handler(req: Request) {
  const rawBody = await req.text();
  const signature = req.headers.get("x-alchemy-signature") || "";
  const secret = process.env.ALCHEMY_WEBHOOK_SECRET || "";

  // Verify signature (see webhooks-verify-signatures.md)
  // ...

  const payload = JSON.parse(rawBody);
  // Process payload.activity
  return new Response("ok");
}
```

## Output
- Real-time activity events for your addresses.

## Related Files
- `webhooks-address-activity.md`
- `webhooks-verify-signatures.md`

## Official Docs
- [Notify API Quickstart](https://www.alchemy.com/docs/reference/notify-api-quickstart)

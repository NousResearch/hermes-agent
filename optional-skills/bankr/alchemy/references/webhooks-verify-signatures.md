---
id: references/webhooks-verify-signatures.md
name: 'Verify Webhook Signatures'
description: 'Always verify webhook signatures to ensure payloads are authentic and untampered.'
tags:
  - alchemy
  - webhooks
  - webhook
related:
  - webhooks-webhook-payloads.md
  - webhooks-details.md
updated: 2026-02-05
---
# Verify Webhook Signatures

## Summary
Always verify webhook signatures to ensure payloads are authentic and untampered.

## Auth Token vs Signing Key
- **Auth token**: used for managing webhooks via the Notify API. Generate it in the Alchemy dashboard.
- Placeholder: `ALCHEMY_NOTIFY_AUTH_TOKEN=<GENERATE_IN_DASHBOARD>`
- **Signing key/secret**: used to verify webhook payload signatures.

## Recommended Steps
1. Read the raw request body (before JSON parsing).
2. Compute the HMAC signature using the webhook signing key/secret.
3. Compare to the `X-Alchemy-Signature` header provided by Alchemy.
4. Reject the request if verification fails.

## Example (Pseudo)
```ts
import crypto from "crypto";

function verify(rawBody: string, signature: string, secret: string) {
  const hmac = crypto.createHmac("sha256", secret).update(rawBody).digest("hex");
  return crypto.timingSafeEqual(Buffer.from(hmac), Buffer.from(signature));
}
```

## Gotchas
- The signature is over the raw body bytes, not the parsed JSON.
- Enforce replay protection if provided (timestamp or nonce).

## Related Files
- `webhooks-webhook-payloads.md`
- `webhooks-details.md`

## Official Docs
- [Notify API FAQ](https://www.alchemy.com/docs/reference/notify-api-faq)

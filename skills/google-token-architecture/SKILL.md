---
name: google-token-architecture
description: Google token storage, identity, and email delivery architecture. Critical reference for avoiding token confusion and email delivery failures.
category: devops
---

# Google Token Architecture

## Token Files (MANDATORY NAMING)
- **Jared's token**: `/root/.hermes/jared_google_token.json` (jared.zimmerman@gmail.com)
- **Indigo's token**: `/root/.hermes/indigo_google_token.json` (mx.indigo.karasu@gmail.com)
- **NEVER** use generic `google_token.json` — always use the explicit prefixed filenames

## OAuth Clients
- Jared's client_id: `112292610034-1revbmnkves56ago2c2t5dul5mj9bc17` (secret: `/root/.hermes/google_client_secret.json`)
- Indigo's client_id: `550801240087-vmc47b1gflj2biblqdr6bkekl7qqm8ls` (secret: `/root/.hermes-indigo/google_client_secret.json`)

## Scopes
Both tokens were re-authorized Apr 15 2026 with ALL possible Google Workspace scopes (gmail.readonly, gmail.send, gmail.modify, calendar, drive, contacts, directory.readonly, spreadsheets, documents, presentations, forms).

## Email Delivery Architecture
- **ocas-dispatch** owns the email lifecycle (send, scan, label, draft)
- **send_message_tool.py** — email path REMOVED; only a dumb pipe for non-email platforms (Telegram, Discord, Slack, etc.)
- **google_api.py** — low-level engine, callable by Dispatch but NOT directly for cron delivery
- **email-delivery-routing** — documentation skill only, not executable code
- **himalaya** — listed as a skill but NOT the correct tool; do NOT use for email
- PR posted: `dispatch-email-ownership` on indigokarasu GitHub

## Critical Rules
1. NEVER overwrite one account's token with another's. This was the root cause of the morning briefing delivery failure (Apr 12-14 2026).
2. When checking email delivery, check BOTH inboxes directly using the correct named tokens — never ask the user to confirm receipt.
3. If `invalid_grant` appears, check which token is actually being used before assuming the API is down.
4. When generating OAuth auth URLs, use the correct client_secret for the target account, and save the PKCE verifier to complete the exchange.

## Cron Delivery
- All cron jobs switched from Telegram delivery to `local` (Apr 15 2026) to stop spamming user's channel.
- If cron output still appears on Telegram, check for hardcoded delivery calls inside skill scripts bypassing the registry.
- Midnight UTC collision: many cron jobs fire simultaneously → rate limit cascade → API 429 errors.
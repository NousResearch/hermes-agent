# Blooio (iMessage) platform plugin

[Blooio](https://blooio.com) is a hosted iMessage API — you send and receive
real iMessages (with automatic SMS/RCS fallback) through a REST API and inbound
webhooks, without running a Mac yourself. This plugin wires Blooio into the
Hermes gateway as a first-class messaging platform, alongside the built-in
BlueBubbles iMessage channel and the Photon Spectrum plugin.

## How it works

- **Inbound = webhooks.** The plugin runs a small `aiohttp` webhook server and
  receives Blooio events at `/blooio/webhook`. Each event is HMAC-SHA256
  signature-verified (Stripe-style `X-Blooio-Signature: t=<ts>,v1=<hex>` over
  `"{ts}.{rawBody}"`), deduped, and dispatched to the agent.
- **Outbound = REST.** Replies POST to `/chats/{chatId}/messages` on the Blooio
  v2 API (`https://api.blooio.com/v2/api`) with your API key as a bearer token.
- **Reactions, typing, read receipts** each map to a dedicated Blooio endpoint.
- **Attachments** are HTTPS URLs — remote URLs pass straight through; local
  files are served from the same webhook server behind `BLOOIO_PUBLIC_URL`.

## Requirements

Blooio delivers inbound messages via webhooks and fetches local-file
attachments from a public URL, so Hermes must be reachable at a **public HTTPS
hostname**. Expose it with Cloudflare Tunnel, Tailscale Funnel, or ngrok and set
`BLOOIO_PUBLIC_URL`.

## Setup

```bash
hermes setup blooio
```

or set the environment variables directly:

```bash
export BLOOIO_API_KEY="sk_live_..."
export BLOOIO_WEBHOOK_SECRET="whsec_..."      # from the webhook you create
export BLOOIO_PUBLIC_URL="https://my-tunnel.example.com"
export BLOOIO_ALLOWED_USERS="+15551234567"    # allowlist senders
```

Then, in the Blooio dashboard, add a webhook pointing at
`<BLOOIO_PUBLIC_URL>/blooio/webhook` (type: `all`) and copy its signing secret
into `BLOOIO_WEBHOOK_SECRET`. (Or set `BLOOIO_AUTO_REGISTER_WEBHOOK=true` to have
the plugin register the webhook and capture the secret on connect.)

See [`website/docs/user-guide/messaging/blooio.md`](../../../website/docs/user-guide/messaging/blooio.md)
for the full walkthrough, including the Cloudflare Tunnel setup and the complete
environment-variable reference.

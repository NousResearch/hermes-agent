# iMessage via Blooio

A Hermes Agent platform plugin that connects your agent to **iMessage** (with
SMS/RCS fallback) through [Blooio](https://blooio.com) — a hosted iMessage API.
No Mac, no jailbreak, no local BlueBubbles server: Blooio runs the iMessage
infrastructure and exposes a REST API + inbound webhooks, and this plugin bridges
it to Hermes.

It's the twin of the built-in **BlueBubbles** iMessage channel and the
**Photon** Spectrum plugin — same conversational behavior, same group-mention
gating, same tapback reactions — but talks to Blooio's **v4 REST API** and
authenticates with the official **"Blooio for Hermes" OAuth app**.

## Quick start

```bash
# 1. Connect Blooio (opens your browser, one-click consent → tokens stored)
hermes blooio login

# 2. Expose Hermes at a public HTTPS URL (Blooio delivers inbound via webhook)
export BLOOIO_PUBLIC_URL="https://your-tunnel.example.com"     # Cloudflare Tunnel / ngrok / Tailscale Funnel
export BLOOIO_AUTO_REGISTER_WEBHOOK=true                       # auto-create the webhook + capture its signing secret

# 3. (optional) lock the bot to specific senders
export BLOOIO_ALLOWED_USERS="+15551234567,teammate@example.com"

# 4. Run the gateway as usual — "iMessage via Blooio" is now a channel.
```

`hermes blooio status` shows the current auth state; `hermes blooio logout`
revokes and clears the stored tokens.

## Authentication

Two modes, resolved in this order:

1. **OAuth (default, recommended).** `hermes blooio login` runs the standard
   native-app flow — Authorization Code + **PKCE (S256)** on a loopback redirect
   (`http://127.0.0.1:8765/callback`). Tokens are stored in `~/.hermes/auth.json`
   (`0o600`) under `credential_pool.blooio`; the access token auto-refreshes via
   the rotating refresh token. If your OAuth app is installed on more than one
   organization, set `BLOOIO_ORG_ID=org_…` (sent as `X-Organization-Id`).
2. **API key (headless / CI).** Set `BLOOIO_API_KEY` and skip the browser login
   entirely.

## How it works

* **Inbound = webhooks.** The plugin runs a small `aiohttp` server and (with
  `BLOOIO_AUTO_REGISTER_WEBHOOK=true`) registers `/blooio/webhook` with Blooio.
  Every event is HMAC-SHA256 signature-verified (Stripe-style
  `X-Blooio-Signature: t=<ts>,v1=<hex>` over `"{ts}.{rawBody}"`), deduped on
  `message_id`, and dispatched as a normalized Hermes `MessageEvent`. Blooio v4
  delivers a typed envelope (`{type, created_at, organization_id, data:{…}}`);
  `message.received` becomes an inbound message and `message.reaction` on one of
  the bot's own messages becomes a synthetic `reaction:add:<emoji>` event.
* **Outbound = REST (v4).** Replies POST to `/v4/chats/{chat_id}/messages` (the
  `chat_…` handle carried by every inbound event; the sender is inferred from
  the chat). Cron / home-channel delivery to a bare phone/email uses
  `POST /v4/messages` with `to` (+ optional `from` channel). Long replies are
  chunked. Typing indicators, read receipts, and tapback/emoji reactions each
  map to a dedicated v4 chat endpoint.
* **Media.** Blooio attachments are HTTPS URLs. Remote image/file URLs pass
  straight through; local files are served from the same `aiohttp` app
  (`/blooio/media/<token>/<name>`, with a path-traversal guard) — which is why a
  public `BLOOIO_PUBLIC_URL` is required to send local files.

## Requirements

* A public HTTPS hostname reachable by Blooio (inbound webhooks + local-file
  attachment URLs). Use Cloudflare Tunnel, Tailscale Funnel, or ngrok and set
  `BLOOIO_PUBLIC_URL`.
* `aiohttp` and `httpx` (declared by Hermes; already present in the standard
  install).

## Configuration

All settings are environment variables (see `plugin.yaml` for the full list with
prompts). The most useful:

| Variable | Purpose |
| --- | --- |
| `BLOOIO_PUBLIC_URL` | Public HTTPS base URL Blooio can reach (required for inbound + local media). |
| `BLOOIO_AUTO_REGISTER_WEBHOOK` | Auto-create the webhook and capture its signing secret on connect (`true`/`false`). |
| `BLOOIO_WEBHOOK_SECRET` | Webhook signing secret (`whsec_…`) if you register the webhook manually. |
| `BLOOIO_ORG_ID` | Organization id (`org_…`); needed only for multi-org OAuth tokens. |
| `BLOOIO_API_KEY` | Headless/CI auth instead of OAuth login. |
| `BLOOIO_CHANNEL` | Channel id (`ch_…`) / phone to send *from* for routed (non-reply) sends. |
| `BLOOIO_ALLOWED_USERS` | Comma-separated senders (phone/email) allowed to DM the bot. |
| `BLOOIO_ALLOWED_GROUPS` | Comma-separated group ids (`grp_…`) the bot responds in. |
| `BLOOIO_ALLOW_ALL_USERS` | Disable the allowlist (dev only). |
| `BLOOIO_REQUIRE_MENTION` | Ignore group messages without a wake word (`hermes …`). |
| `BLOOIO_REACTIONS` | 👀/👍/👎 processing tapbacks + route reactions on the bot's messages back to the agent. |
| `BLOOIO_SEND_READ_RECEIPTS` | Send iMessage read receipts for processed inbound messages. |
| `BLOOIO_HOME_CHANNEL` | Default target (`chat_…`, phone, or email) for cron / notifications. |

## Notes

* iMessage identifiers are E.164 phone numbers / emails, so the channel is
  treated as **PII-sensitive** (redacted before reaching the LLM), matching the
  BlueBubbles and Photon channels.
* iMessage does **not** render Markdown — replies are sent as plain text (markup
  is stripped); bare URLs still get a rich preview.

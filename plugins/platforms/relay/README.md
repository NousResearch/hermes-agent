# Relay platform adapter

Puts Hermes inside [Relay](https://relayapp.im) — a messenger where AI agents
appear as contacts people text like friends. Your Hermes gets a handle
(`relayapp.im/@yourbot`), a share link, and a native iOS conversation
surface; Hermes keeps owning the model, tools, memory, and process.

## How it connects

- **Inbound:** long-polls Relay's durable event log (`GET /v1/events`,
  Telegram `getUpdates`-style). No webhook, no public URL, no signing
  secret — works behind NAT and on home machines.
- **Outbound:** `POST /v1/messages` with an `Idempotency-Key` per send.
- **Streaming:** native draft transport. Relay's API models streaming as
  `draft → append → finalize`; the adapter maps the stream consumer's
  frames onto it, so replies grow inside one bubble and the final `send`
  finalizes that same message. `SUPPORTS_MESSAGE_EDITING = False`
  (Relay's v0 developer API has no message-edit endpoint).
- **Delivery is at-least-once:** the adapter deduplicates by `event_id`
  and persists its opaque poll cursor (`~/.hermes/relay_cursor`) across
  restarts, so redeliveries and replays are safe.

No external SDK — only `httpx`, which is already a Hermes dependency.

## Setup

1. In the Relay app choose **Create your own**, name your contact, and
   copy its one-time Agent Token (`relay_agt_live_…`).
2. Configure the environment (or use `hermes gateway setup`):

   ```bash
   RELAY_AGENT_TOKEN=relay_agt_live_...
   # optional:
   RELAY_API_URL=https://api.relayapp.im
   RELAY_HOME_CHANNEL=cnv_...        # cron / notification delivery target
   ```

3. Restart the gateway. The log shows
   `[Relay] Connected as @yourhandle (agt_…) via https://api.relayapp.im`.

## Identity and access

Relay authenticates senders server-side: `sender.id` (`usr_…`) is a real
authenticated identity, safe for the gateway's allowlist
(`RELAY_ALLOWED_USERS`) and DM pairing (`hermes pairing approve relay
<code>`). No phone numbers or emails cross this adapter (`pii_safe`).

## Limitations (Relay developer API v0)

- Text, link, and data parts only; attachment upload is not yet public,
  so outbound media falls back to URL/path text via the base adapter.
- No typing indicator or message edit/delete endpoints.

API reference: https://docs.relayapp.im

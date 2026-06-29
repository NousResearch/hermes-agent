# Sendblue platform plugin

Sendblue connects Hermes Agent to iMessage, SMS, and RCS through Sendblue's
hosted REST API and receive webhooks.

This is a bundled platform plugin, not a core gateway adapter. It follows the
same Hermes platform-plugin surface as other bundled channels:

- `plugin.yaml` declares `kind: platform` and the `SENDBLUE_*` environment
  variables surfaced by `hermes config` / `hermes gateway setup`.
- `adapter.py` defines `SendblueAdapter(BasePlatformAdapter)`.
- `register(ctx)` calls `ctx.register_platform(name="sendblue", ...)`.
- The registered platform opts into allowlists, pairing, cron delivery,
  standalone `send_message` delivery, gateway status, and platform hints through
  the plugin registry instead of editing Hermes core files.

## Architecture

```text
User phone
  <-> Sendblue iMessage/SMS/RCS line
  <-> Sendblue receive webhook + REST API
  <-> SendblueAdapter
  <-> Hermes gateway runner
```

Inbound messages arrive as Sendblue receive webhooks. Hermes validates the
configured webhook secret, deduplicates by `message_handle`, normalizes the
payload into a `MessageEvent`, and calls `BasePlatformAdapter.handle_message`.

Outbound replies use:

- `/api/send-message` for direct messages
- `/api/send-group-message` for `group:<id>` targets

Sendblue requires a `from_number` on every send. If multiple Sendblue lines are
configured, the adapter keeps a sticky recipient -> `from_number` map so each
conversation continues from the same visible line.

## Configuration

Required:

```bash
SENDBLUE_API_KEY_ID=...
SENDBLUE_API_SECRET_KEY=...
SENDBLUE_FROM_NUMBER=+15551234567
SENDBLUE_WEBHOOK_SECRET=...
```

Recommended:

```bash
SENDBLUE_ALLOWED_USERS=+15559876543
SENDBLUE_HOME_CHANNEL=+15559876543
```

Optional:

```bash
SENDBLUE_FROM_NUMBERS=+15551234567,+15557654321
SENDBLUE_WEBHOOK_HOST=127.0.0.1
SENDBLUE_WEBHOOK_PORT=8650
SENDBLUE_WEBHOOK_PATH=/sendblue/webhook
SENDBLUE_STATUS_CALLBACK=https://example.com/sendblue/status
SENDBLUE_SEAT_ID=...
SENDBLUE_STICKY_STATE_PATH=~/.hermes/sendblue_sticky_senders.json
```

Use `SENDBLUE_INSECURE_NO_SIGNATURE=true` only for local development. Public
gateways should always configure a receive webhook secret.

## CLI

The plugin registers a small management surface:

```bash
hermes sendblue status
hermes sendblue setup
```

`status` reports whether credentials, line selection, webhook binding, access
control, home channel, and sticky state are configured without printing secrets
or full phone numbers.

`setup` delegates to the same prompts used by `hermes gateway setup`.

## Notes

- Phone numbers are PII-sensitive; the platform registers `pii_safe=True` so
  session descriptions are redacted before LLM-visible context.
- Attachment-only inbound webhooks are accepted and dispatched with
  `(attachment)` placeholder text plus `media_urls` metadata.
- Local generated files from cron/send-message are not uploaded automatically.
  Use Sendblue `media_url` metadata when sending outbound media.

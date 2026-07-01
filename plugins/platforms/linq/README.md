# Linq iMessage platform plugin

A **Linq iMessage** gateway adapter for Hermes Agent. Send and receive real
iMessage "blue bubbles" through the hosted [Linq](https://linqapp.com) partner
API — **no Mac and no BlueBubbles server required**.

This is a bundled platform plugin: it ships in-tree under
`plugins/platforms/linq/` and sits alongside the bundled **BlueBubbles** and
**Photon** iMessage channels as a third independent way to reach iMessage.

## How it compares to the other iMessage channels

| | BlueBubbles | Photon | **Linq (this plugin)** |
|---|---|---|---|
| Requires a Mac | ✅ yes | ❌ no | ❌ no |
| Outbound transport | BlueBubbles REST | Node `spectrum-ts` **sidecar** | **direct Linq REST** |
| Inbound transport | webhook | signed webhook | **signed webhook** |
| Extra runtime deps | BlueBubbles server | Node 18+ | none beyond Hermes |

Because Linq exposes a public send endpoint, this adapter needs **no Node
sidecar** — every outbound message, typing indicator, and read receipt is a
direct `httpx` call.

## Dependencies

The plugin is bundled, so there is nothing to install — the gateway discovers
it automatically. Its one optional dependency is `aiohttp` (for the inbound
webhook listener), which ships with the `hermes-agent[messaging]` extra rather
than the core install. If `hermes linq status` reports it missing:

```bash
pip install aiohttp
```

## Setup

Run the one-time setup (stores your token + from-phone in `~/.hermes/auth.json`):

```bash
hermes linq setup
```

You'll be asked for:

1. **Linq API token** — from your [linqapp.com](https://linqapp.com) dashboard
   (or set `LINQ_API_TOKEN` in your environment instead).
2. **From-phone** (optional) — the Linq number this agent sends from, in E.164
   (`+15551234567`). Only needed to pin a multi-number account to one line.

Linq also surfaces in the unified wizard:

```bash
hermes gateway setup        # pick "Linq iMessage"
```

### Register the inbound webhook

Linq delivers inbound iMessages as webhooks. Point your Linq dashboard at the
gateway's public URL:

```bash
hermes linq webhook show --public-url https://your-public-host
# → register  https://your-public-host/linq/webhook  in the Linq dashboard
```

Then export the signing secret the dashboard gives you so deliveries are
verified:

```bash
export LINQ_WEBHOOK_SECRET=<secret-from-linq-dashboard>
```

### Start the gateway

```bash
hermes gateway start --platform linq
```

Verify any time with:

```bash
hermes linq status     # credential state + live connectivity probe
```

## Configuration

Everything is configurable by environment variable, by `~/.hermes/config.yaml`,
or in `~/.hermes/auth.json`. Precedence is **env → config.yaml → auth.json**.

### `config.yaml`

```yaml
platforms:
  linq:
    enabled: true
    extra:
      from_phone: "+15551234567"
      webhook_port: 8790
      webhook_path: /linq/webhook
      send_read_receipts: true
      require_mention: false
      mention_patterns:
        - '(?<![\w@])@?hermes\b[,:\-]?'
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LINQ_API_TOKEN` | — | **Required.** Linq partner API bearer token |
| `LINQ_FROM_PHONE` | — | Pin a multi-number account to one Linq line (E.164) |
| `LINQ_WEBHOOK_SECRET` | — | HMAC-SHA256 secret for inbound webhook verification |
| `LINQ_WEBHOOK_PORT` | `8790` | Local webhook listener port |
| `LINQ_WEBHOOK_PATH` | `/linq/webhook` | Local webhook listener path |
| `LINQ_WEBHOOK_BIND` | `0.0.0.0` | Webhook listener bind address |
| `LINQ_API_BASE` | `https://api.linqapp.com/api/partner/v3` | Linq API base URL |
| `LINQ_SEND_READ_RECEIPTS` | `true` | Send read receipts + typing on inbound |
| `LINQ_ALLOWED_USERS` | — | Comma-separated allowlist of E.164 senders |
| `LINQ_ALLOW_ALL_USERS` | `false` | Allow any sender (dev only) |
| `LINQ_REQUIRE_MENTION` | `false` | Gate group chats on a wake word |
| `LINQ_MENTION_PATTERNS` | Hermes wake words | Group mention regexes |
| `LINQ_HOME_CHANNEL` | — | Default chat id for cron / notification delivery |

## Authorization

The gateway denies unknown senders by default. Authorize them one of two ways
(same model as every Hermes platform):

1. **DM pairing** — a new sender gets a code; approve it:
   ```bash
   hermes pairing approve linq <CODE>
   ```
2. **Pre-authorize** — `export LINQ_ALLOWED_USERS=+15551234567,+15557654321`

Set `LINQ_ALLOW_ALL_USERS=true` only for local development.

## Features

- Real iMessage blue bubbles via the Linq API — no Mac
- Inbound via HMAC-SHA256 signed webhooks (replay-protected, 5-min drift window)
- At-least-once delivery dedup on `message.id`
- Outbound text + media-by-URL, typing indicators, read receipts
- Inbound image attachments downloaded locally for the vision tools
- Group-chat mention gating (parity with the Photon / BlueBubbles channels)
- E.164 phone numbers treated as PII and redacted before reaching the LLM
- Cron / `send_message` delivery, including out-of-process (`standalone_sender`)
- Unified `hermes gateway setup` onboarding + a dedicated `hermes linq` CLI

## Architecture

```
Inbound:   iMessage → Linq → signed webhook → aiohttp listener → MessageEvent → agent
Outbound:  agent → LinqClient (httpx) → Linq REST API → iMessage
```

- `adapter.py` — `LinqAdapter(BasePlatformAdapter)`, the webhook server, and the
  `register(ctx)` plugin entry point.
- `linq_api.py` — async Linq REST client (send / typing / read / reaction / probe).
- `signing.py` — dependency-free webhook-signature verification, mention gating,
  and payload parsing (unit-tested in isolation).
- `auth.py` — credential storage in `~/.hermes/auth.json`.
- `cli.py` — `hermes linq {setup,status,probe,webhook show}`.

## Development

The security-critical and parsing logic lives in `signing.py` and `auth.py`,
which import nothing from Hermes, so the test suite runs anywhere:

```bash
python -m pytest tests/plugins/platforms/linq/ -v
```

`adapter.py`, `cli.py`, and `linq_api.py` import the host `gateway.*` package
and Hermes' `BasePlatformAdapter`, so they exercise fully only inside a Hermes
runtime. Tests for the bundled plugin live at
`tests/plugins/platforms/linq/` and run under the repo's pytest suite.

## Notes / assumptions to confirm against a live account

- **Group detection.** Linq's documented `message.received` payload (mirrored
  from the OpenClaw plugin) does not include a first-class chat-type field, so
  `signing.is_group_chat()` infers group vs. DM from `is_group` / `group_id` /
  `participants`. Confirm against a real Linq group webhook and tighten if Linq
  exposes an explicit type.
- **Outbound media.** Linq sends media by **public URL**, not multipart upload,
  so `send_image` forwards a URL and the standalone/cron path skips local files
  with a logged note. If your Linq plan offers an upload endpoint, wire it into
  `LinqClient.send_message`.
- **Webhook signature.** Verification matches the OpenClaw plugin's scheme
  (`hex(hmac_sha256(secret, "{timestamp}.{body}"))`, `X-Webhook-Timestamp` /
  `X-Webhook-Signature`). Adjust `signing.verify_signature` if your dashboard
  documents a different header layout.

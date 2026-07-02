# WeChat (Wechaty) platform plugin

This plugin connects Hermes Agent to **personal WeChat** through
[Wechaty](https://wechaty.js.org/) — a Node library with pluggable
"puppet" backends. Hermes runs Wechaty inside a small supervised
**Node sidecar** and talks to it over loopback HTTP.

Use this plugin when you need a **real personal WeChat account** (including
group `@`-mentions). For Tencent's official iLink bot API, use the
built-in [Weixin](../../../website/docs/user-guide/messaging/weixin.md)
adapter instead.

## Architecture

Wechaty is a **persistent-connection** channel — no public URL or webhook.

```
                         Wechaty puppet
┌─────────────────────────┐ ◄──────────────► ┌──────────────────────┐
│  WeChat (personal acct) │   login / msgs   │  Node sidecar        │
└─────────────────────────┘                  │  (plugins/…/sidecar) │
                                             └──────────┬───────────┘
                                        GET /inbound (NDJSON) │  ▲ POST /send
                                        inbound events        ▼  │ /typing
                                               ┌──────────────────────┐
                                               │  WechatyAdapter       │
                                               │  (Python, in gateway) │
                                               └──────────────────────┘
```

- **Inbound** — the sidecar streams normalized events over loopback
  `GET /inbound` (NDJSON). The adapter dedupes on `messageId` and dispatches
  `MessageEvent` to the gateway.
- **Outbound** — `send` / `send_typing` / file helpers POST to the sidecar
  (`/send`, `/send-file`, `/typing`), authenticated with
  `X-Hermes-Sidecar-Token`.

## Weixin vs Wechaty

| | Weixin (built-in) | Wechaty (this plugin) |
|--|-------------------|----------------------|
| API | Tencent iLink (official) | Wechaty puppet (often unofficial) |
| Identity | iLink bot (`@im.bot`) | Real personal WeChat account |
| Group @-mentions | Usually does not work on iLink | Supported (with `require_mention`) |
| Stability / ToS | Lower ban risk | Higher maintenance + account risk |
| When to use | Official path, DMs to bot | Need real account + group chat |

:::warning
Unofficial WeChat protocols may violate Tencent's Terms of Service and
can lead to account restrictions. Prefer [Weixin](../../../website/docs/user-guide/messaging/weixin.md)
or [WeCom](../../../website/docs/user-guide/messaging/wecom.md) when
official APIs meet your needs.
:::

## Prerequisites

- **Node.js 18.17+** on PATH (`node --version`)
- A Wechaty puppet — `wechaty-puppet-wechat4u` works without a paid token
  (trial / dev); production setups often use `wechaty-puppet-service` with a
  PadLocal or WorkPro token
- Python package `httpx` (included in Hermes messaging dependencies)

## First-time setup

```bash
# Interactive wizard (installs sidecar deps, prompts for puppet/token)
hermes gateway setup
# Select "WeChat (Wechaty)"
```

Or configure manually:

```bash
cd plugins/platforms/wechaty/sidecar   # or ~/.hermes/hermes-agent/plugins/…
npm install
```

Add to `~/.hermes/.env`:

```bash
WECHATY_PUPPET=wechaty-puppet-wechat4u
# For wechaty-puppet-service:
# WECHATY_PUPPET=wechaty-puppet-service
# WECHATY_PUPPET_SERVICE_TOKEN=your-token
```

Enable in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    wechaty:
      enabled: true
```

Start the gateway and scan the QR code shown in the logs:

```bash
hermes gateway run
```

## Authorization

**DM pairing (default).** Unknown senders receive a pairing code; approve with:

```bash
hermes pairing approve wechaty <CODE>
```

**Pre-authorize contacts** (in `~/.hermes/.env`):

```bash
WECHATY_ALLOWED_USERS=contact:id1,contact:id2
```

**Open access** (dev only):

```bash
WECHATY_ALLOW_ALL_USERS=true
```

## Group @-mentions

By default (`WECHATY_REQUIRE_MENTION=true`), the bot only responds in group
chats when `@`-mentioned or when the message matches a wake-word pattern
(default: `@hermes`).

```yaml
gateway:
  platforms:
    wechaty:
      enabled: true
      extra:
        require_mention: true
        mention_patterns: ["@hermes", "@?hermes\\s+agent"]
```

DMs are never mention-gated.

## Cron delivery

Set a home channel for `deliver=wechaty` cron jobs:

```bash
WECHATY_HOME_CHANNEL=contact:wxid_abc123
# or for groups:
WECHATY_HOME_CHANNEL=room:room-id-here
WECHATY_HOME_CHANNEL_NAME=Ops Group
```

`chatId` format matches the sidecar: `contact:<id>` for DMs,
`room:<id>` for groups.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Sidecar deps missing | `cd sidecar && npm install` |
| Port already in use | Set `WECHATY_SIDECAR_PORT` to a free port |
| QR not visible | Check `gateway.log` for `wechaty.js.org/qrcode/` URL |
| Group messages ignored | Ensure `@` mention or set `require_mention: false` |
| Login keeps failing | Try a different puppet; wechat4u is fragile |

## Files

| Path | Role |
|------|------|
| `adapter.py` | Python gateway adapter |
| `sidecar/index.mjs` | Node Wechaty bridge |
| `plugin.yaml` | Plugin manifest + env var metadata |

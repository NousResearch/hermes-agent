# Rabbit R1 platform plugin

This plugin connects Hermes Agent to a [Rabbit R1][r1] hardware device. The
R1 is a pocket AI companion with a 2.88-inch touchscreen and voice output; this
adapter lets it talk to your own Hermes agent instead of the stock cloud
assistant.

It ships as a self-contained platform plugin under
`plugins/platforms/rabbit_r1/` and wires itself into the gateway entirely
through `ctx.register_platform(...)` — **no core files are modified**.

## Architecture

Unlike webhook or long-poll channels, the R1 dials *in*: the adapter runs a
**WebSocket server** that the device connects to and speaks the
OpenClaw/clawdbot-gateway protocol. An optional tunnel (Tailscale Funnel or
Cloudflare) publishes that server so the R1 can reach it from anywhere, not
just the home LAN. A QR code printed on startup carries the public URL + auth
token for one-scan pairing.

```
                       wss:// (clawdbot-gateway proto)
┌──────────────────┐  ◄──────────────────────────────►  ┌──────────────────────┐
│  Rabbit R1        │   pair / message / keepalive        │  RabbitR1Adapter     │
│  (hardware)       │                                     │  WebSocket server    │
└──────────────────┘                                      └──────────┬───────────┘
        ▲   scans QR (public URL + token)                            │ MessageEvent
        │                                                            ▼
┌──────────────────┐        Tailscale Funnel /            ┌──────────────────────┐
│  Terminal QR PNG  │ ◄───── Cloudflare tunnel ─────────► │  Hermes gateway       │
│  ~/.hermes/…qr.png│           (public URL)              │  (agent core)         │
└──────────────────┘                                      └──────────────────────┘
```

- **Inbound**: the device opens a WebSocket, authenticates with a hex token
  (timing-safe compare + per-IP rate limiting on failures), and streams
  messages. The adapter normalizes each one into a `MessageEvent` for the
  gateway.
- **Outbound**: the agent's reply is sent back over the same socket. Responses
  are capped at 2000 chars to stay readable on the small screen, and a
  server→device keepalive (default every 300 s) holds the connection open
  under the R1's ~30-minute inactivity timeout.

## First-time setup

```bash
# Optional guided prompts (token, tunnel, port, allowed devices):
hermes setup rabbit_r1

# Start the gateway — a pairing QR code is printed on startup:
hermes gateway start
```

Scan the printed QR code with your R1 to pair. The token is auto-generated if
you leave it blank; the QR PNG is also saved to `~/.hermes/rabbit_r1_qr.png` so
you can re-open or share it later.

## Dependencies

```
websockets   # required — the WebSocket server
qrcode       # optional — terminal QR code display for pairing
```

Install with `pip install websockets qrcode`. If `websockets` is missing the
platform stays hidden from `hermes status`.

## Configuration

All settings are optional and read from the environment (surfaced in
`hermes setup` / `hermes config` via `plugin.yaml`). Only `RABBIT_R1_TOKEN` is
a secret.

| Variable | Default | Description |
|----------|---------|-------------|
| `RABBIT_R1_TOKEN` | auto-generated | 64-char hex auth token the R1 presents when pairing. |
| `RABBIT_R1_PORT` | `18789` | Local WebSocket server listen port. |
| `RABBIT_R1_TUNNEL` | `tailscale` | Public tunnel mode: `tailscale`, `cloudflare`, or `none`. |
| `RABBIT_R1_PUBLIC_URL` | *(auto)* | Explicit `wss://` URL to advertise in the QR code (overrides tunnel auto-detection). |
| `RABBIT_R1_KEEPALIVE_INTERVAL` | `300` | Seconds between server→device keepalive heartbeats. |
| `RABBIT_R1_ALLOWED_USERS` | *(any)* | Comma-separated device IDs allowed to talk to the bot. |
| `RABBIT_R1_ALLOW_ALL_USERS` | `false` | Allow any device without allowlisting (token still required; dev only). |
| `RABBIT_R1_HOME_CHANNEL` | *(unset)* | Default device ID for cron / notification delivery. |

## Security notes

- **Token auth** is timing-safe (`secrets.compare_digest`) with per-IP rate
  limiting (5 failures / 300 s) to blunt brute-force attempts.
- **Device IDs** (64-char hex) are redacted in this adapter's own logs, so the
  plugin declares `pii_safe=False` to the registry.
- Cron/out-of-process delivery returns a descriptive error: the R1 is only
  reachable from the live gateway process that owns the WebSocket server, not
  from a standalone `hermes` invocation.

[r1]: https://www.rabbit.tech/

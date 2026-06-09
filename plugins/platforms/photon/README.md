# Photon iMessage platform plugin

This plugin connects Hermes Agent to iMessage (and WhatsApp Business +
future Spectrum interfaces) through [Photon][photon] — a managed
service that handles the iMessage line allocation, delivery, and
abuse-prevention layer so users don't have to run their own Mac
relay.

The free tier uses Photon's shared iMessage line pool (`type: shared`)
and is the path we recommend for everyone who doesn't already pay for a
dedicated number.

## Architecture

```
┌─────────────────────────┐    HMAC-signed POSTs      ┌──────────────────┐
│  Photon Spectrum cloud  │ ──────────────────────►   │  Hermes Agent    │
│  (iMessage line owner)  │                           │  (Python)        │
└─────────────────────────┘    JSON over loopback     │                  │
        ▲                  ◄──────────────────────    │  PhotonAdapter   │
        │                                             │  + aiohttp recv  │
        │  spectrum-ts                                │                  │
        │  SDK (Node)                                 │  spawns + super- │
        ▼                                             │  vises ▼         │
┌─────────────────────────┐                           ├──────────────────┤
│  Node sidecar           │   ◄────  X-Hermes-      ─ │  Node sidecar    │
│  (plugins/.../sidecar)  │       Sidecar-Token       │  child process   │
└─────────────────────────┘                           └──────────────────┘
```

Inbound traffic is webhook-only — Hermes runs an aiohttp listener
that verifies `X-Spectrum-Signature` and dedupes on `message.id`.

Outbound traffic goes through a tiny Node sidecar that runs the
`spectrum-ts` SDK. Photon does not currently expose an HTTP
send-message endpoint; their own docs say:

> Pass `space.id` to `Space.send(...)` from a separate `spectrum-ts`
> SDK instance to reply. **No public HTTP send endpoint exists today.**
> — https://photon.codes/docs/webhooks/events

When Photon ships an HTTP send endpoint, `_sidecar_send` is the one
function that swaps and the sidecar disappears. The rest of the
plugin stays the same.

## First-time setup

```bash
# 1. One-shot setup: device login (opens browser) + project + user + sidecar deps
hermes photon setup --phone +15551234567

# 2. Expose your webhook URL to the public internet
#    (cloudflared, ngrok, your gateway's public hostname, etc.)
#    Then register it with Photon:
hermes photon webhook register https://your-host.example.com/photon/webhook

# 3. Save the signing secret it prints to ~/.hermes/.env
#    as PHOTON_WEBHOOK_SECRET=...
#    Photon only returns it ONCE.

# 4. Start the gateway
hermes gateway start --platform photon
```

`hermes photon setup` runs the RFC 8628 device-code login as its first
step — it opens `https://app.photon.codes/` for approval, then
provisions the Spectrum project + iMessage line. There is no separate
`login` command; like every other Hermes channel, onboarding goes
through one setup surface. Re-running `setup` reuses an existing token
and project, so it's safe to run again to finish a partial setup.

## Credentials

Stored in `~/.hermes/auth.json` under `credential_pool`:

```jsonc
{
  "credential_pool": {
    "photon": [
      { "access_token": "<dashboard-bearer>", "issued_at": ... }
    ],
    "photon_project": [
      { "project_id": "...", "project_secret": "...", "name": "Hermes Agent" }
    ]
  }
}
```

The per-URL webhook signing secret is treated like an API key and
lives in `~/.hermes/.env` as `PHOTON_WEBHOOK_SECRET`.

## Configuration knobs

All env vars are documented in `plugin.yaml`. The most important are:

| Env var                  | Default            | Meaning                                 |
|--------------------------|--------------------|-----------------------------------------|
| `PHOTON_PROJECT_ID`      | from auth.json     | Spectrum project ID                     |
| `PHOTON_PROJECT_SECRET`  | from auth.json     | Spectrum project secret (HTTP Basic)    |
| `PHOTON_WEBHOOK_SECRET`  | (unset)            | Signing secret returned at register     |
| `PHOTON_WEBHOOK_PORT`    | 8788               | Local port for the aiohttp listener     |
| `PHOTON_WEBHOOK_PATH`    | /photon/webhook    | Path under which the listener mounts    |
| `PHOTON_SIDECAR_PORT`    | 8789               | Loopback port for sidecar control      |
| `PHOTON_HOME_CHANNEL`    | (unset)            | Default space ID for cron delivery     |
| `PHOTON_ALLOWED_USERS`   | (unset)            | Comma-separated E.164 allowlist        |

## Capability coverage

This plugin exposes the **full reachable surface** of the `spectrum-ts`
iMessage provider — the layer Photon's managed (incl. free shared-line)
service makes available. Mapped against the SDK:

**Covered (in / out, end to end):**

| Capability | How |
|---|---|
| Send text (+ native link previews) | `/send` → `space.send(text)` |
| Screen/bubble effects | `/send` `effect` → `space.send(effect(...))` |
| Send image / GIF / video / voice / sticker / document | `/send-attachment` → `space.send(attachment()/voice())`; documents use an `application/octet-stream` MIME fallback |
| Multiple attachments (paced) | `/send-attachments` |
| Bot → user tapbacks (👀/✅/❌ acks) | `/react` → `space.send(reaction(...))` |
| Typing indicator (tracks compute) | `/typing` → `space.send(typing(...))` |
| Read receipts | auto on every inbound → `space.read(message)` |
| Inbound text | gRPC stream → adapter |
| Inbound photo → vision | `attachments.downloadStream` → temp file → `media_urls` |
| Inbound video → keyframes + audio transcript | ffmpeg keyframes + WAV → vision routing + STT |
| Inbound voice memo → transcription | `.caf` → `.wav` transcode → STT |
| Inbound reactions (observe) | gRPC stream (unreliable on the free shared line) |

Outbound text is markdown-stripped and split into paragraph bubbles
(BlueBubbles-parity); attachments over `PHOTON_MAX_ATTACHMENT_MB` forward
a note instead of blocking.

**Reliability:** `spectrum-ts` tracks a cursor and runs
`catchUpThenConsumeLive()` internally, so the live `app.messages` stream
replays missed events on a gRPC reconnect — no message loss within a
process. The stream takes no resume cursor, so messages during a full
sidecar *restart* (a brief window) aren't recovered.

**Not reachable from this layer.** `edit`, `unsend`, threaded `reply`,
`sendMultipart`, `placeSticker`, `listRecent` (history), `polls`,
`groups`, `addresses` (`isIMessageAvailable`), `getEmbeddedMedia`,
`locations`, and `chats.create` (proactive cold-start DM to a new
handle) live on the lower `@photon-ai/advanced-imessage` client, which
`spectrum-ts` holds as internal context (`client: AdvancedIMessage`) and
does **not** expose — the provider only surfaces `getAttachment`,
`getMessage`, `read`, `background`. Reaching them would require a
*separate* direct advanced-imessage connection (its own auth, likely a
paid/dedicated line, not the free shared tier). When that's wired, the
SOTA exposure is a few **semantic** agent tools (send / manage-message /
manage-chat / history) with capability-gating and confirmation for
destructive ops (`unsend`, group changes) — not raw 1:1 endpoints.

[photon]: https://photon.codes/

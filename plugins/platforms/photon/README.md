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
# 1. Log in via the device-code flow (opens browser)
hermes photon login

# 2. Full setup: project, user, sidecar deps
hermes photon setup --phone +15551234567

# 3. Expose your webhook URL to the public internet
#    (cloudflared, ngrok, your gateway's public hostname, etc.)
#    Then register it with Photon:
hermes photon webhook register https://your-host.example.com/photon/webhook

# 4. Save the signing secret it prints to ~/.hermes/.env
#    as PHOTON_WEBHOOK_SECRET=...
#    Photon only returns it ONCE.

# 5. Start the gateway
hermes gateway start --platform photon
```

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

## Limitations (current Photon API)

- **Attachments are metadata only.** Inbound webhooks include the
  filename + MIME type but no download URL. The plugin surfaces a
  text marker (`[Photon attachment received: …]`) so the agent knows
  something arrived, but cannot read the bytes.  Photon's docs note
  an attachment retrieval endpoint is on the roadmap.
- **Outbound attachments are not supported yet.** Adding them is
  straightforward once the sidecar wires up `attachment(...)` /
  `space.send(attachment(...))` from `spectrum-ts`.
- **Reactions, message effects, polls** — not exposed yet; the
  `spectrum-ts` SDK supports them, and the sidecar is the natural
  place to add them when the agent has reason to use them.

[photon]: https://photon.codes/

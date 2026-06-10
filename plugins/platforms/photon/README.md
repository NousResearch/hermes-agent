# Photon iMessage platform plugin

This plugin connects Hermes Agent to iMessage (and other Spectrum
interfaces) through [Photon][photon] — a managed service that handles
iMessage line allocation, delivery, and abuse-prevention so users don't
have to run their own Mac relay.

The free tier uses Photon's shared iMessage line pool and is the path we
recommend for everyone who doesn't already pay for a dedicated number.

## Architecture

Like Discord and Slack, Photon is a **persistent-connection** channel — no
public URL, no webhook, no signing secret. The `spectrum-ts` SDK holds a
long-lived **gRPC stream** to Photon for both directions. Because the SDK is
TypeScript-only, Hermes runs it inside a small supervised Node sidecar and
talks to it over loopback.

```
                         gRPC (spectrum-ts)
┌─────────────────────────┐ ◄───────────────► ┌──────────────────────┐
│  Photon Spectrum cloud  │   app.messages    │  Node sidecar        │
│  (iMessage line owner)  │   space.send()    │  (plugins/…/sidecar) │
└─────────────────────────┘                   └──────────┬───────────┘
                                       GET /inbound (NDJSON) │  ▲ POST /send
                                       inbound events        ▼  │ /typing
                                              ┌──────────────────────┐
                                              │  PhotonAdapter        │
                                              │  (Python, in gateway) │
                                              └──────────────────────┘
```

- **Inbound**: the sidecar consumes the SDK's `app.messages` gRPC stream,
  normalizes each message, and streams it to the adapter over a loopback
  `GET /inbound` (NDJSON). The adapter dedupes on `messageId` and dispatches
  a `MessageEvent` to the gateway. It reconnects automatically if the stream
  drops; the sidecar owns the gRPC reconnect to Photon.
- **Outbound**: `send` / `send_typing` are loopback POSTs to the sidecar,
  authenticated with a shared `X-Hermes-Sidecar-Token`.

## First-time setup

```bash
# One-shot setup: device login (opens browser) + project + user + sidecar deps
hermes photon setup --phone +15551234567

# Start the gateway
hermes gateway start --platform photon
```

`hermes photon setup` does, in order:

1. **Device login** (RFC 8628, `client_id=photon-cli`) — opens
   `https://app.photon.codes/` for approval and stores the bearer token.
2. **Find or create** the `Hermes Agent` project on the Photon dashboard.
3. **Enable Spectrum**, read the project's `spectrumProjectId`, rotate the
   project secret, and persist both.
4. **Register your phone number** as a Spectrum user (idempotent — skipped if
   a user with that number already exists).
5. **Print the assigned iMessage line** — the number you text to reach your
   agent.
6. **Install the sidecar deps** (`spectrum-ts`).

There is no separate `login` command; like every other Hermes channel,
onboarding goes through one setup surface. Re-running `setup` reuses an
existing token/project, so it's safe to run again to finish a partial setup.
Run `hermes photon status` to see what's configured.

## Credentials

Runtime SDK credentials live in `~/.hermes/.env` (the same place every other
channel keeps its token), and the adapter reads them from the environment:

```bash
PHOTON_PROJECT_ID=<spectrumProjectId>   # the SDK's projectId
PHOTON_PROJECT_SECRET=<projectSecret>
```

Management metadata lives in `~/.hermes/auth.json` under `credential_pool`:

```jsonc
{
  "credential_pool": {
    "photon": [
      { "access_token": "<device-bearer>", "issued_at": ... }
    ],
    "photon_project": [
      {
        "dashboard_project_id": "<dashboard id>",
        "spectrum_project_id": "<spectrumProjectId>",
        "project_secret": "<projectSecret>",
        "name": "Hermes Agent"
      }
    ]
  }
}
```

> **Note on ids.** A Photon project has two identifiers: the dashboard `id`
> (used for management API calls) and the `spectrumProjectId` (what the SDK
> authenticates with). `PHOTON_PROJECT_ID` is the **spectrum** id.

## Configuration knobs

All env vars are documented in `plugin.yaml`. The most important:

| Env var                   | Default                    | Meaning                              |
|---------------------------|----------------------------|--------------------------------------|
| `PHOTON_PROJECT_ID`       | from .env / auth.json      | Spectrum project id (SDK `projectId`)|
| `PHOTON_PROJECT_SECRET`   | from .env / auth.json      | Project secret                       |
| `PHOTON_SIDECAR_PORT`     | 8789                       | Loopback port for the sidecar        |
| `PHOTON_SIDECAR_AUTOSTART`| true                       | Spawn the sidecar on connect         |
| `PHOTON_DASHBOARD_HOST`   | https://app.photon.codes   | Dashboard API host                   |
| `PHOTON_SPECTRUM_HOST`    | https://spectrum.photon.codes | Spectrum API host                 |
| `PHOTON_HOME_CHANNEL`     | your number (set by setup) | Default space for cron delivery — a space id, or a bare E.164 number (resolved to a DM) |
| `PHOTON_ALLOWED_USERS`    | your number (set by setup) | Comma-separated E.164 allowlist      |
| `PHOTON_REQUIRE_MENTION`  | false                      | Gate group chats on a wake word      |
| `PHOTON_MAX_INLINE_ATTACHMENT_BYTES` | 20 MB           | Max inbound attachment size the sidecar reads & inlines |

## Attachments & limitations

- **Inbound attachments and voice notes are downloaded.** The sidecar reads
  the bytes (`content.read()`) and base64-inlines them on the NDJSON event; the
  adapter caches them to the shared media cache and populates `media_urls` /
  `media_types`, so the agent sees the real image/file or can transcribe the
  voice note — parity with the BlueBubbles iMessage channel. Media larger than
  `PHOTON_MAX_INLINE_ATTACHMENT_BYTES` (default 20 MB), or any byte read that
  fails, falls back to a text marker (`[Photon attachment received: …]` or
  `[Photon voice received: …]`) so the agent still knows something arrived.
- **Outbound attachments are supported.** Images, voice notes, video, and
  documents are sent via `space.send(attachment(...))` /
  `space.send(voice(...))` through the sidecar's `/send-attachment`
  endpoint; a caption is delivered as a separate text bubble after the media.
- **Reactions, message effects, polls** — supported by `spectrum-ts` but not
  yet exposed; the sidecar is the natural place to add them.

[photon]: https://photon.codes/

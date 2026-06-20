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
- **Outbound**: `send` / `send_typing` / reaction tapbacks are loopback POSTs
  to the sidecar (`/send`, `/send-attachment`, `/typing`, `/react`,
  `/unreact`), authenticated with a shared `X-Hermes-Sidecar-Token`.

## First-time setup

```bash
# One-shot setup: device login (opens browser) + project + user + sidecar deps
hermes photon setup --phone +15551234567

# Start the gateway
hermes gateway start
```

`hermes photon setup` does, in order:

1. **Device login** (RFC 8628, `client_id=photon-cli`) — opens
   `https://app.photon.codes/` for approval and stores the bearer token.
2. **Find or create** the `Hermes Agent` project on the Photon dashboard.
3. **Provision the project secret** — mint a fresh project secret (the
   dashboard reveals it only once) and persist it to `~/.hermes/.env` so the
   sidecar can authenticate `spectrum-ts`. Spectrum is always on, so there's no
   separate enable step.
4. **Register your phone number** as a Spectrum user (idempotent — skipped if
   a user with that number already exists).
5. **Print the assigned iMessage line** — the number you text to reach your
   agent.
6. **Install the sidecar deps** (`npm ci` — installs the committed lockfile
   verbatim, so every setup runs the exact `spectrum-ts` version this plugin
   was written against).

There is no separate `login` command; like every other Hermes channel,
onboarding goes through one setup surface. Re-running `setup` reuses an
existing token/project, so it's safe to run again to finish a partial setup.
Run `hermes photon status` to see what's configured.

## Credentials

Runtime SDK credentials live in `~/.hermes/.env` (the same place every other
channel keeps its token), and the adapter reads them from the environment:

```bash
PHOTON_PROJECT_ID=<projectId>   # the SDK's projectId (same as the dashboard project id)
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
        "dashboard_project_id": "<project id>",
        "spectrum_project_id": "<project id>",
        "project_secret": "<projectSecret>",
        "name": "Hermes Agent"
      }
    ]
  }
}
```

> **Note on ids.** A Photon project's dashboard id and its Spectrum project id
> are the same value, exposed as `PHOTON_PROJECT_ID`. The `dashboard_project_id`
> and `spectrum_project_id` keys in `auth.json` both hold that id.

## Configuration knobs

All env vars are documented in `plugin.yaml`. The most important:

| Env var                   | Default                    | Meaning                              |
|---------------------------|----------------------------|--------------------------------------|
| `PHOTON_PROJECT_ID`       | from .env / auth.json      | Spectrum project id (SDK `projectId`)|
| `PHOTON_PROJECT_SECRET`   | from .env / auth.json      | Project secret                       |
| `PHOTON_SIDECAR_PORT`     | 8789                       | Loopback port for the sidecar        |
| `PHOTON_SIDECAR_SEND_TIMEOUT` | 120                    | Loopback HTTP timeout (seconds) for `/send` and `/send-attachment` |
| `PHOTON_SIDECAR_AUTOSTART`| true                       | Spawn the sidecar on connect         |
| `PHOTON_DASHBOARD_HOST`   | https://app.photon.codes   | Dashboard API host                   |
| `PHOTON_SPECTRUM_HOST`    | https://spectrum.photon.codes | Spectrum API host                 |
| `PHOTON_HOME_CHANNEL`     | your number (set by setup) | Default space for cron delivery — a space id, or a bare E.164 number (resolved to a DM) |
| `PHOTON_ALLOWED_USERS`    | your number (set by setup) | Comma-separated E.164 allowlist      |
| `PHOTON_REQUIRE_MENTION`  | false                      | Gate group chats on a wake word      |
| `PHOTON_MAX_INLINE_ATTACHMENT_BYTES` | 20 MB           | Max inbound attachment size the sidecar reads & inlines |
| `PHOTON_TELEMETRY`        | false                      | Spectrum SDK telemetry — toggle with `hermes photon telemetry on\|off` (restart the gateway to apply) |
| `PHOTON_MARKDOWN`         | true                       | Send agent replies as markdown (iMessage renders natively). `false` strips formatting to plain text |
| `PHOTON_REACTIONS`        | false                      | Tapback 👀/👍/👎 as processing status; tapbacks on bot messages reach the agent as `reaction:added:<emoji>` |

## Attachments & limitations

- **Inbound attachments and voice notes are downloaded.** The sidecar reads
  the bytes (`content.read()`) and base64-inlines them on the NDJSON event; the
  adapter caches them to the shared media cache and populates `media_urls` /
  `media_types`, so the agent sees the real image/file or can transcribe the
  voice note — parity with the BlueBubbles iMessage channel. Mixed iMessage
  bubbles that contain both text and attachments are normalized as a grouped
  payload so the user's typed text is preserved alongside the cached media.
  Media larger than `PHOTON_MAX_INLINE_ATTACHMENT_BYTES` (default 20 MB), or
  any byte read that fails, falls back to a text marker (`[Photon attachment
  received: …]` or `[Photon voice received: …]`) so the agent still knows
  something arrived.
- **Outbound attachments are supported.** Images, voice notes, video, and
  documents are sent via `space.send(attachment(...))` /
  `space.send(voice(...))` through the sidecar's `/send-attachment`
  endpoint; a caption is delivered as a separate text bubble after the media.
- **Markdown is rendered.** Replies go out via spectrum-ts' `markdown()`
  builder; iMessage renders bold/italics/lists/code natively and other
  Spectrum platforms degrade to readable plain text. `PHOTON_MARKDOWN=false`
  reverts to stripped plain text.
- **Reactions (tapbacks) are supported** behind `PHOTON_REACTIONS` (default
  off): the adapter tapbacks 👀 while processing and swaps it for 👍/👎 on
  completion, and a user tapback on a bot-sent message is routed to the agent
  as a synthetic `reaction:added:<emoji>` event. Removal after a sidecar
  restart is best-effort — the live reaction handle is lost, so a stale
  tapback heals when the next reaction replaces it. Group spaces stay
  reachable across restarts via spectrum-ts v3's `space.get(id)`.
- **Message effects, polls** — supported by `spectrum-ts` but not yet
  exposed; the sidecar is the natural place to add them.

## Upgrading spectrum-ts

`spectrum-ts` is pinned to an **exact version** in `sidecar/package.json`
(no `^` range) and installed with `npm ci`, because the SDK ships breaking
majors (v2 removed `defineFusorPlatform`; v3 reworked space construction).
A floating range or `npm install spectrum-ts@latest` would let a breaking
release take down fresh setups silently. Upgrades are deliberate:

1. Read the [SDK release notes](https://github.com/photon-hq/spectrum-ts/releases)
   for every version between the current pin and the target.
2. Bump the exact pin in `sidecar/package.json`, then run `npm install`
   inside `sidecar/` to regenerate `package-lock.json`. Commit both.
3. Migrate `sidecar/index.mjs` against the new typings
   (`sidecar/node_modules/spectrum-ts/dist/*.d.ts` is the source of truth —
   the hosted docs can lag).
4. Run `pytest tests/plugins/platforms/photon/`.
5. Verify end-to-end: `hermes photon status`, a DM and a group roundtrip,
   and an agent reply into a group right after a gateway restart (exercises
   `space.get` rehydration).

[photon]: https://photon.codes/

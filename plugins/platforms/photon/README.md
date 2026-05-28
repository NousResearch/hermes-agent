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

One implementation detail matters for shared iMessage lines: webhook
events identify a conversation with a canonical Spectrum space id
like `any;-;+<phone>`, while the current `spectrum-ts`
`imessage(app).space(...)` helper resolves direct-message spaces by
recipient address (`+<phone>`). The sidecar therefore caches
send-capable `Space` objects from the inbound stream and, for uncached
shared-line DMs, strips the `any;-;` prefix before sending. This
bridges Photon's webhook shape to the SDK's outbound lookup shape
without changing the Python gateway contract.

## Quick setup

```bash
# Log in first. Quick setup intentionally does not start this flow for you.
hermes photon login

# Then run the all-in-one local setup. Replace the placeholder with
# your E.164 number: + followed by country code and number, no spaces.
hermes photon quick-setup --phone '+<country-code><number>'
```

`hermes setup gateway` runs the same guided Photon flow when you choose
Photon from the platform list. It will ask you to run
`hermes photon login` first if no Photon token is present. After login,
the quick setup path is idempotent: it reuses local credentials first,
then adopts a matching Photon dashboard project named `Hermes Agent` if
one exists, creates one if none exists, and stops if multiple matching
projects require an explicit selection.

After login, quick setup performs the same work as:

```bash
# 1. Reuse/adopt/create the Photon project, create the shared user,
#    and install the sidecar dependencies.
hermes photon setup --phone '+<country-code><number>'

# 2. Start the managed local webhook tunnel and register it with Photon.
hermes photon webhook tunnel start

# 3. Check the computed next step.
hermes photon status

# 4. Start the gateway in foreground QA mode.
hermes gateway run -v
```

If the gateway was already running when the webhook was registered,
restart it so the adapter loads `PHOTON_WEBHOOK_SECRET`:

```bash
hermes gateway restart
```

## Project management

Use these commands when quick setup stops because multiple compatible
Photon projects exist, or when you intentionally want to bind Hermes to
a specific project.

```bash
hermes photon projects list
hermes photon projects select <dashboard-or-spectrum-project-id>
hermes photon setup --new-project --phone '+<country-code><number>'
```

Use `--new-project` only when you intentionally want a separate Photon
dashboard project.

## Managed Cloudflare Quick Tunnel

Cloudflare Quick Tunnel is the default local development path. It is
temporary: the `trycloudflare.com` URL can change when the tunnel
restarts. The managed command starts `cloudflared`, captures the public
URL, registers the Photon webhook, and saves the local webhook secret.

```bash
hermes photon webhook tunnel start
hermes photon webhook tunnel status
hermes photon webhook tunnel logs
hermes photon webhook tunnel stop
```

The managed tunnel command can delete stale `trycloudflare.com` webhooks
it previously created; it leaves user-owned/manual webhook URLs alone.

## Manual webhook management

Use manual webhook registration for production, a named Cloudflare
Tunnel, or any other stable user-owned reverse proxy.

```bash
hermes photon webhook register https://YOUR-PUBLIC-URL/photon/webhook
hermes photon webhook list
hermes photon webhook delete <webhook-id>
```

Webhook registration is duplicate-aware: if the same URL is already
registered and `PHOTON_WEBHOOK_SECRET` is present,
`hermes photon webhook register ...` is a no-op. If the URL exists but
the local secret is missing, delete or recreate the webhook in the Photon
dashboard and save the new signing secret locally.

## Status and gateway runtime

`hermes photon status` is the fastest way to see the current readiness
state. The final row is a computed next step.

```bash
hermes photon status

hermes gateway run -v
hermes gateway restart
```

For always-on local use, install/start the launchd service instead:

```bash
hermes gateway install --force
hermes gateway start
```

## Detailed command reference

```bash
hermes photon login
hermes photon quick-setup --phone '+<country-code><number>'
hermes photon setup --phone '+<country-code><number>'
hermes photon setup --new-project --phone '+<country-code><number>'
hermes photon projects list
hermes photon projects select <dashboard-or-spectrum-project-id>
hermes photon install-sidecar
hermes photon webhook tunnel start
hermes photon webhook tunnel status
hermes photon webhook tunnel logs
hermes photon webhook tunnel stop
hermes photon webhook register https://YOUR-PUBLIC-URL/photon/webhook
hermes photon webhook list
hermes photon webhook delete <webhook-id>
hermes photon status
hermes gateway run -v
hermes gateway restart
hermes gateway install --force
hermes gateway start
```

## Credentials

The Photon dashboard token is stored in `~/.hermes/auth.json` under
`credential_pool`:

```jsonc
{
  "credential_pool": {
    "photon": [
      { "access_token": "<dashboard-bearer>", "issued_at": ... }
    ]
  }
}
```

The Spectrum project credentials used by the gateway are stored in
`~/.hermes/.env`:

```bash
PHOTON_PROJECT_ID=...
PHOTON_PROJECT_SECRET=...
```

The per-URL webhook signing secret is treated like an API key and
lives in `~/.hermes/.env` as `PHOTON_WEBHOOK_SECRET`.
The registered public webhook URL is stored as
`PHOTON_WEBHOOK_PUBLIC_URL` so `hermes photon status` can tell you
whether the next action is tunnel setup, gateway start, or gateway
restart.

## Configuration knobs

All env vars are documented in `plugin.yaml`. The most important are:

| Env var                  | Default            | Meaning                                 |
|--------------------------|--------------------|-----------------------------------------|
| `PHOTON_PROJECT_ID`      | (unset)            | Spectrum project ID                     |
| `PHOTON_PROJECT_SECRET`  | (unset)            | Spectrum project secret (HTTP Basic)    |
| `PHOTON_WEBHOOK_SECRET`  | (unset)            | Signing secret returned at register     |
| `PHOTON_WEBHOOK_PUBLIC_URL` | (unset)         | Public webhook URL registered with Photon |
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
- **Threaded reply metadata is not sent yet.** Hermes may pass a
  `replyTo` id to the sidecar, but the sidecar currently sends plain
  text with `space.send(text(...))`; true Spectrum replies need the
  SDK `reply(...)` content builder and the original message object.
- **Reactions, message effects, polls** — not exposed yet; the
  `spectrum-ts` SDK supports them, and the sidecar is the natural
  place to add them when the agent has reason to use them.

[photon]: https://photon.codes/

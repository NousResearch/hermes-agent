# Photon iMessage platform plugin

Photon lets you text your local Hermes Agent through iMessage and use
that thread as your main Hermes interface. Photon owns the iMessage
line and delivery layer; your computer runs Hermes, the webhook
receiver, and the local tunnel Photon uses to reach it.

For daily use, the tunnel and the Hermes gateway must stay running. If
either stops, incoming texts will not reach Hermes until both are back
up.

The free tier uses Photon's shared iMessage line pool (`type: shared`).
Use that unless you already pay for a dedicated Photon number.

## Quick setup

```bash
hermes photon login
hermes photon quick-setup --phone '+<country-code><number>'
hermes gateway run -v
```

Use your E.164 phone number: `+` plus country code and number, with no
spaces. Setup authorizes that phone automatically, so your first
iMessage goes straight to Hermes. Photon may assign a different shared
iMessage sending number for each phone you register; use the assigned
number shown for that user in the Photon dashboard. To add another
phone later:

```bash
hermes photon allow-phone '+<country-code><number>'
```

`quick-setup` creates or adopts the Photon project, creates the shared
iMessage user, installs the sidecar dependencies, starts/registers the
managed webhook tunnel, and writes credentials to `~/.hermes/.env`. It
is safe to run again.

`quick-setup` does not start the Hermes gateway. iMessage replies work
only while `hermes gateway run -v` is running, or while the gateway
service is installed and started. When the Photon adapter starts, it
also verifies the managed tunnel registration for that same profile and
cleans stale managed `trycloudflare.com` webhooks. If you use a custom
`HERMES_HOME`, export the same value before starting the gateway.

If the gateway was already running when the webhook was registered,
restart it so the adapter loads `PHOTON_WEBHOOK_SECRET`:

```bash
hermes gateway restart
```

`hermes setup gateway` runs the same guided Photon flow when you choose
Photon from the platform list.

## Run the gateway

After `quick-setup`, test the gateway in the foreground:

```bash
hermes gateway run -v
```

That is the gateway process. `-v` prints INFO-level startup and message
logs in your terminal; press `Ctrl+C` to stop it.

For always-on use, install the gateway service once and start it in the
background:

```bash
hermes gateway install --force
hermes gateway start
```

`gateway start` controls the installed system service. Do not run it
and `gateway run -v` at the same time.

Cloudflare Quick Tunnel URLs can change when the tunnel restarts. For
local Quick Tunnel setups, the Photon adapter starts or reuses the
managed tunnel when the gateway connects. If you need to repair or
inspect the tunnel manually, use:

```bash
hermes photon webhook tunnel start
```

`hermes photon status` shows the current readiness state and the next
command to run.

## Troubleshooting runtime logs

On gateway startup, Photon logs the active Hermes home. It must match
the profile where you ran `hermes photon quick-setup`. If quick setup
wrote credentials under a custom `HERMES_HOME`, start the gateway with
that same `HERMES_HOME` or the gateway will read a different `.env`.

If you see `photon-sidecar: observed SDK inbound ...` but do not see
`[photon] webhook delivery received`, the sidecar has observed the
message through the SDK stream but Hermes has not received the signed
webhook it uses for inbound dispatch. Run `hermes photon status` in the
same profile, then check that the current public webhook URL is
registered and reachable.

## How it works

```
iMessage phone
    |
    v
Photon Spectrum cloud
    |
    | signed webhook through local tunnel
    v
Hermes gateway on your computer
    |
    | outbound replies through spectrum-ts sidecar
    v
Photon Spectrum cloud -> iMessage phone
```

- Inbound messages arrive as signed Photon webhooks. Hermes verifies
  `X-Spectrum-Signature` and dedupes on `message.id`.
- Outbound replies go through the Node sidecar because Photon does not
  expose a public HTTP send-message endpoint yet.
- Shared-line conversations can arrive with Spectrum space IDs like
  `any;-;+<phone>`. The sidecar caches send-capable spaces and falls
  back to the recipient address when needed.

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
restarts. The managed command installs or updates a profile-local
`cloudflared` binary if one is not already on PATH, starts it, captures
the public URL, registers the Photon webhook, and saves the local webhook
secret.

```bash
hermes photon webhook tunnel start
hermes photon webhook tunnel status
hermes photon webhook tunnel logs
hermes photon webhook tunnel stop
```

The managed tunnel command can delete stale `trycloudflare.com` webhooks
it previously created; it leaves user-owned/manual webhook URLs alone.
The gateway performs the same cleanup when Photon connects, so the
active gateway profile owns the current managed webhook registration.

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

## Status

`hermes photon status` is the fastest way to see the current readiness
state. The final row is a computed next step.

```bash
hermes photon status
```

## Detailed command reference

```bash
hermes photon login
hermes photon quick-setup --phone '+<country-code><number>'
hermes photon setup --phone '+<country-code><number>'
hermes photon setup --new-project --phone '+<country-code><number>'
hermes photon allow-phone '+<country-code><number>'
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

Photon secrets are stored in `~/.hermes/.env`:

```bash
PHOTON_DASHBOARD_TOKEN=...
PHOTON_PROJECT_ID=...
PHOTON_PROJECT_SECRET=...
```

The dashboard token is used only by `hermes photon` management commands
such as project listing/creation. The Spectrum project credentials are
used by the gateway and sidecar at runtime. The per-URL webhook signing
secret is treated like an API key and lives in `.env` as
`PHOTON_WEBHOOK_SECRET`.
The registered public webhook URL is stored as
`PHOTON_WEBHOOK_PUBLIC_URL` so `hermes photon status` can tell you
whether the next action is tunnel setup, gateway start, or gateway
restart.

## Configuration knobs

All env vars are documented in `plugin.yaml`. The most important are:

| Env var                  | Default            | Meaning                                 |
|--------------------------|--------------------|-----------------------------------------|
| `PHOTON_DASHBOARD_TOKEN` | (unset)            | Dashboard token set by `hermes photon login` |
| `PHOTON_PROJECT_ID`      | (unset)            | Spectrum project ID                     |
| `PHOTON_PROJECT_SECRET`  | (unset)            | Spectrum project secret (HTTP Basic)    |
| `PHOTON_WEBHOOK_SECRET`  | (unset)            | Signing secret returned at register     |
| `PHOTON_WEBHOOK_PUBLIC_URL` | (unset)         | Public webhook URL registered with Photon |
| `PHOTON_WEBHOOK_TUNNEL_AUTOSTART` | true      | Gateway starts/registers managed tunnel for trycloudflare URLs |
| `PHOTON_WEBHOOK_TUNNEL_STOP_ON_DISCONNECT` | true | Stop managed tunnel on gateway shutdown |
| `PHOTON_WEBHOOK_PORT`    | 8788               | Local port for the aiohttp listener     |
| `PHOTON_WEBHOOK_PATH`    | /photon/webhook    | Path under which the listener mounts    |
| `PHOTON_SIDECAR_PORT`    | 8789               | Loopback port for sidecar control      |
| `PHOTON_HOME_CHANNEL`    | (unset)            | Default space ID for cron delivery     |
| `PHOTON_ALLOWED_USERS`   | (unset)            | Comma-separated E.164 allowlist; setup seeds `--phone` |

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

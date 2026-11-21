---
name: hermes-desktop-install
description: Publish the Hermes Desktop workspace on a headless Linux host in the browser, with authenticated HTTPS (Caddy TLS + Nous Portal OAuth).
version: 0.3.0
author: Welliv (welliv), Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [desktop, webapp, caddy, headless, install, https, portal-oauth, autonomous-ai-agents]
    category: autonomous-ai-agents
    related_skills: [hermes-agent]
when_to_use:
  - Put Hermes Desktop on this server / VM / VPS
  - A headless host needs the Desktop workspace in the browser
  - Replacing an ad-hoc `hermes serve` dashboard with the real Desktop surface
when_not_to_use:
  - A real desktop OS (`hermes desktop` runs natively there)
  - The admin dashboard (`hermes dashboard`, a different surface)
dependencies: []
---

# Hermes Desktop Install Skill

Publishes the **Desktop workspace in the browser** on a Linux host that has no
display: the gateway serves the same `apps/desktop` renderer that Electron ships,
and Caddy puts an authenticated HTTPS hostname in front of it.

Two legacy designs this version deliberately rejects:

1. **No `desk-webapp.service` that owns the listener.** `hermes webapp` has no
   `--host/--port` and is not itself a server. The **gateway** owns the
   listener (`gateway/webapp_listener.py`, configured by `webapp.*` in
   `config.yaml`). A separate unit that binds the same port fights the gateway,
   or wins and then serves stale code after the next gateway restart.
2. **The auth gate is driven by `webapp.public_url`, not by the bind address.**
   Behind a reverse proxy you keep the loopback host and declare the public URL;
   the gateway then requires sign-in, even though it only listens on the loopback.

## Result

A secure, browser-accessible Hermes Desktop at `<DESK_HOST>`: the full Desktop
workspace (sessions, files, Git, terminal, cron, Bots screen) from any device,
sign-in via Nous Portal OAuth, and a pass/fail verifier proving the install is
healthy.

## When to use

- "Put Hermes Desktop on this server / VM / VPS"
- A headless host needs the Desktop workspace in the browser
- Replacing an ad-hoc `hermes serve` dashboard with the real Desktop surface

Not for: a real desktop OS (`hermes desktop` runs natively there), and not the
admin dashboard (`hermes dashboard`, a different surface).

## Prerequisites

- Hermes installed and working (`hermes doctor`), with `hermes webapp` present.
  Check first: `hermes webapp --help` (must list `--build-only / --status / --sign-out`).
- **Version matrix** (validated on this host; minimum tested):

  | Requirement | Minimum tested | Notes |
  |---|---|---|
  | Hermes gateway | `2026.9.24` (v0.23.1, commit `3b4215785882`) | has `hermes webapp` and `hermes_cli.webapp_access`. Verify, don't guess. |
  | Caddy | `2.6.x` (2.6.2 verified) | `caddy version` |
  | Node | any with `npm` | first renderer bundle build |

- A DNS **A record resolving** to this host before any certificate is requested
  (the A record must point at the public IP).
- Caddy installed: `apt-get install -y caddy` on Debian/Ubuntu.
- Ports **80 and 443** reachable from the internet (cloud security group *and*
  host firewall) — ACME cannot issue a certificate otherwise.
- Node + outbound network for the first bundle build.

Resolve the home once and reuse it: `HERMES_HOME=${HERMES_HOME:-$HOME/.hermes}`.

**Bundle note:** all supporting artifacts live inside the skill directory —
`scripts/install-desktop.sh`, `scripts/verify-desktop.sh`, `scripts/selftest.sh`,
`templates/`, and `references/`.

## Recommended flow

The default path is the **idempotent script**, which performs steps 2-6 and ends
by running the verifier:

```bash
scripts/install-desktop.sh --desk-host <DESK_HOST> [--login <USER>]
```

Flags:

- `--password` (with `DASH_PASSWORD` in the environment, never argv): password
  provider instead of Portal OAuth.
- `--bind 0.0.0.0` with `--trusted-proxy <CIDR>` (repeatable): when a proxy or
  tunnel runs in another container. Loopback stays the default.
- `--no-caddy`: you front the port yourself (external proxy, Docker).
- `--dry-run`: print every plan and change nothing. `--check`: preflight only.
- `--restart`: restart the gateway from the script; omit it when the script runs
  inside the agent's own session, where a restart would kill the caller.

**Manual alternative** (only if you are overriding the script): follow the
numbered procedure below, substituting the templates in `templates/`.

## Procedure

### 1. Confirm the Webapp is installed

`hermes webapp --help` and `ls $HERMES_HOME/hermes-agent/apps/desktop`. Done:
both succeed. If `webapp` is missing, stop — see *Version requirements*.

### 2. Choose a sign-in provider

The loopback listener engages the gate, and a gated listener with **no provider
refuses to serve** (`hermes webapp --status` explains why). Choose one:

- **Portal OAuth (recommended for anything internet-facing)** — use the Webapp's
  own client, the same code the native app's Settings → Web Access → Start runs.
  Requires being signed in (`hermes portal`). Do **not** use `hermes dashboard
  register` here: it registers a *dashboard* OAuth client and writes an OAuth
  client id (plus a public URL, only when `--redirect-uri` is given) to
  `~/.hermes/.env`. It never registers a Webapp-only client, so the Webapp's
  sign-in gate would have no Nous client of its own.

  The installer uses the Webapp's client and only falls back to `dashboard
  register` (with a warning) on builds lacking `hermes_cli.webapp_access`.

- **Username + password (trusted networks only)** — three keys go in
  `$HERMES_HOME/.env` (mode 0600, never in `config.yaml`): the login, a
  scrypt-hashed password, and a session secret. Derivation of the hash is in
  `references/password-provider.md`; the installer computes it with the
  install venv python so `plugins.*` imports resolve. Completion: `hermes
  plugins list | grep basic` shows the provider `enabled`, and the keys are
  present.

### 3. Point the Webapp at the public URL

Four keys via the CLI (never hand-edit `config.yaml`) — the installer performs
these, defaulting to the loopback host and the agent's own listener port:

```bash
hermes config set webapp.enabled true
hermes config set webapp.host <LOOPBACK_HOST>      # loopback; Caddy fronts it
hermes config set webapp.port <WEBAPP_PORT>
hermes config set webapp.public_url https://<DESK_HOST>
```

Keep the loopback behind a proxy: `webapp.public_url` alone engages the gate,
and never expose the port from the router. If the proxy is not on this host, the
edge shows a 502 (loopback unreachable); bind wider, declare the proxy's network
in `dashboard.trusted_proxies`, or the callback is built as `http://` and the
session cookies are not `Secure`:

```bash
hermes config set webapp.host 0.0.0.0
```

Completion: `hermes config get webapp` echoes all four.

### 4. Build the renderer bundle

```bash
hermes webapp --build-only
ls "$HERMES_HOME/hermes-agent/apps/desktop/dist-webapp/index.html"
```

Completion: the build receipt and `index.html` exist (~47 MB of assets). The
gateway serves the bundle without a restart once it is on disk, but it *starts
the listener* only on a boot that read `webapp.enabled`.

### 5. Add the Caddy site

Render the template `templates/Caddyfile.site` (placeholders `<DESK_HOST>`,
`<UPSTREAM>`, `<PORT>`):

```caddy
<DESK_HOST> {
    reverse_proxy <UPSTREAM>:<PORT>
    encode zstd gzip
}
```

Back up first, then `caddy validate --config /etc/caddy/Caddyfile`, then
`systemctl reload caddy`. Remove any default `:80 { ... }` site block — it
shadows the ACME HTTP-01 challenge for the hostname.

Completion: validate prints `Valid configuration`, and `ss -ltnp` shows Caddy on
`:80` and `:443`.

### 6. Restart the gateway

`webapp.*` and the sign-in provider are read at startup only:

```bash
hermes gateway restart          # or send /restart in a Hermes chat
```

Completion: `ss -ltnp` shows the **gateway's** PID on the Webapp port.

### 7. Verify

`scripts/verify-desktop.sh <DESK_HOST> [USER PASS]`. Completion: every line
`PASS`, exit 0. Fix, do not explain away, any `FAIL` (a `000` in the first
~60s is usually pending issuance — sleep and retry).

### 8. Optional — Bot Screen

A headless host gives each bot its own Xfce desktop that `computer_use` and headed
browsing drive, viewable at Bots → Screen:

```bash
hermes computer-use screen install   # TigerVNC Xvnc + Xfce core (~930 MB)
hermes computer-use install          # cua-driver
hermes computer-use screen start
hermes computer-use doctor           # ax_capability + screen_capture green
```

There is **no autostart unit** by design: the screen is on-demand (`screen
start`, or Start in the pane), and Hermes owns its control lease. The dock's
browser uses Hermes' managed Chromium, launched with the screen's
`XAUTHORITY`; do not build an Xvfb/x11vnc/noVNC stack instead.

Completion: `hermes computer-use screen status` shows `running` on a display
(`:20`) with an RFB socket, and `doctor` is green.

## Acceptance criteria (Done checklist)

The installation counts only when all of the following hold:

1. `hermes webapp --status` → a serving line whose PID is the gateway's
   `MainPID`.
2. `ss -ltnp` shows Caddy on `:80` and `:443`, and the loopback Webapp port
   owned by the gateway.
3. `GET https://<DESK_HOST>/` → `30x` to `/login` (the auth gate).
4. `/login` renders a sign-in form (`200`, contains `Sign in`).
5. Wrong password → `401`; correct password → `200` + session cookie.
6. `GET /api/health` with that cookie → `200` + a `version`.
7. TLS certificate subject matches `<DESK_HOST>` and is not expired.
8. `apps/desktop/dist-webapp/index.html` exists (renderer artifact).

## References (bundled with this skill)

- `scripts/install-desktop.sh` — idempotent installer (fast path)
- `scripts/verify-desktop.sh` — live deployment verifier
- `scripts/selftest.sh` — offline self-test for the scripts
- `templates/Caddyfile.site` — Caddy site template
- `references/` — version matrix, security architecture, password provider,
  network configuration

## Pitfalls

- The old unit-based design is wrong now. Running
  `hermes webapp --host ... --port ...` fails (no such flags). A unit that binds
  the port behind the gateway's back either cannot bind, or serves stale code
  after the next restart. The gateway is the server.
- `webapp.*` is read at startup. Changing URL, port or auth env does nothing
  until the gateway restarts, and the old password keeps working until then.
- No provider, no service. With `webapp.public_url` set and no usable provider,
  the gateway still serves native clients but not the Webapp; `hermes webapp
  --status` explains why. Check `hermes plugins list | grep -E 'basic\|nous'`.
- A `:80` site block breaks issuance. Caddy's stock `:80 { root * ... }` answers
  for every hostname and can shadow the HTTP-01 challenge; delete it.
- Caddy can come up HTTP-only after a cold start. `systemctl reload caddy`
  engages the TLS listener and triggers issuance.
- A VPS often cannot reach its own public IP. Verify with
  `curl --resolve <DESK_HOST>:443:<PUBLIC_IP> https://<DESK_HOST>/` before
  concluding the proxy is broken.
- HTTPS is not optional. Clipboard write, microphone capture, notifications and
  the Bot Screen viewer require a secure context; plain HTTP silently disables
  them.
- Webapp access is root-equivalent. A signed-in user runs commands, touches
  host files and Git, and reads the account's credentials. Prefer OAuth for
  public exposure; never publish the loopback port directly.
- `hermes dashboard register` is not the Webapp's registration path (the
  installer uses `webapp_access` and only falls back to it). It registers a
  *dashboard* OAuth client and writes a dashboard OAuth client id (plus a public
  URL, only with `--redirect-uri`) to `~/.hermes/.env`, and never a Webapp-only
  client. Desktop's Settings → Web Access does the Webapp registration correctly
  through `hermes_cli/webapp_access.py` (`plan` → `register` → `serve`), keeping
  a Webapp-only client in `webapp/sign-in-registration.json` and leaving `.env`
  alone. Use `dashboard register` only when the Webapp is the sole surface.
- Docker / proxy hosts need the bind and the proxy trust set. The Web Access
  path pins the loopback; a proxy in another container then means connection
  refused (502 at the edge). Use `webapp.host: 0.0.0.0` and list the proxy's
  network in `dashboard.trusted_proxies`, otherwise the auth callback is built
  as `http://` and the cookies are not `Secure`. Loopback proxies (Caddy on the
  same host) are trusted already and need neither.
- Old chats need adopting before they can be continued from the browser:
  `hermes sessions adopt` on the host.
- `hermes update` on `main` is frequent. Keep a copy of `config.yaml`, `.env`
  and the current commit before updating.

## Before you publish (security checklist)

- Ensure grouping/allowlists so the gateway does not accept arbitrary DMs
  (e.g. `TELEGRAM_ALLOWED_USERS` / `group_policy` and any channel-level
  allowlists), and the webapp sign-in is the only entry path.
- Never publish the loopback webapp port directly; always behind Caddy/HTTPS.
- Adopt old chats before exposing the host: `hermes sessions adopt`.
- Keep a copy of `config.yaml`, `.env` and the current commit before `hermes
  update`.
- Treat `webapp.*` as startup-only — restart the gateway after changing URL,
  port or auth providers.

## Quick reference

```bash
hermes webapp --help                                  # does this build have it?
hermes config set webapp.enabled true                 # + host / port / public_url
hermes webapp --build-only                            # build the renderer once
caddy validate --config /etc/caddy/Caddyfile && systemctl reload caddy
hermes gateway restart                                # settings are startup-only
hermes webapp --status                                # Serving at the loopback listener port
scripts/verify-desktop.sh <host> [user pass]
```

## Security note for submitters

This skill performs real system administration: it writes a local password hash
and secrets into `$HERMES_HOME/.env`, binds a loopback webapp listener, and
configures a reverse proxy. These actions are the legitimate function of the
skill, and they are gated by sign-in and HTTPS. This package is the official
Nous optional-skills submission; the underlying installer behavior is the same
as the community GitHub-tap version (`hermes skills install
welliv/hermes-desktop-install/skills/autonomous-ai-agents/hermes-desktop-install`).

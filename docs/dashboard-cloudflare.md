# Hermes Dashboard via Cloudflare

## Current Endpoint

The Hermes dashboard is exposed at:

```text
https://hermes.yallaplay.com/hermes/
```

The experimental `nesquena/hermes-webui` is exposed at:

```text
https://hermes.yallaplay.com/webui/
```

The trusted-user simplified WebUI is exposed at:

```text
https://hermes.yallaplay.com/ui/
```

Cloudflare terminates HTTPS. Locally, traffic flows through a user-owned nginx
reverse proxy that forwards plain HTTP to the Hermes dashboard and preserves
WebSocket upgrades for chat, events, and PTY streams.

```text
Cloudflare Tunnel -> 127.0.0.1:9120 nginx -> 127.0.0.1:9119 Hermes dashboard (/hermes/)
                                             -> 127.0.0.1:9121 hermes-webui (/webui/)
                                             -> 127.0.0.1:9122 hermes-webui-public (/ui/)
```

## Running Processes

Three local processes are needed:

1. Hermes dashboard on localhost:

```bash
claudio-lab dashboard --host 127.0.0.1 --port 9119 --no-open --isolated --skip-build
```

2. Local nginx reverse proxy on localhost:

```bash
.local/nginx/sbin/nginx -p "$PWD/.local/nginx" -c hermes.conf
```

Reload nginx after config changes:

```bash
.local/nginx/sbin/nginx -p "$PWD/.local/nginx" -c hermes.conf -s reload
```

Stop nginx:

```bash
.local/nginx/sbin/nginx -p "$PWD/.local/nginx" -c hermes.conf -s quit
```

3. Named Cloudflare tunnel:

```bash
cloudflared tunnel \
  --config .local/cloudflared/config.yml \
  run
```

4. Experimental Hermes WebUI on localhost:

```bash
systemctl --user start hermes-webui
```

The source checkout is `.local/hermes-webui`, cloned from
`https://github.com/YallaPlay/hermes-webui` on branch `yallaplay/main`, with
`https://github.com/nesquena/hermes-webui` as the `upstream` remote. It runs with
`HERMES_HOME=/home/ubuntu/.hermes/profiles/claudio-lab`, stores WebUI-specific
state under `/home/ubuntu/.hermes/profiles/claudio-lab/webui-nesquena`, and
binds only to `127.0.0.1:9121`.

5. Trusted-user simplified Hermes WebUI on localhost:

```bash
systemctl --user start hermes-webui-public
```

This uses the same `.local/hermes-webui` checkout, a separate state directory at
`/home/ubuntu/.hermes/profiles/claudio-lab/webui-nesquena-public`, and
`HERMES_WEBUI_SIMPLE_UI=1`. It binds only to `127.0.0.1:9122` and is mounted at
`/ui/`. This is UI hiding for trusted users, not a server-side API security
boundary.

## nginx Config

The active local config is `.local/nginx/hermes.conf`. It intentionally binds
only to `127.0.0.1:9120`; Cloudflare is the public edge. `/hermes/` proxies to
the built-in Hermes dashboard. `/webui/` strips that prefix and proxies to the
admin WebUI on `127.0.0.1:9121`. `/ui/` strips that prefix and proxies to the
simplified trusted-user WebUI on `127.0.0.1:9122`. The bundled nginx was compiled
without `http_rewrite`, so slashless `/`, `/hermes`, `/webui`, and `/ui` are
served tiny HTML meta-refresh pages from `.local/nginx/html/` instead of nginx
`return 302` redirects.

Important proxy settings:

```nginx
proxy_http_version 1.1;
proxy_set_header Host 127.0.0.1:9119;
proxy_set_header Origin http://127.0.0.1:9119;
proxy_set_header X-Forwarded-Prefix /hermes;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection $connection_upgrade;
proxy_read_timeout 86400s;
proxy_send_timeout 86400s;
```

`Host` is pinned to `127.0.0.1:9119` because Hermes rejects dashboard requests
whose host does not match the bind host. `Origin` is also normalized to the local
Hermes origin because browser WebSocket requests from `https://hermes.yallaplay.com`
otherwise receive `403` from Hermes. `X-Forwarded-Prefix: /hermes` is required
for the dashboard SPA to inject `window.__HERMES_BASE_PATH__="/hermes"`; without
it dashboard navigation links such as Chat route to `/chat` instead of
`/hermes/chat`.

## Cloudflare Resources

- Tunnel name: `claudio-hermes-dashboard`.
- DNS hostname: `hermes.yallaplay.com`.
- Tunnel ID: `2eb94bf8-b7fb-4f8f-b058-d32c3632bdec`.
- Cloudflared local config and credentials: `.local/cloudflared/`.

The tunnel was created via Cloudflare API using the existing
`CLOUDFLARE_API_TOKEN` from the legacy Claudio repo's `vars.toml`. The token
itself is not copied into this repo.

## Cloudflare Access

The public hostname is protected by Cloudflare Zero Trust Access before traffic
reaches the tunnel or local nginx.

- Access organization: `YallaPlay`.
- Access auth domain: `yallaplay.cloudflareaccess.com`.
- Access application: `Hermes Dashboard`.
- Application domain: `hermes.yallaplay.com`.
- Application type: `self_hosted`.
- Session duration: `168h` (7 days).
- Identity provider: Cloudflare `One-Time PIN`.
- Google SSO was tested with the shared Google OAuth client from the legacy
  Claudio repo's `vars.toml`, but that client is a Google OAuth Desktop app and
  fails Google's web OAuth redirect validation. Keep it detached until a proper
  Web application OAuth client exists.
- Allow policy: `YallaPlay team`.
- Allowed users: `*@yallaplay.com` and `israel.lot@gmail.com`.

To enable Google SSO, create a Google OAuth **Web application** client with this
authorized redirect URI, then update the Cloudflare Access Google IdP with that
client ID and secret:

```text
https://yallaplay.cloudflareaccess.com/cdn-cgi/access/callback
```

Unauthenticated requests should receive a Cloudflare `302` to
`yallaplay.cloudflareaccess.com` and should not increment local nginx access
logs.

## Safety Notes

Hermes dashboard HTML includes a live session token for the local dashboard
session. Keep Cloudflare Access enabled before sharing the URL. Local nginx does
not perform authentication; it only handles reverse proxying and WebSocket header
normalization.

## Smoke Tests

Local dashboard HTML should return `200`:

```bash
curl -sS -o /tmp/hermes-nginx.html -w '%{http_code}\n' http://127.0.0.1:9120/
```

Public unauthenticated access should be blocked by Cloudflare Access with a
`302` redirect, not served by local nginx:

```bash
before=$(wc -l < .local/nginx/logs/access.log)
curl -sS -D /tmp/hermes-access.headers -o /tmp/hermes-access.html -w '%{http_code}\n' https://hermes.yallaplay.com/
after=$(wc -l < .local/nginx/logs/access.log)
echo "nginx log delta: $((after-before))"
rg -i 'location:|cf-ray|set-cookie' /tmp/hermes-access.headers
```

API status should return `200` locally. Public `/api/status` should require an
Access-authenticated browser session:

```bash
curl -sS http://127.0.0.1:9120/api/status
```

After signing in through Cloudflare Access in a browser, WebSocket upgrades
should return `101` in nginx access logs for `/api/ws`, `/api/events`, and
`/api/pty`.

For the current setup, open `https://hermes.yallaplay.com` in a fresh browser
session and use One-Time PIN. Google should remain detached until the OAuth
client is replaced with a Web application client.

## Always-On Services

The pilot runs under user-level systemd units because this host blocks `sudo`
service installation with `no_new_privileges`. User lingering is enabled, so the
services are started by the `ubuntu` user manager across logins/boot.

- `hermes-dashboard.service` runs `claudio-lab --yolo dashboard` on
  `127.0.0.1:9119`.
- `hermes-webui.service` runs `nesquena/hermes-webui` on `127.0.0.1:9121`.
- `hermes-webui-public.service` runs the trusted-user simplified WebUI on
  `127.0.0.1:9122`.
- `hermes-gateway.service` runs `claudio-lab --yolo gateway run --replace
  --accept-hooks`.
- `hermes-nginx.service` runs the local nginx proxy on `127.0.0.1:9120`.
- `hermes-cloudflared.service` runs the named Cloudflare tunnel.

Manage services with:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
systemctl --user status hermes-dashboard hermes-webui hermes-webui-public hermes-gateway hermes-nginx hermes-cloudflared
systemctl --user restart hermes-dashboard hermes-webui hermes-webui-public hermes-gateway hermes-nginx hermes-cloudflared
journalctl --user -u hermes-dashboard -f
```

The unit source files are tracked in `scripts/systemd/`; the active user units
live in `~/.config/systemd/user/`.

## Rollback

Stop exposure immediately by stopping the `cloudflared tunnel run` process.

Stop local dashboard access by stopping nginx and the Hermes dashboard process.

To remove the public DNS/tunnel permanently, delete the `hermes.yallaplay.com`
CNAME and the `claudio-hermes-dashboard` tunnel in Cloudflare.

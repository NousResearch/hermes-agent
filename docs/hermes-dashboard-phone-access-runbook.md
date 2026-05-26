# Hermes Dashboard Phone Access Runbook

## Scope

This runbook covers the authenticated phone-accessible Hermes dashboard for the default `/home/oscar/.hermes` runtime. The public listener is an HTTPS reverse proxy. The Hermes dashboard itself stays bound to loopback.

## Runtime

- Local dashboard service: `hermes-dashboard.service`
- Public auth proxy service: `hermes-dashboard-auth-proxy.service`
- Browser-trusted quick tunnel service: `hermes-dashboard-cloudflared.service`
- Local dashboard upstream: `http://127.0.0.1:9119`
- Direct phone URL: `https://5.161.56.251:9443` (self-signed certificate warning expected)
- Browser-trusted phone URL: read the current `trycloudflare.com` URL from `/home/oscar/.hermes/cloudflared/dashboard-tunnel.log`
- Auth model: browser Basic auth at the proxy, then Hermes dashboard session-token protection for dashboard API routes
- Username: `oscar`
- Password file on host: `/home/oscar/.hermes/dashboard-access/password`
- TLS certificate: self-signed certificate with SANs for `5.161.56.251`, `127.0.0.1`, and `localhost`

The password is intentionally not stored in this runbook, Kanban comments, dashboard config, or logs.

## Start

```bash
systemctl --user start hermes-dashboard.service
systemctl --user start hermes-dashboard-auth-proxy.service
systemctl --user start hermes-dashboard-cloudflared.service
```

## Stop

```bash
systemctl --user stop hermes-dashboard-auth-proxy.service
systemctl --user stop hermes-dashboard-cloudflared.service
systemctl --user stop hermes-dashboard.service
```

## Restart

```bash
systemctl --user restart hermes-dashboard.service
systemctl --user restart hermes-dashboard-auth-proxy.service
systemctl --user restart hermes-dashboard-cloudflared.service
```

## Status

```bash
systemctl --user status hermes-dashboard.service --no-pager
systemctl --user status hermes-dashboard-auth-proxy.service --no-pager
systemctl --user status hermes-dashboard-cloudflared.service --no-pager
```

## Health Checks

Unauthenticated requests must fail:

```bash
curl -k -I https://5.161.56.251:9443/
```

Expected: `401 Unauthorized`.

Authenticated dashboard status must work:

```bash
curl -k -u "oscar:$(cat /home/oscar/.hermes/dashboard-access/password)" \
  https://5.161.56.251:9443/api/status
```

Expected: `200 OK` JSON response.

Kanban plugin discovery must include the bundled Kanban tab:

```bash
curl -k -u "oscar:$(cat /home/oscar/.hermes/dashboard-access/password)" \
  https://5.161.56.251:9443/api/dashboard/plugins
```

Expected: an entry named `kanban` with tab path `/kanban`.

Kanban board API must load the Hermes Agent dev board:

```bash
curl -k -u "oscar:$(cat /home/oscar/.hermes/dashboard-access/password)" \
  "https://5.161.56.251:9443/api/plugins/kanban/board?board=hermes-agent-dev"
```

Expected: `200 OK` JSON with Kanban columns and tasks.

To get the browser-trusted URL:

```bash
grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' /home/oscar/.hermes/cloudflared/dashboard-tunnel.log | tail -1
```

Quick tunnels do not require a Cloudflare login, but the hostname can change if `hermes-dashboard-cloudflared.service` restarts. Use a named Cloudflare Tunnel on a real domain if a stable no-warning URL is needed.

## Rollback

Stop and disable the phone-facing proxy first:

```bash
systemctl --user disable --now hermes-dashboard-auth-proxy.service
systemctl --user disable --now hermes-dashboard-cloudflared.service
```

If needed, stop the local dashboard too:

```bash
systemctl --user disable --now hermes-dashboard.service
```

The dashboard remains available locally only if restarted with:

```bash
HERMES_HOME=/home/oscar/.hermes /home/oscar/.hermes/runtime/hermes-agent/.venv/bin/python -m hermes_cli.main dashboard --port 9119 --host 127.0.0.1 --no-open
```

## Security Notes

- Do not run `hermes dashboard --host 0.0.0.0 --insecure` for phone access.
- Do not put the password in Kanban comments, docs, dashboard config, or service logs.
- Rotate the password by replacing the password file and restarting `hermes-dashboard-auth-proxy.service`.
- The Chat/TUI tab is not enabled by these services.

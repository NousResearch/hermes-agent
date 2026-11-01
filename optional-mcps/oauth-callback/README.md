# Shared OAuth callback catcher

Not an MCP server (no `manifest.yaml` here — never appears in `hermes mcp
catalog`). A tiny shared HTTP service that several connectors' browser-based
OAuth flows redirect to, for providers (like Avito) that require the
registered Redirect URI to be a real, owned-domain HTTPS endpoint rather
than accepting a bare "show the code on a confirmation page" fallback or a
third-party catcher (webhook.site was tried first and rejected by Avito's
app-registration form: "domain must belong to you").

## Live setup (this install)

- **Domain**: `cb.mstylehome.ru` (zone `mstylehome.ru` in this account's
  Cloudflare).
- **Tunnel**: named Cloudflare Tunnel `mstyle-oauth-callback`
  (id `62a91058-ca25-45ad-b707-55e00c4ef32b`), DNS CNAME
  `cb.mstylehome.ru` → `<tunnel-id>.cfargotunnel.com` (proxied), ingress
  rule `cb.mstylehome.ru` → `http://localhost:8743`. Created via the
  Cloudflare API (`cloudflare_mcp` connector) — tunnel credentials/token
  are NOT committed here (secret); they live only in
  `~/.hermes/mcp-servers/avito/tunnel/tunnel_token.txt` on this host.
- **Local server**: this script, listening on `127.0.0.1:8743`.

Both `cloudflared tunnel run --token <token>` and `callback_server.py` are
separate long-lived background processes — **not** managed by any MCP
server, so they don't restart automatically with Hermes. After a host
reboot, restart both:

```bash
python3 ~/.hermes/mcp-servers/oauth-callback/callback_server.py &
cloudflared tunnel run --token "$(cat ~/.hermes/mcp-servers/avito/tunnel/tunnel_token.txt)" &
```

(Or set them up as systemd services / a Hermes cronjob health-check if this
needs to survive unattended for a long time — not done yet.)

## Adding another connector's callback path

Each connector gets its own path under the same domain instead of a new
subdomain per service (e.g. `/avito`, `/some-other-service`). The server
writes each incoming request's query params to
`~/.hermes/mcp-tokens/oauth-callbacks/<path>.json` — a connector's own
`*_login_wait_callback` tool polls its own file (see
`optional-mcps/avito/server/avito_mcp_server.py` for the pattern). No code
change needed in `callback_server.py` itself for a new path; just register
`https://cb.mstylehome.ru/<new-path>` as that service's Redirect URI and
write the matching wait-and-finish tool in its own connector.

## Network note

Outbound TCP/443 from this specific host is filtered (only QUIC/UDP gets
through cleanly to Cloudflare's edge, confirmed via `cloudflared`'s own
connectivity precheck). This does **not** affect end users opening the
authorize URL in their own browser on a normal network — only matters if
testing connectivity to `cb.mstylehome.ru` *from this host itself*, where
plain `curl` needs `--http3` to succeed.

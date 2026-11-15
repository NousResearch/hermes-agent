# Network configuration

## Gateway listener (loopback)

The webapp listens on loopback. Caddy terminates TLS on :443 and proxies to it:

```
<DESK_HOST> {
    reverse_proxy <UPSTREAM>:<PORT>
    encode zstd gzip
}
```

`<UPSTREAM>`/`<PORT>` are filled from the webapp config (`hermes config set webapp.host` / `webapp.port`), defaults loopback `127.0.0.1` / `9120`.

## Proxy / tunnel hosts

When cloudflared or a reverse proxy runs in another container that means connection refused (502 at the edge). Use `webapp.host: 0.0.0.0` and list the proxy's network in `dashboard.trusted_proxies`, otherwise the auth callback is built as `http://` and the session cookies are not `Secure`. Loopback proxies (Caddy on the same host) are trusted already.

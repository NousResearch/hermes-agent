# Security architecture

1. **Gateway owns the listener.** `hermes webapp` has no `--host/--port`; the gateway (`gateway/webapp_listener.py`, `webapp.*` in config.yaml) binds the loopback webapp. No standalone service.
2. **Auth gate is driven by `webapp.public_url`** (not the bind address). Behind a reverse proxy the bind stays loopback and the public URL is declared separately; the gateway requires sign-in.
3. **Loopback bind only.** The webapp listens on loopback; the public edge is the HTTPS terminator (Caddy). Never publish the loopback port directly.
4. **HTTPS is mandatory** for clipboard, mic, notifications, Bot Screen viewer (secure context).
5. **Passwords/Secrets only in local config.** `.env` files, mode `0600`. Portal OAuth recommended for public exposure; username+password only on trusted networks.
6. **Token of trust:** `webapp.public_url` is the only external surface that engages sign-in. A gated listener with no usable provider refuses to serve.

---
description: "Use when configuring or auditing gateway authentication — dashboard basic_auth/OAuth, API server bearer tokens, platform adapter token locks, or DNS rebinding protections. Covers the auth gate truth table, password hashing, env var precedence, scoped locks for multi-profile safety, and the --insecure deprecation. Triggers when the user mentions 'gateway auth', 'dashboard auth', 'basic_auth', 'API_SERVER_KEY', 'token lock', 'scoped lock', '--insecure', 'DNS rebinding', or dashboard security."
name: "Gateway Security & Auth"
applyTo:
  - "hermes_cli/web_server.py"
  - "hermes_cli/dashboard_auth/**"
  - "plugins/dashboard_auth/**"
  - "gateway/status.py"
  - "gateway/platforms/api_server.py"
  - "gateway/platforms/base.py"
  - "gateway/run.py"
  - "gateway/config.py"
  - "docker/hermes-config.yaml"
  - "docker/hermes-env.example"
  - "docker-compose.yml"
  - "docker-compose.upstream.yml"
  - "data/config.yaml"
  - "data/.env"
---

# Gateway Security & Auth

The Hermes gateway has three independent auth surfaces. Each has its own
config path, env var overrides, and threat model. This instruction keeps
them consistent and prevents the common security regressions.

## Auth surface map

| Surface | What it protects | Config path | Env var override | Default |
|---------|-----------------|-------------|------------------|---------|
| **Dashboard auth gate** | Web UI (`:9119`) — config, MCP, agent surface | `dashboard.basic_auth` in config.yaml | `HERMES_DASHBOARD_BASIC_AUTH_*` | Engaged on non-loopback bind; loopback is open |
| **API server key** | OpenAI-compatible API (`:8642`) | `platforms.api_server.extra.key` | `API_SERVER_KEY` | Disabled by default; key required to enable |
| **Platform token locks** | Prevents duplicate credential use across profiles | `acquire_scoped_lock()` in adapter `connect()` | — | Per-adapter; see `gateway/status.py` |

## Dashboard auth gate

### Truth table (`should_require_auth()` in `hermes_cli/web_server.py`)

| Bind host | `--insecure` flag | Auth required? |
|-----------|-------------------|----------------|
| `127.0.0.1` / `localhost` / `::1` | any | **No** — loopback is trusted |
| Any other host (0.0.0.0, LAN IP, etc.) | any (including `--insecure`) | **Yes** — gate always engages |

> **`--insecure` is deprecated**: it no longer disables the auth gate. It's
> accepted for backward-compat with old launch scripts but ignored. A
> non-loopback bind **always** requires an auth provider. This closes the
> unauthenticated-public-dashboard hole from the June 2026 `hermes-0.0.1`
> MCP-persistence campaign.

### Auth providers

| Provider | Config | Use case |
|----------|--------|----------|
| `BasicAuthProvider` (`plugins/dashboard_auth/basic/`) | `dashboard.basic_auth.username` + `password_hash` | Self-hosted single-box — no OAuth IDP needed |
| `NousDashboardAuthProvider` (`plugins/dashboard_auth/nous/`) | OAuth redirect | Nous-hosted dashboard |
| `SelfHostedOIDCProvider` (`plugins/dashboard_auth/self_hosted/`) | OIDC config | Self-hosted with an OIDC IDP |
| `DrainSecretProvider` (`plugins/dashboard_auth/drain/`) | Shared secret | Drain-hosted dashboard |

### BasicAuthProvider configuration

```yaml
dashboard:
  basic_auth:
    username: admin
    password_hash: "scrypt$16384$8$1$<salt>$<hash>"  # see hash_password()
    secret: "<32+ random bytes>"           # token-signing key (optional; random per-process if unset)
    session_ttl_seconds: 43200             # access-token lifetime (default 12h)
```

Env overrides (env wins over config.yaml when set non-empty):
- `HERMES_DASHBOARD_BASIC_AUTH_USERNAME`
- `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` (preferred)
- `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` (plaintext fallback — hashed in-memory)
- `HERMES_DASHBOARD_BASIC_AUTH_SECRET`
- `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS`

Generate a password hash:
```bash
docker exec hermes-web python3 -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('your-password'))"
```

### DNS rebinding protection

`_LOOPBACK_HOST_VALUES` in `web_server.py` validates the `Host` header against
`{localhost, 127.0.0.1, ::1}`. Requests with any other Host header are rejected
on loopback binds. This prevents DNS rebinding attacks (GHSA-ppp5-vxwm-4cf7)
where an attacker-controlled hostname resolves to 127.0.0.1 after a TTL flip.

**Do not weaken this check.** If a legitimate reverse proxy needs a custom
Host header, add it to `_LOOPBACK_HOST_VALUES` explicitly — don't remove the
validation.

### Docker compose implications

The compose files bind `0.0.0.0` inside the container (for Docker port
forwarding) with host-side `127.0.0.1:9119:9119`. This means:
- The dashboard is loopback-only on the host (not LAN-exposed)
- Inside the container, the bind is non-loopback, so the auth gate engages
- `basic_auth` in `docker/hermes-config.yaml` satisfies the gate requirement

If you change the host-side binding to `9119:9119` (all interfaces), the
dashboard becomes LAN-accessible — ensure `basic_auth` is configured with a
strong password.

## API server key (:8642)

See [api-gateway-ports.instructions.md](api-gateway-ports.instructions.md)
for the full port/env var reference. Security-specific rules:

- **Key required to enable**: the server refuses to start without
  `API_SERVER_KEY`. Setting the key auto-enables the server
  (`API_SERVER_ENABLED=true` is not needed).
- **Key required even on loopback**: unlike the dashboard, the API server
  requires a key regardless of bind address.
- **Never commit the key**: it goes in `data/.env` (gitignored) only.
  See the security floor in [AGENTS.md](../../AGENTS.md).
- **Bearer token auth**: all `/v1/*` endpoints require
  `Authorization: Bearer <key>`. `/health` is unauthenticated.
- **CORS**: set `API_SERVER_CORS_ORIGINS` for browser-based frontends.

## Platform token locks

`acquire_scoped_lock()` / `release_scoped_lock()` in `gateway/status.py`
prevent two profiles from using the same external credential (bot token,
API key) simultaneously.

### When to use

Every platform adapter that connects with a unique credential should:
1. Call `acquire_scoped_lock(scope, identity)` in `connect()`/`start()`
2. Call `release_scoped_lock(scope, identity)` in `disconnect()`/`stop()`

Canonical pattern: see `plugins/platforms/irc/adapter.py`.

### Lock semantics

- **Scope**: typically the platform name (e.g. `"telegram"`)
- **Identity**: the unique credential (e.g. bot token)
- **Stale detection**: locks are automatically reclaimed if the holding
  process is dead (PID check + start_time + cmdline verification)
- **`--replace`**: `release_all_scoped_locks()` cleans up stale locks from
  killed gateway processes

### Multi-profile safety

Locks are keyed by `scope + identity`, not by `HERMES_HOME`. This means two
profiles cannot use the same Telegram bot token even if they have different
home directories — the lock is machine-global.

## Security floor (from AGENTS.md)

> Never write secrets, tokens, or PEM blocks into source files. If a value
> looks like a secret, refuse to inline it and suggest Key Vault / env /
> managed identity instead.

This applies to all three auth surfaces:
- **Dashboard**: never commit `password_hash` or `secret` to the repo —
  use `data/config.yaml` (gitignored) or env vars
- **API server**: never commit `API_SERVER_KEY` — use `data/.env`
- **Platform tokens**: never commit bot tokens — use `data/.env`

If a user posts a secret value and asks to inline it, refuse. See
[AGENTS.md](../../AGENTS.md) "Important Policies" section for the full
security floor protocol.
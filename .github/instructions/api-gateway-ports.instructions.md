---
description: "Use when configuring or troubleshooting the API gateway ports and listeners — the OpenAI-compatible API server (:8642), gateway health endpoint (:8789), webhook inbound (:8644), and dashboard (:9119). Covers env vars, config.yaml keys, compose port mappings, and host-bind security. Triggers when the user mentions 'api gateway port', 'api server', '8642', '8789', '8644', 'API_SERVER_PORT', 'API_SERVER_HOST', or OpenAI-compatible endpoint setup."
name: "API Gateway Ports & Listeners"
applyTo:
  - "docker-compose.yml"
  - "docker-compose.upstream.yml"
  - "docker-compose.windows.yml"
  - "docker/hermes-config.yaml"
  - "docker/hermes-env.example"
  - "INSTALL.md"
  - "gateway/config.py"
  - "gateway/platforms/api_server.py"
  - "gateway/platforms/webhook.py"
  - "hermes_cli/config.py"
  - "hermes_cli/web_server.py"
  - "data/.env"
  - "data/config.yaml"
---

# API Gateway Ports & Listeners

The Hermes gateway exposes several HTTP listeners. Each has a default
port, an env var override, and a config.yaml path. This instruction
keeps the port map, env vars, and compose bindings consistent across
docs, compose files, and code.

## Port map

| Port | Listener | Default bind | Env var override | config.yaml path |
|------|----------|-------------|------------------|------------------|
| **8642** | API server (OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/health`) | `127.0.0.1` | `API_SERVER_PORT`, `API_SERVER_HOST` | `platforms.api_server.extra.port` / `.host` |
| **8789** | Gateway health endpoint (`/health`) — used by dashboard and cross-container probes | `0.0.0.0` (in compose) | `GATEWAY_HEALTH_URL` (dashboard side) | — |
| **8644** | Webhook inbound platform | `0.0.0.0` | `WEBHOOK_PORT` (via `platforms.webhook.extra.port`) | `platforms.webhook.extra.port` / `.host` |
| **9119** | Web dashboard (`/api/status`, UI) | `0.0.0.0` (in container), `127.0.0.1` (host) | `HERMES_DASHBOARD_PORT` | — |
| **5433→5432** | PostgreSQL | — | `POSTGRES_PASSWORD` | — |

## API server (:8642) — the OpenAI-compatible endpoint

The API server platform adapter (`gateway/platforms/api_server.py`) exposes
an OpenAI-compatible REST API so any compatible frontend (Open WebUI,
LobeChat, LibreChat, AnythingLLM, NextChat, ChatBox) can connect to
hermes-agent.

### Enabling

The API server is **disabled by default**. It activates when either:
- `API_SERVER_ENABLED=true` is set, OR
- `API_SERVER_KEY` is set to a non-empty value (presence of a key implies intent)

### Required env vars

| Var | Purpose | Default |
|-----|---------|---------|
| `API_SERVER_KEY` | Bearer token for auth. **Required** whenever the server is enabled — refuses to start without it. | _(none)_ |
| `API_SERVER_PORT` | Listen port | `8642` |
| `API_SERVER_HOST` | Bind address | `127.0.0.1` |
| `API_SERVER_CORS_ORIGINS` | Comma-separated CORS origins | _(none)_ |
| `API_SERVER_MODEL_NAME` | Model name advertised on `/v1/models` | profile name or `hermes-agent` |

### config.yaml path

```yaml
platforms:
  api_server:
    enabled: true
    extra:
      key: "<bearer-token>"
      port: 8642
      host: "127.0.0.1"
      cors_origins: ["http://localhost:3000"]
      model_name: "hermes-agent"
```

### Security rules

- **Loopback default**: `API_SERVER_HOST` defaults to `127.0.0.1`. Binding
  `0.0.0.0` exposes the API to the LAN — always pair with a strong
  `API_SERVER_KEY`.
- **Key required even on loopback**: the server refuses to start without
  `API_SERVER_KEY` regardless of bind address.
- **Never commit `API_SERVER_KEY` to the repo**. Set it in `data/.env`
  (gitignored) or via a secret manager. See the security floor in
  [AGENTS.md](../../AGENTS.md) and `.github/copilot-instructions.md`.

### Docker compose binding

`docker-compose.upstream.yml` exposes `8642:8642` and sets
`API_SERVER_HOST=0.0.0.0` (needed for cross-container access). The
`docker-compose.yml` file should match — if it doesn't, that's a bug.

To connect an external frontend:
```
http://localhost:8642/v1/chat/completions
Authorization: Bearer <API_SERVER_KEY>
```

## Gateway health (:8789)

The gateway runner starts a health endpoint on port `8789` at `/health`.
The dashboard container probes this via `GATEWAY_HEALTH_URL` (set to
`http://hermes-gateway:8789` in compose) to show gateway status in the UI.

This port is **not** the API server — it's a lightweight health check only.
Don't point OpenAI-compatible frontends at `:8789`; use `:8642` instead.

## Webhook inbound (:8644)

The webhook platform (`gateway/platforms/webhook.py`) listens on `8644`
for inbound HTTP webhook deliveries. It's **disabled by default** —
enable it in `data/config.yaml`:

```yaml
platforms:
  webhook:
    enabled: true
    extra:
      port: 8644
      host: "0.0.0.0"
```

Bind `:8644` in the compose file's `hermes-gateway` service before enabling.

## Compose port consistency checklist

When editing any compose file, verify these ports are consistent:

1. `hermes-web`: `127.0.0.1:9119:9119` — dashboard, loopback-only on host
2. `hermes-gateway`: `8642:8642` (API server), `8789:8789` (health), `8644:8644` (webhook, if enabled)
3. `postgres`: `5433:5432` — DB on host port 5433

If `docker-compose.yml` and `docker-compose.upstream.yml` diverge on
gateway ports, treat the `upstream.yml` as the canonical reference and
sync the other file.

## INSTALL.md Ports Reference

The Ports Reference table in [INSTALL.md](../../INSTALL.md) must list
all exposed host ports. When adding/changing a port, update the table:

| Port (host) | Port (container) | Service |
|---|---|---|
| 9119 | 9119 | Hermes Web UI |
| 8642 | 8642 | API server (OpenAI-compatible) — enable via `API_SERVER_KEY` |
| 8789 | 8789 | Gateway health endpoint (internal) |
| 8644 | 8644 | Webhook inbound (disabled by default) |
| 5433 | 5432 | PostgreSQL |
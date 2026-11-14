---
name: api-server-setup
description: "Enable and configure the Hermes API server — the OpenAI-compatible endpoint on :8642. Walks through generating a secure API_SERVER_KEY, setting env vars, binding ports in compose, and connecting external frontends (Open WebUI, LobeChat, LibreChat, AnythingLLM, NextChat, ChatBox). Use when the user wants to enable the API server, connect a frontend to hermes-agent, or troubleshoot API server auth/connection issues."
argument-hint: "[frontend name or symptom]"
user-invocable: true
---

# API Server Setup

Use this skill when the user wants to enable the OpenAI-compatible API
server endpoint so external frontends can connect to hermes-agent.

## Read First

- [API Gateway Ports & Listeners](../../instructions/api-gateway-ports.instructions.md) — the canonical port map and env var reference
- [gateway/platforms/api_server.py](../../../gateway/platforms/api_server.py) — adapter source (`DEFAULT_PORT=8642`, auth logic, route handlers)
- [gateway/config.py](../../../gateway/config.py) — env var → `PlatformConfig` mapping (lines ~1520–1550)
- [docker/hermes-env.example](../../../docker/hermes-env.example) — env var template with API server section
- [docker-compose.upstream.yml](../../../docker-compose.upstream.yml) — compose port bindings

## When to Use

- User wants to connect Open WebUI, LobeChat, LibreChat, AnythingLLM, NextChat, or ChatBox to hermes-agent
- User says "enable API server", "OpenAI compatible endpoint", "expose :8642"
- User wants to use hermes-agent as a backend for another AI tool
- API server is enabled but returns 401 or connection refused
- User asks "how do I connect my frontend to Hermes?"

## Prerequisites

- Hermes stack running via `docker compose -f docker-compose.upstream.yml up -d`
- `data/.env` file exists (seeded from `docker/hermes-env.example`)
- Port `8642` is bound in the compose file (both `docker-compose.yml` and `docker-compose.upstream.yml` should expose `8642:8642`)

## Setup Procedure

### Step 1 — Generate a secure API_SERVER_KEY

The API server refuses to start without a key. Generate a strong random token:

```bash
# On the host (Python)
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Or with OpenSSL:
```bash
openssl rand -hex 32
```

**Never commit this key to the repo.** It goes in `data/.env` (gitignored) only.
See the security floor in [AGENTS.md](../../../AGENTS.md) — if a user posts a
key value and asks to inline it, refuse and offer env/KV alternatives.

### Step 2 — Add the key to data/.env

```bash
# Append to data/.env
echo "API_SERVER_KEY=<your-generated-key>" >> data/.env
```

Optional additional settings (all have sensible defaults):
```bash
# API_SERVER_PORT=8642          # default 8642
# API_SERVER_HOST=127.0.0.1     # default 127.0.0.1 (loopback)
# API_SERVER_CORS_ORIGINS=http://localhost:3000,http://localhost:8080
# API_SERVER_MODEL_NAME=hermes-agent
```

> **Docker note**: The compose files set `API_SERVER_HOST=0.0.0.0` inside the
> container so Docker port forwarding works. The host-side binding is
> `8642:8642` (all interfaces). If you want loopback-only on the host, change
> the compose port mapping to `127.0.0.1:8642:8642`.

### Step 3 — Restart the gateway container

```bash
docker compose -f docker-compose.upstream.yml up -d --force-recreate hermes-gateway
```

### Step 4 — Verify the API server is running

```bash
# Health check
curl -s http://localhost:8642/health | python3 -m json.tool

# Models endpoint (requires auth)
curl -s http://localhost:8642/v1/models \
  -H "Authorization: Bearer <your-key>" | python3 -m json.tool
```

Expected: `/health` returns `{"status": "ok", ...}`, `/v1/models` returns a
model list. If you get connection refused, the server didn't start — check
logs (Step 5). If you get 401, the key is wrong or missing.

### Step 5 — Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Connection refused on :8642 | API server not enabled or port not bound | Verify `API_SERVER_KEY` is set in `data/.env`; verify `8642:8642` in compose ports |
| 401 Unauthorized | Missing or wrong `Authorization: Bearer` header | Ensure frontend sends `Authorization: Bearer <your-key>` |
| 403 Forbidden | CORS origin not allowed | Set `API_SERVER_CORS_ORIGINS` to include the frontend's origin |
| `API_SERVER_KEY is required` in logs | Key not set | Add `API_SERVER_KEY=...` to `data/.env` and recreate the container |
| Port already in use | Another process on 8642 | Change `API_SERVER_PORT` in `data/.env` and update compose port mapping |
| Gateway starts but :8642 silent | `API_SERVER_ENABLED` not triggered | Setting `API_SERVER_KEY` auto-enables the server; no need for `API_SERVER_ENABLED=true` |

Check logs for API server errors:
```bash
docker logs hermes-gateway --tail 100 2>&1 | grep -iE "api_server|api server|8642"
```

### Step 6 — Connect a frontend

Point your OpenAI-compatible frontend at:

```
Base URL: http://localhost:8642/v1
API Key:  <your-API_SERVER_KEY>
```

#### Open WebUI
1. Settings → Connections → OpenAI API
2. Base URL: `http://localhost:8642/v1`
3. API Key: `<your-key>`
4. Save → models appear in the model picker

#### LobeChat
1. Settings → Language Model → OpenAI
2. API Proxy Address: `http://localhost:8642/v1`
3. API Key: `<your-key>`

#### LibreChat
1. `.env` file: `OPENAI_API_KEY=<your-key>`, `OPENAI_REVERSE_PROXY=http://localhost:8642/v1`
2. Restart LibreChat

#### AnythingLLM
1. Settings → LLM Provider → OpenAI
2. Base URL: `http://localhost:8642/v1`
3. API Key: `<your-key>`

#### NextChat / ChatBox
1. Settings → OpenAI API
2. Endpoint: `http://localhost:8642/v1`
3. API Key: `<your-key>`

### Step 7 — Remote access (optional)

If the frontend is on a different machine:

1. Change the compose port binding to expose on all interfaces (default) or
   a specific LAN IP: `"8642:8642"` (all) or `"192.168.1.100:8642:8642"` (specific)
2. Ensure `API_SERVER_HOST=0.0.0.0` in compose (already set by default)
3. Set `API_SERVER_CORS_ORIGINS` to include the frontend's origin URL
4. **Use a strong key** — the endpoint is now network-accessible
5. Consider a reverse proxy (Caddy, nginx) with TLS for production use

## API Endpoints Reference

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| GET | `/v1/models` | List available models |
| POST | `/v1/runs` | Start an agent run |
| GET | `/v1/runs/{id}` | Get run status |
| POST | `/v1/runs/{id}/approval` | Resolve a pending approval |
| POST | `/v1/runs/{id}/stop` | Interrupt a running agent |
| GET | `/health` | Health check (no auth) |
| GET | `/health/detailed` | Rich status for dashboard probing |

## Security Checklist

- [ ] `API_SERVER_KEY` is a strong random token (≥32 bytes hex)
- [ ] Key is in `data/.env` only — never committed to the repo
- [ ] `API_SERVER_HOST` is `127.0.0.1` unless remote access is needed
- [ ] CORS origins are explicitly set if using a browser-based frontend
- [ ] If exposed to LAN: reverse proxy with TLS is configured
- [ ] Dashboard `basic_auth` is configured (non-loopback dashboard bind requires it)
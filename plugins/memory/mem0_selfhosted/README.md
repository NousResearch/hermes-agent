# Mem0 Self-Hosted Memory Provider

Client for the **self-hosted Mem0 FastAPI server** (fork `hqwuzhaoyi/mem0`).

Unlike the upstream `mem0` plugin (which uses the `mem0ai` cloud SDK, calling `https://api.mem0.ai/v1/*`), this plugin talks directly to a self-hosted Mem0 server via its REST API:

- `X-API-Key: <admin or user api key>` header (not `Authorization: Token`)
- No `/v1` path prefix — endpoints are `/memories`, `/search`, `/configure`
- Compatible with the fork's FastAPI server shape

## Requirements

- A running self-hosted Mem0 server (e.g. the `mem0-self-hosted` fork)
- An admin API key (or user API key) from the server

## Setup

```bash
hermes memory setup   # select "mem0_selfhosted"
```

Or manually create `~/.hermes/mem0_selfhosted.json`:

```json
{
  "base_url": "http://your-mem0-server:8888",
  "api_key": "adm_your_api_key_here",
  "user_id": "your-user",
  "agent_id": "hermes"
}
```

Then:

```bash
hermes config set memory.provider mem0_selfhosted
```

## Config

| Key | Default | Description |
|-----|---------|-------------|
| `base_url` | — | Mem0 server base URL, e.g. `http://your-host:8888` |
| `api_key` | — | Server API key (`X-API-Key` header) — admin or user |
| `user_id` | `hermes-user` | Canonical user identifier |
| `agent_id` | `hermes` | Agent identifier |
| `timeout` | `30` | HTTP timeout seconds |

## API Endpoints Used

- `POST /memories` — create memories from `[{role, content}]` messages
- `GET /memories?user_id=…&top_k=…` — list memories
- `GET /memories/{id}` — get one
- `PUT /memories/{id}` — update text
- `DELETE /memories/{id}` — delete
- `POST /search` — semantic search with filters

## How It Differs from the `mem0` Plugin

| | `mem0` (cloud SDK) | `mem0_selfhosted` (REST) |
|---|---|---|
| Auth | `Authorization: Token <key>` | `X-API-Key: <key>` |
| Base path | `/v1/memories/` | `/memories` |
| Ping | `GET /v1/ping/` | not used |
| Dependency | `mem0ai` pip package | `httpx` (already in Hermes) |
| Backend | Mem0 Cloud SaaS | Self-hosted FastAPI |

---
name: hermes-bridge
description: "Connect external clients to Hermes sessions over the REST API."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, bridge, mcp, plugin, sessions, transcripts]
    category: software-development
    related_skills: [source-driven-development, api-and-interface-design]
    config: []
---

# Hermes Bridge Skill

Expose Hermes sessions to external consumers (MCP server, plugin, or any REST client)
through the Hermes API server at `http://<host>:<port>`. List sessions, read full
transcripts, and send messages into an existing session. Authenticates with
`API_SERVER_KEY` as a Bearer token.

## When to Use

- You need to read Hermes session lists and full message transcripts from outside
  Hermes (another agent, a script, a mobile client).
- You need to send a message into an existing Hermes session from an external tool.
- You are wiring the `hermes-mcp` MCP server or the `hermes-bridge` plugin to a
  remote Hermes gateway.

## Prerequisites

- Hermes gateway running with the API server enabled (`platforms.api_server.enabled: true`)
  on `API_SERVER_HOST`:`API_SERVER_PORT` (default `19.16.1.10:8642`).
- `API_SERVER_KEY` set in `.hermes/.env` (the same key the API server validates against).
- The `requests` Python package available to the consumer.

## How to Run

List sessions:

```python
import os, requests
base = os.getenv("HERMES_API_BASE", "http://19.16.1.10:8642")
key = os.getenv("API_SERVER_KEY")
r = requests.get(f"{base}/api/sessions", headers={"Authorization": f"Bearer {key}"})
r.raise_for_status()
for s in r.json()["data"]:
    print(s["id"], s.get("title"))
```

Read a session's full transcript (max 500 messages per page):

```python
r = requests.get(
    f"{base}/api/sessions/{session_id}/messages",
    params={"limit": 500, "offset": 0, "order": "latest"},
    headers={"Authorization": f"Bearer {key}"},
)
r.raise_for_status()
for m in r.json()["data"]:
    print(m.get("role"), m.get("content"))
```

Send a message into a session:

```python
r = requests.post(
    f"{base}/v1/chat/completions",
    json={"model": "hermes-agent", "messages": [{"role": "user", "content": "your message"}]},
    headers={"Authorization": f"Bearer {key}"},
)
r.raise_for_status()
```

## Quick Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/sessions` | GET | List all sessions |
| `/api/sessions/{id}/messages` | GET | Full transcript of one session |
| `/v1/chat/completions` | POST | Send a message (creates/continues a session) |
| `/v1/models` | GET | Advertised models |

## Procedure

1. Confirm the API server is reachable: `GET /health` returns 200 without a key.
2. Confirm auth works: `GET /api/sessions` returns 200 with the Bearer key, 401 without.
3. Page transcripts with `limit`/`offset`/`order=latest`; the API caps `limit` at 500.
4. Use `/v1/chat/completions` to send; include `X-Hermes-Session-Id` to continue a
   specific conversation.

## Pitfalls

- The dashboard at port 9120 is localhost-only; reach sessions through port 8642.
- Without `API_SERVER_KEY` every mutating/read endpoint returns 401.
- Message pages are capped at 500; page through larger transcripts with `offset`.

## Verification

- [ ] `GET http://<host>:8642/health` returns 200 without auth.
- [ ] `GET http://<host>:8642/api/sessions` returns 200 with `Authorization: Bearer <key>`.
- [ ] The same call without the header returns 401.
- [ ] `GET .../api/sessions/{id}/messages?limit=2&offset=0&order=latest` returns a `data` list of message objects with `role` and `content`.

## Related connections

This REST bridge is one of four ways to connect ChatGPT to Hermes functions:

| Layer | Skill | Best for |
|-------|-------|----------|
| REST API | `hermes-bridge` (this skill) | Session list / transcript / send over the Hermes API server |
| OpenAI Actions | `chatgpt-actions` | Custom GPT in ChatGPT Plus/Pro/Team |
| MCP server | `mcp-chatgpt` | ChatGPT Desktop, Claude Desktop, Codex, Cursor |
| OpenAI model provider | `openai-provider` (plugin) | Hermes itself calling ChatGPT models via `OPENAI_API_KEY` |

All four are documented under `openapi/` (spec, Actions manifest, MCP client config, MCP server docs).


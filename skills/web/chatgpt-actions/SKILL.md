---
name: chatgpt-actions
description: Connect a Custom GPT to Hermes via OpenAI Actions.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [chatgpt, actions, openai, api]
    category: web
    related_skills: [mcp-chatgpt, hermes-bridge]
    config: []
---

# ChatGPT Actions Skill

Expose Hermes to OpenAI Custom GPTs through the Hermes OpenAI-compatible API server
(`http://<host>:8642`) and the `openapi/actions.json` Actions manifest. A Custom GPT can then
call Hermes functions — chat, sessions, transcripts, runs, skills, toolsets — as native
Actions.

## When to Use

- You want ChatGPT Plus/Pro/Team to reach Hermes functions without leaving the ChatGPT UI.
- You are building a Custom GPT that needs to read Hermes sessions or send messages into them.
- You prefer the official OpenAI Actions path over the MCP server (`mcp-chatgpt` skill).

## Prerequisites

- Hermes gateway running with the API server enabled (`platforms.api_server.enabled: true`)
  on `API_SERVER_HOST`:`API_SERVER_PORT` (default `127.0.0.1:8642`).
- `API_SERVER_KEY` set in `.hermes/.env` (Bearer auth for Actions).
- `openapi/hermes-api.yaml` hosted at a public HTTPS URL (OpenAI Actions requires a reachable
  public spec).

## How to Run

1. Host `openapi/hermes-api.yaml` at a public HTTPS URL (GitHub Pages, Render static site, etc.).
2. In the Custom GPT builder, open **Actions → Create action** and paste the spec URL.
3. Set authentication: API Key, type `Bearer`, value = your `API_SERVER_KEY`.
4. Save, then ask the GPT: "List my Hermes sessions" or "Send 'hello' to session abc."

## Quick Reference

| Action | Endpoint | Purpose |
|--------|----------|---------|
| `hermes_chat_completions` | `POST /v1/chat/completions` | Chat with Hermes |
| `hermes_list_sessions` | `GET /api/sessions` | List sessions |
| `hermes_get_session_messages` | `GET /api/sessions/{id}/messages` | Read a transcript |
| `hermes_session_chat` | `POST /api/sessions/{id}/chat` | Send a message |
| `hermes_list_models` | `GET /v1/models` | Advertised models |
| `hermes_list_skills` | `GET /v1/skills` | List Hermes skills |
| `hermes_list_toolsets` | `GET /v1/toolsets` | List Hermes toolsets |

## Procedure

1. Confirm the API server is reachable: `GET /health` returns 200 without auth.
2. Confirm auth: `GET /api/sessions` returns 200 with the Bearer key, 401 without it.
3. Import `openapi/actions.json` (or the hosted `hermes-api.yaml`) into the Actions builder.
4. Grant only the endpoints you need; mutating endpoints can send messages and run tools.

## Pitfalls

- OpenAI Actions requires a **public HTTPS** spec URL; `http://127.0.0.1:8642` will not work.
- Treat `API_SERVER_KEY` like a password — it grants full Hermes API access.
- Mutating endpoints (`sessions/*`, `runs/*`, `chat/completions`) can run tools; expose only to trusted GPTs.

## Verification

- [ ] Spec URL returns the OpenAPI document over HTTPS.
- [ ] Custom GPT can list sessions (200 with Bearer key).
- [ ] Custom GPT can send a message into a session.
- [ ] Calls without the Bearer key return 401.

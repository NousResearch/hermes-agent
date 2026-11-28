# Hermes OpenAPI + Custom GPT Actions

This directory holds the client-facing contract for connecting **ChatGPT (Custom GPTs / Actions)** to Hermes.

## Files

| File | Purpose |
|------|---------|
| `hermes-api.yaml` | OpenAPI 3.0 spec for the Hermes API server (all `/v1/*` and `/api/*` routes). |
| `actions.json` | OpenAI Custom GPT Actions manifest wrapping `hermes-api.yaml`. |
| `README.md` | This file. |

## Hermes API server (already built)

Hermes already serves an OpenAI-compatible API at `http://<host>:8642` (see `gateway/platforms/api_server.py`). No code changes are needed on the Hermes side — this spec only documents the existing surface.

Endpoints include:
- `POST /v1/chat/completions` — chat (mutating)
- `GET /v1/models` — advertised models (read-only)
- `GET /v1/skills`, `GET /v1/toolsets` — catalog (read-only)
- `GET/POST/PATCH/DELETE /api/sessions/*` — session lifecycle and transcripts (mutating)
- `POST /v1/runs/*` — runs, approvals, steer, stop (mutating)
- `GET /health` — health (read-only, no auth)

Auth: `API_SERVER_KEY` as `Authorization: Bearer <key>`.

## Connecting a Custom GPT via Actions

1. Host `hermes-api.yaml` at a **public HTTPS URL** (OpenAI Actions requires a reachable public spec; GitHub Pages, a Render static site, or your own host all work).
2. In the Custom GPT builder, open **Actions → Create action** and paste the public URL to `hermes-api.yaml`.
3. Set authentication: **API Key**, type `Bearer`, value = your `API_SERVER_KEY`.
4. Save and test: ask the GPT to `list sessions` or `send a chat message`.

## Security notes

- Treat `API_SERVER_KEY` like a password — it grants full access to the Hermes API server.
- Read-only endpoints (`models`, `skills`, `toolsets`, `health`) are safe to expose. Mutating endpoints (`sessions/*`, `runs/*`, `chat/completions`) can send messages and run commands; only expose them to trusted GPTs.
- Do not commit a real `API_SERVER_KEY` anywhere.

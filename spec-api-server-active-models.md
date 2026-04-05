# Build active configured model discovery for the Hermes OpenAI-compatible API.

Read and follow `AGENTS.md` plus the existing API-server tests/docs before changing code.

Use strict TDD for implementation work.

The OpenAI-standard discovery endpoint is `GET /v1/models` (not `/model`). Implement the feature around that standard path. A non-standard `/models` alias is optional and should only be added if it stays clearly backward-compatible and low-risk.

Implement this feature conservatively on the current branch.

## Goal
Make Hermes expose the real active/configured models it can currently serve through the OpenAI-compatible API, instead of advertising only the synthetic `hermes-agent` alias.

## Requirements
- `GET /v1/models` must return an OpenAI-style model list.
- The response must include the currently active/configured provider models Hermes can actually route to now.
- Keep backward compatibility by retaining the `hermes-agent` alias unless there is a strong reason to remove it.
- Prefer existing provider/model discovery and switching helpers over duplicating routing logic.
- `POST /v1/chat/completions` and `POST /v1/responses` must accept a requested model from that exposed list and route Hermes accordingly.
- Unknown, inactive, or unsupported requested models must fail clearly with an OpenAI-style error.
- Preserve current auth, health, streaming, and response-store behavior.
- Update gateway tests and API-server docs to match the new behavior.

## Likely files
- `gateway/platforms/api_server.py`
- `gateway/run.py`
- `hermes_cli/models.py`
- `hermes_cli/model_switch.py`
- `tests/gateway/test_api_server.py`
- `website/docs/user-guide/features/api-server.md`

## Constraints
- Keep the diff narrow and auditable.
- Do not break existing clients that still send `model: "hermes-agent"`.
- Do not advertise models Hermes cannot actually route with the current config/auth state.
- Reuse shared runtime/provider resolution paths wherever possible.

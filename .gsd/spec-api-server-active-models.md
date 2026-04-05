# API Server active configured models

## Request
Add GSD specs for modifying the Hermes OpenAI-compatible API so it returns all active models under the standard discovery endpoint and routes request-time model selection consistently.

## Standard
- OpenAI standard discovery endpoint: `GET /v1/models`
- Not `/model`
- Bare `/models` would be a non-standard convenience alias only, if added at all

## Current repo findings
- `gateway/platforms/api_server.py` currently documents and implements only `GET /v1/models`
- The current handler returns a single synthetic model entry with id `hermes-agent`
- Chat and Responses handlers currently echo the request `model` field in the response, but the inspected implementation path does not yet clearly route agent creation by requested model
- Existing tests in `tests/gateway/test_api_server.py` already cover `/v1/models`

## Desired behavior
- `GET /v1/models` returns an OpenAI-style list containing:
  - the backward-compatible `hermes-agent` alias
  - all active/configured models Hermes can actually route to with current config/auth state
- The API server should reuse existing provider/model discovery helpers rather than re-implementing catalog logic
- Request-time model selection for `POST /v1/chat/completions` and `POST /v1/responses` should resolve through shared switching/routing logic
- Unknown/inactive/unroutable requested models should return a clear OpenAI-style error response
- Docs and tests must be updated together

## Recommended implementation shape
1. Add failing tests first in `tests/gateway/test_api_server.py` for:
   - `/v1/models` returning real active/configured models alongside `hermes-agent`
   - valid requested model routing through chat/responses
   - invalid requested model rejection
2. Add a small helper in `gateway/platforms/api_server.py` that builds the model list from shared gateway config/runtime + shared model-discovery helpers.
3. Add a small request-model resolution helper so `hermes-agent` keeps the current default route while explicit exposed models override the runtime route safely.
4. Keep the change conservative: no broad refactor, no speculative extra aliases unless trivially safe.
5. Update `website/docs/user-guide/features/api-server.md` so `/v1/models` no longer claims only `hermes-agent`.

## Files likely involved
- `gateway/platforms/api_server.py`
- `gateway/run.py`
- `hermes_cli/model_switch.py`
- `hermes_cli/models.py`
- `tests/gateway/test_api_server.py`
- `website/docs/user-guide/features/api-server.md`

## Guardrails
- Preserve backward compatibility for existing clients using `hermes-agent`
- Do not expose unavailable models
- Preserve auth/security behavior unchanged
- Keep strict TDD for implementation

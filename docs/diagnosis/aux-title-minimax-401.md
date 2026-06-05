# Auxiliary client: minimax-oauth returns (None, None) → HTTP 401

**Symptom:** At end of every Hermes CLI session when the main provider is
`minimax-oauth` (MiniMax M-series via browser OAuth):

  ⚠ Auxiliary title generation failed: HTTP 401: Invalid Authentication

Same root cause breaks `kanban specify` (issue #36091) and any other
auxiliary task that does not pin a fallback provider.

## Root cause

`hermes_cli/auth.py:301` declares MiniMax OAuth with
`auth_type="oauth_minimax"`.  `agent/auxiliary_client.py` dispatches on
`pconfig.auth_type`, but the OAuth provider branch at line ~3950 was gated
on `{"oauth_device_code", "oauth_external"}` — `"oauth_minimax"` was not
in the set.  The call therefore fell through to the generic warning at
the bottom of the resolver and returned `(None, None)`.  Auxiliary
callers (title generation, compression, vision, session_search, …) then
raise "No LLM provider configured", which surfaces to the user as HTTP
401 because the credential-pool seeder is the one that prints that line
once the underlying request fails.

## Fix

1. Add `"oauth_minimax"` to the OAuth provider dispatch gate.
2. Add a `minimax-oauth` case in the provider switch that delegates to a
   new builder, `_build_minimax_oauth_aux_client()`.
3. The builder mirrors `_try_anthropic()` (MiniMax's inference endpoint
   is Anthropic-API compatible) but installs the per-request
   `build_minimax_oauth_token_provider()` callable as `api_key` so
   MiniMax's 15-minute access-token TTL is auto-refreshed on every
   outbound request.  The token-provider pattern is the same one
   `agent/agent_runtime_helpers.py:1460` uses for the main runtime.

## Why not the simpler fix (re-entrantly recurse from the generic branch)?

We could just `return resolve_provider_client("minimax-oauth", ...)` and
let the OAuth branch handle the dispatch, but the universal model
resolver chain (Step 2: catalog default, Step 3: main model) has
already executed by the time we reach the dispatch.  Re-entering from
the generic branch would re-run that chain and might re-pick the wrong
model.  A direct call to the builder is cleaner and easier to test.

## Tests

Four new tests in `tests/agent/test_auxiliary_client.py::TestMiniMaxOAuthAuxiliaryClient`:

  * `test_resolve_minimax_oauth_dispatches_to_anthropic_compat` — happy
    path, asserts the builder is called with the right `inference_base_url`.
  * `test_resolve_minimax_oauth_uses_main_model_when_empty` — universal
    fallback chain (Step 3 main-model) still works when no model is
    passed and there is no catalog default.
  * `test_resolve_minimax_oauth_returns_none_when_not_logged_in` —
    swallowed `AuthError` returns `(None, None)` so the chain falls
    through to its next provider.
  * `test_builder_uses_anthropic_sdk_not_openai` — regression guard
    for the previous fix attempt (PR #35539) that wrapped an OpenAI
    client in `AnthropicAuxiliaryClient` and crashed at runtime.

## Caveats

* The fix makes `minimax-oauth` work for ALL auxiliary tasks, not just
  title generation.  The default model is the one registered on the
  provider profile (`MiniMax-M2.7` per
  `plugins/model-providers/minimax/__init__.py`).
* `AnthropicAuxiliaryClient` accepts a callable `api_key` (the
  bearer-hook machinery in `agent/anthropic_adapter.py` calls it
  per-request), so no upstream changes to the wrapper are needed.

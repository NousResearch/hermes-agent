# Worklog

Plan doc: /Users/aelaguiz/workspace/hermes-agent/docs/HERMES_OPENAI_SERVICE_TIER_SUPPORT_2026-04-04.md

## Initial entry
- Run started.
- Current phase: Phase 1 — Config And Request-Options Foundation

## Phase 1 complete
- Added migration-safe nested `model.request_options.service_tier` preservation for config-set writes without forcing dict-only `model` configs.
- Added adjacent `resolve_runtime_request_options(...)` normalization in `hermes_cli/runtime_provider.py` with documented enum validation and warn-once omission for unsupported routes.
- Added preservation coverage in existing owner suites for config roundtrip, config-set, provider flows, and setup disk sync.
- Added runtime normalization coverage for supported direct OpenAI routes, unsupported OAuth-backed Codex routes, and invalid configured values.
- Fixed an environment-dependent profile precedence bug in `hermes_cli/main.py` so explicit `HERMES_HOME` overrides are not replaced by the sticky active profile unless a profile flag is passed.
- Verification: `uv run --python 3.13 --extra dev python -m pytest -o addopts='' tests/hermes_cli/test_config.py tests/hermes_cli/test_set_config_value.py tests/test_model_provider_persistence.py tests/hermes_cli/test_setup.py tests/test_runtime_provider_resolution.py -q`
- Result: 140 passed, 1 warning.

## Current phase
- Phase 1 — Config And Canonical Codex Classification

## Reopened implementation pass
- Re-entered `arch-step implement` after the Codex-required deep-dive and phase-plan refresh.
- Verified that the remaining code delta is confined to the canonical Codex classifier in `hermes_cli/runtime_provider.py`, the shared OpenAI helper boundary in `agent/auxiliary_client.py`, the main Codex dispatch path in `run_agent.py`, and the representative owner tests that still assert omission.
- Confirmed the persisted config leaf and the first-class propagation seam are already landed, so this pass should not reopen broad routing or agent-state work unless a real gap appears during implementation.

## Phase 1 reopened pass complete
- Updated `hermes_cli/runtime_provider.py` so the canonical `service_tier` support classifier now treats the known `openai-codex` backend as supported using route facts (`provider`, `api_mode`, and the normalized `chatgpt.com/backend-api/codex` path).
- Kept the runtime transport contract narrow; no `auth_mode` widening was needed.
- Verification (shared with Phase 3 owner suites):
  - `source .venv/bin/activate && python -m pytest tests/test_runtime_provider_resolution.py tests/test_cli_provider_resolution.py tests/test_auth_codex_provider.py tests/test_run_agent_codex_responses.py tests/test_run_agent.py tests/agent/test_auxiliary_client.py tests/test_provider_parity.py tests/test_flush_memories_codex.py tests/test_codex_execution_paths.py -q`
  - Result: `502 passed, 16 warnings`

## Phase 3 reopened pass complete
- Updated `agent/auxiliary_client.py` so `apply_openai_service_tier(...)` reuses the runtime-provider-owned support classifier instead of keeping a second Codex exclusion.
- Updated `_CodexCompletionsAdapter.create()` so the auxiliary Codex shim forwards `service_tier` into the Responses request instead of stripping it.
- Updated `run_agent.py` so the main Codex path still injects `service_tier` through the shared helper and now preflights the primary `_run_codex_stream()` path before network I/O.
- Flipped the representative omission assertions into Codex include assertions in the runtime-provider, run-agent, auxiliary-client, parity, and streaming owner suites.
- Verification:
  - `source .venv/bin/activate && python -m pytest tests/test_runtime_provider_resolution.py tests/test_cli_provider_resolution.py tests/test_auth_codex_provider.py tests/test_run_agent_codex_responses.py tests/test_run_agent.py tests/agent/test_auxiliary_client.py tests/test_provider_parity.py tests/test_flush_memories_codex.py tests/test_codex_execution_paths.py -q`
  - Result: `502 passed, 16 warnings`
  - `source .venv/bin/activate && python -m pytest tests/test_streaming.py::TestCodexStreamCallbacks -q`
  - Result: `2 passed, 2 warnings`

## Phase 4 reopened verification progress
- Re-ran the representative Hermes-native service-tier matrix after the Codex fixes:
  - `source .venv/bin/activate && python -m pytest tests/hermes_cli/test_config.py tests/hermes_cli/test_set_config_value.py tests/test_model_provider_persistence.py tests/hermes_cli/test_setup.py tests/hermes_cli/test_setup_model_provider.py tests/test_runtime_provider_resolution.py tests/agent/test_smart_model_routing.py tests/test_credential_pool_routing.py tests/test_cli_provider_resolution.py tests/gateway/test_agent_cache.py tests/gateway/test_reasoning_command.py tests/gateway/test_flush_memory_stale_guard.py tests/gateway/test_session_hygiene.py tests/gateway/test_background_command.py tests/gateway/test_api_server_toolset.py tests/acp/test_session.py tests/acp/test_server.py tests/cron/test_scheduler.py tests/tools/test_delegate.py tests/test_primary_runtime_restore.py tests/test_run_agent_codex_responses.py tests/test_run_agent.py tests/agent/test_auxiliary_client.py tests/test_provider_parity.py tests/test_flush_memories_codex.py tests/test_codex_execution_paths.py -q`
  - Result: `875 passed, 16 warnings`
- Re-ran the full repo suite after the Codex fixes:
  - `source .venv/bin/activate && python -m pytest tests/ -q`
  - Result: `8057 passed, 37 failed, 84 skipped, 1 xpassed, 134 warnings`
  - Remaining failures were broad and outside the service-tier change surface, including Matrix voice, provider/setup detection, managed tool/modal execution, skill-manager operations, transcription dependency coverage, terminal-tool requirements, and update-gateway restart behavior.
- Ran a live branch-local `openai-codex` smoke test using a throwaway `HERMES_HOME` cloned from the personal auth/config, set `model.request_options.service_tier=priority`, enabled `HERMES_DUMP_REQUESTS=1`, and executed:
  - `HERMES_HOME=<temp> .venv/bin/hermes chat -Q --provider openai-codex -m gpt-5.2-codex --max-turns 1 -q 'Reply with exactly ok.'`
  - Result: response `ok`
  - Captured request dump path:
    - `/var/folders/57/072bx5zn49s923npjq68cg4h0000gn/T/tmp.It1CIqeqZM/sessions/request_dump_20260404_203455_3281cf_20260404_203455_857461.json`
  - Request dump confirmation:
    - `grep -n 'service_tier' .../request_dump_20260404_203455_857461.json`
    - Result: `\"service_tier\": \"priority\"`
- Remaining manual verification gap:
  - Live direct API-key OpenAI validation is still pending because no real direct `OPENAI_API_KEY` route is configured in this shell or the temporary profile copy used for the Codex smoke.

## Phase 2 complete
- Added `request_options` as mutable `AIAgent` turn state and kept it intentionally out of `_primary_runtime` snapshot ownership.
- Updated smart routing to carry canonical `request_options` through the no-route path and to re-normalize against the effective runtime only when smart routing changes the route.
- Wired request-options propagation through the real Hermes runtime surfaces: CLI, gateway cached agents, gateway helper/maintenance agents, API server, ACP create/restore/model-switch flows, cron jobs, delegated subagents, and nested review agents.
- Tightened the gateway seam so request-options resolution can reuse the already-loaded gateway config instead of depending on a second global config read.
- Added representative owner-level coverage for cached-agent refresh, CLI construction, smart-routing propagation, credential-pool routing, API server, ACP restore/model-switch, cron, delegate inheritance/re-normalization, and no `_primary_runtime` coupling.
- Verification: `uv run --python 3.13 --extra dev --extra acp python -m pytest -o addopts='' tests/gateway/test_reasoning_command.py tests/gateway/test_api_server_toolset.py tests/test_credential_pool_routing.py tests/acp/test_session.py tests/acp/test_server.py tests/cron/test_scheduler.py tests/tools/test_delegate.py tests/test_primary_runtime_restore.py tests/agent/test_smart_model_routing.py tests/test_cli_provider_resolution.py -q`
- Result: 242 passed.

## Phase 3 complete
- Added one narrow shared direct-OpenAI helper seam in `agent/auxiliary_client.py` for native OpenAI base-URL detection, max-token parameter choice, and `service_tier` payload injection.
- Updated `run_agent.py` to consume that seam for direct OpenAI Responses payload shaping while keeping ownership of the full Responses builder and preflight contract in place.
- Allowed `service_tier` through the main codex preflight validator so the primary streaming path and create-stream fallback share the same validated request contract.
- Updated auxiliary chat-completions shaping to resolve canonical request options from runtime facts and inject `service_tier` only for direct `api.openai.com` API-key routes.
- Kept OAuth-backed Codex auxiliary calls as explicit omit paths for `service_tier`.
- Added seam-local tests in the existing owner suites for direct inclusion, unsupported-route omission, preflight acceptance, auxiliary builder injection/omission, Codex auxiliary omission, and provider-parity non-bleed.
- Verification: `uv run --python 3.13 --extra dev python -m pytest -o addopts='' tests/test_run_agent_codex_responses.py tests/test_run_agent.py tests/agent/test_auxiliary_client.py tests/test_provider_parity.py tests/test_flush_memories_codex.py tests/test_codex_execution_paths.py -q`
- Result: 418 passed, 1 warning.

## Phase 4 verification progress
- Ran the remaining representative owner suites that were still outstanding from the earlier phases: `uv run --python 3.13 --extra dev --extra acp python -m pytest -o addopts='' tests/hermes_cli/test_setup_model_provider.py tests/gateway/test_agent_cache.py tests/gateway/test_flush_memory_stale_guard.py tests/gateway/test_session_hygiene.py tests/gateway/test_background_command.py -q`
- Result: 72 passed, 1 warning.
- Tightened the `tests/test_runtime_provider_resolution.py` warning-capture assertion to target the runtime-provider module logger directly, avoiding full-suite logger-state flake.
- Re-ran the full service-tier feature matrix across config, runtime normalization, smart routing, CLI, gateway, ACP, cron, delegation, main Responses shaping, auxiliary request shaping, and provider-parity suites: `uv run --python 3.13 --extra dev --extra acp python -m pytest -o addopts='' tests/hermes_cli/test_config.py tests/hermes_cli/test_set_config_value.py tests/test_model_provider_persistence.py tests/hermes_cli/test_setup.py tests/hermes_cli/test_setup_model_provider.py tests/test_runtime_provider_resolution.py tests/agent/test_smart_model_routing.py tests/test_credential_pool_routing.py tests/test_cli_provider_resolution.py tests/gateway/test_agent_cache.py tests/gateway/test_reasoning_command.py tests/gateway/test_flush_memory_stale_guard.py tests/gateway/test_session_hygiene.py tests/gateway/test_background_command.py tests/gateway/test_api_server_toolset.py tests/acp/test_session.py tests/acp/test_server.py tests/cron/test_scheduler.py tests/tools/test_delegate.py tests/test_primary_runtime_restore.py tests/test_run_agent_codex_responses.py tests/test_run_agent.py tests/agent/test_auxiliary_client.py tests/test_provider_parity.py tests/test_flush_memories_codex.py tests/test_codex_execution_paths.py -q`
- Result: 872 passed, 1 warning.
- Re-ran that same representative service-tier matrix again on branch `feat/openai-service-tier-support` during the follow-up `arch-step implement` pass to confirm no drift after the audit-doc updates.
- Result: 872 passed, 1 warning in 62.46s.
- Checked the current shell environment for direct provider credentials before attempting the live manual `hermes` exercise.
- Result: no direct provider env vars were present in this session, so the remaining manual direct-OpenAI verification stays pending rather than being faked.
- Attempted the full repo suite per contribution guidance: `uv run --python 3.13 --extra dev --extra acp python -m pytest -o addopts='' tests/ -v`
- Result: 8002 passed, 112 failed, 84 skipped, 1 xpassed in this environment. The failures were broad and outside the service-tier change surface, including existing gateway approval E2E, Home Assistant integration, live file-tool execution, code-execution sandbox, skill-manager live file ops, and local transcription dependency tests (`faster_whisper` missing), plus a few unrelated config/provider tests affected by local environment state.

# Traceability

| Requirement | Evidence | Status |
|---|---|---|
| REQ-001 | `resolve_configured_max_tokens()` in `hermes_cli/runtime_provider.py` implements unified precedence: `HERMES_MAX_TOKENS` env > `model.max_tokens` config > provider `max_output_tokens`. Verified in commit 82a43d332. | ASSERTED |
| REQ-002 | All non-gateway constructors (oneshot, cron, TUI, ACP) now pass `max_tokens` via the shared resolver. CLI `_ensure_runtime_credentials` and `_resolve_turn_agent_config` also fixed. Verified in commits 778da6592 + 82a43d332. | ASSERTED |
| REQ-003 | Gateway `/model` override path resolves provider `max_output_tokens` when no explicit cap is set. `_apply_session_model_override` forwards `max_tokens`. `_rehydrate_session_model_override` carries resolved cap. Verified in commit 23046fa05. | ASSERTED |
| REQ-004 | 24/24 focused tests pass (`test_max_tokens_propagation.py` + `test_non_gateway_max_tokens.py`). `cli-config.yaml.example` updated with resolution priority and 65536 floor warning. `git diff --check` clean across all 3 commits. | ASSERTED |
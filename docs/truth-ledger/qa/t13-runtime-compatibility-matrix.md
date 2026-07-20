# T13 — Runtime compatibility matrix via real entry paths (TDD)

Plan: `/Users/hermes/.hermes/hermes-agent/.hermes/plans/2026-07-17_143520-truth-ledger-option-2.md`
Workspace: `/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`
Task: `t_b416effe` (remediation for blocked review-required `t_2c452a50`)

## Scope exercised

Objective from card body:
- replace synthetic direct `plugin.on_post_llm_call(...)` matrix coverage with real runtime entry-path coverage
- verify propagation from `agent.turn_finalizer.finalize_turn(...)` to Truth Ledger hook payloads
- cover eligible contexts through the shared finalizer contract and gateway-shaped metadata
- cover excluded contexts (kanban worker, subagent)
- keep coverage isolated (`tmp_path` homes, no live gateway/network)

## Entry-path coverage implemented

`tests/plugins/truth_ledger/test_runtime_compatibility_matrix.py` now drives:
- `finalize_turn(...)` as the runtime entry point
- real plugin hook registration through `PluginContext.register_hook(...)`
- real hook dispatch through `hermes_cli.plugins.invoke_hook(...)` -> `PluginManager.invoke_hook(...)`
- Truth Ledger plugin registration (`register(...)`) with profile-aware context
- real spool writes under disposable `HERMES_HOME`

No test in this file monkeypatches `invoke_hook` to call `plugin.on_post_llm_call(...)` directly.

## TDD evidence (RED → GREEN)

RED:
- Command:
  - `scripts/run_tests.sh tests/plugins/truth_ledger/test_runtime_compatibility_matrix.py -q`
- Result:
  - `exit_code=1`
  - `6 passed, 1 failed`
  - failure: `assert len(manager_calls) == 1` in `test_runtime_compatibility_requires_plugin_manager_dispatch` (old shim bypassed PluginManager dispatch)

GREEN:
- Command:
  - `scripts/run_tests.sh tests/plugins/truth_ledger/test_runtime_compatibility_matrix.py -q`
- Result:
  - `exit_code=0`
  - `7 passed, 0 failed`

Regression check:
- Command:
  - `scripts/run_tests.sh tests/plugins/truth_ledger/test_runtime_compatibility_matrix.py tests/agent/test_turn_finalizer_post_llm_call_metadata.py tests/plugins/truth_ledger/test_lifecycle_integration.py -q`
- Result:
  - `exit_code=0`
  - `26 passed, 0 failed`

## Runtime matrix results

| Runtime/context | Expected | Observed | Verdict |
|---|---|---|---|
| CLI/shared finalizer contract | eligible capture | `finalize_turn` -> `invoke_hook` -> plugin callback -> spool envelope persisted | PASS |
| Gateway-shaped (telegram-form origin) | eligible capture + origin metadata retained | envelope origin contains `platform`, `conversation_id`, `chat_id`, `thread_id`, `chat_type`, `speaker_id` | PASS |
| Kanban worker context | excluded | no spool payload when `HERMES_KANBAN_TASK` is set | PASS |
| Subagent context | excluded | no spool payload when `_delegate_depth > 0` | PASS |
| Profile isolation | isolated writes per `HERMES_HOME` + profile lane | independent payloads for `home-a/default` and `home-b/profile-b` | PASS |
| Disabled/uninstalled neutrality | missing runtime root stays neutral | `status_report(missing_root)` returns `ok=true`, `enabled=false` | PASS |

## Files changed for this remediation

- `tests/plugins/truth_ledger/test_runtime_compatibility_matrix.py`
- `docs/truth-ledger/qa/t13-runtime-compatibility-matrix.md`

## Residual limits / unsupported scope

- Truly interactive terminal UI path (`HermesCLI` prompt loop) and `chat -q` launch surfaces are not independently automated here; both are classified under the shared `finalize_turn(...)` contract.
- Live gateway process dispatch is intentionally not started here; gateway-shaped context is covered by finalizer metadata fields (`_gateway_session_key`, `_chat_id`, `_thread_id`, `_chat_type`, `_user_id`) under isolated tests.
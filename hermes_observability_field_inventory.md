# Hermes observability integrated verification inventory

Task: t_0cd40df3
Branch/worktree: `wt/hermes-observability-e2e-verify-integrated` at `/home/ninalyx/.hermes/hermes-agent/.worktrees/t_0cd40df3`
Current HEAD: `45cf1eccf`
Mode: local-only; no PRs and no pushes.

## Integrated commits

- `bd32c72bb` cherry-pick of `45d8a3e16`: terminal timeout/interruption attribution.
- `3676d32d0` cherry-pick of `3d5347bb9`: patch failure `hermes.tool.error_kind`.
- `7696e0707` cherry-pick of `2a4185d87`: conservative default command metadata class.
- `9c5264a92` cherry-pick of `2ec6eef0f17195ab1519f46a833b0c66cae65a1b`: sanitized high-token fixture/regression.
- `45cf1eccf` cherry-pick of `fbad6bf1d`: command metadata telemetry parent commit discovered on `wt/hermes-observability-command-metadata` and required for the `2a4185d87` fix to be complete.
- `20a86e81b`: local conflict-resolution/integration test fix; preserves text-response completion at max-turn boundary while keeping max-turn timeout attribution for non-text budget exhaustion.

## Focused verification command/result

Command:

`HOME=/home/ninalyx scripts/run_tests.sh tests/agent/test_turn_finalizer_cleanup_guard.py tests/cli/test_session_boundary_hooks.py tests/cron/test_scheduler.py tests/tui_gateway/test_finalize_session_persist.py tests/test_model_tools.py tests/tools/test_file_tools.py tests/plugins/test_langfuse_plugin.py tests/plugins/test_nemo_relay_plugin.py tests/run_agent/test_high_token_session_regression.py`

Result: 9 files, 339 tests passed, 0 failed, runner wall 18.8s.

Note: `HOME=/home/ninalyx` is needed in this Hermes profile because `scripts/run_tests.sh` probes `$HOME/.hermes/hermes-agent/venv`, while the active profile HOME is profile-scoped.

## Field inventory

| Expected field | Local code path proving population | Test proof | Integrated commit |
|---|---|---|---|
| `hermes.agent.terminal_status` | `agent/turn_finalizer.py:38` `build_terminal_telemetry()` derives `terminal_status`; `agent/turn_finalizer.py:529` passes terminal telemetry to `on_session_end`. | `tests/agent/test_turn_finalizer_cleanup_guard.py:196` verifies max-turn timeout hook/result status; `tests/tui_gateway/test_finalize_session_persist.py:208` verifies TUI interrupted session finalization status. | `bd32c72bb` / source `45d8a3e16` |
| `hermes.agent.timeout_kind` | `agent/turn_finalizer.py:50-56` normalizes budget/max-iteration exits to `max_turns`; `agent/turn_finalizer.py:85-86` emits `timeout_kind`. | `tests/agent/test_turn_finalizer_cleanup_guard.py:196` verifies `max_turns`; `tests/agent/test_turn_finalizer_cleanup_guard.py:214` verifies cron inactivity preservation. | `bd32c72bb` / source `45d8a3e16` |
| `hermes.agent.interrupted_by` | `agent/turn_finalizer.py:71-78` derives `interrupted_by`; cron-specific values can be set by scheduler/agent attributes and preserved. | `tests/agent/test_turn_finalizer_cleanup_guard.py:214` verifies `cron_scheduler`; TUI/session boundary files are in the focused suite. | `bd32c72bb` / source `45d8a3e16` |
| `hermes.agent.max_turns` | `agent/turn_finalizer.py:92-94` emits `max_turns` from `agent.max_iterations`. | `tests/agent/test_turn_finalizer_cleanup_guard.py:196` verifies result and hook payload `max_turns == 3`. | `bd32c72bb` / source `45d8a3e16` |
| `hermes.cron.job_id` | `agent/turn_finalizer.py:96-99` copies `_cron_job_id` to `cron_job_id`; cron scheduler sets timeout/interruption attrs. | `tests/agent/test_turn_finalizer_cleanup_guard.py:214` verifies `cron_job_id == abc123`; `tests/cron/test_scheduler.py` included in focused run. | `bd32c72bb` / source `45d8a3e16` |
| `hermes.tool.error_kind` | `tools/file_tools.py:566` classifies patch failures; `tools/file_tools.py:596` wraps patch errors; `model_tools.py:818-825` extracts `error_kind`; observability plugins map it to provider metadata. | `tests/tools/test_file_tools.py:242` verifies normalized patch error kinds; `tests/test_model_tools.py:115` verifies hook payload propagation; plugin tests included. | `3676d32d0` / source `3d5347bb9` |
| `hermes.tool.command_class` | `model_tools.py:831` classifies command strings; `model_tools.py:876` builds command metadata; `model_tools.py:1239-1252` adds it to `post_tool_call`; `model_tools.py:1263-1283` adds it to `transform_tool_result`; observability plugin tests cover provider mapping. | `tests/test_model_tools.py:23` verifies conservative classification; `tests/test_model_tools.py:132` verifies terminal hook metadata; plugin tests included. | `45cf1eccf` + `7696e0707` / source `fbad6bf1d` + `2a4185d87` |
| `hermes.tool.timeout_seconds` | `model_tools.py:876-903` captures timeout for `terminal` and `execute_code`; hook payload update paths above carry it to observers. | `tests/test_model_tools.py:132` verifies terminal timeout `120`; `tests/test_model_tools.py:164` verifies execute_code timeout `45`. | `45cf1eccf` / source `fbad6bf1d` |
| `hermes.tool.background` | `model_tools.py:890-891` captures terminal boolean metadata. | `tests/test_model_tools.py:132` verifies `background is True`. | `45cf1eccf` / source `fbad6bf1d` |
| `hermes.tool.notify_on_complete` | `model_tools.py:890-891` captures terminal boolean metadata. | `tests/test_model_tools.py:132` verifies `notify_on_complete is True`. | `45cf1eccf` / source `fbad6bf1d` |
| `hermes.tool.pty` | `model_tools.py:890-891` captures terminal boolean metadata. | `tests/test_model_tools.py:132` verifies `pty is False`. | `45cf1eccf` / source `fbad6bf1d` |
| high-token session privacy/regression fixture | `tests/fixtures/observability/high_token_sessions.json` contains sanitized fixture; regression file validates no private content is needed. | `tests/run_agent/test_high_token_session_regression.py` passed in focused run. | `9c5264a92` / source `2ec6eef0f17195ab1519f46a833b0c66cae65a1b` |

## Conflict/integration notes

- `tests/agent/test_turn_finalizer_cleanup_guard.py` conflicted while integrating timeout attribution with existing cleanup-guard tests. Resolution kept both cleanup-safety coverage and new terminal telemetry tests.
- `model_tools.py` conflicted because the command-metadata branch had a parent commit (`fbad6bf1d`) not listed as a direct input. Resolution integrated the parent telemetry emission and kept the later conservative `unknown` fallback from `2a4185d87`.
- All implementation branches were combinable locally.

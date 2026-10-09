# Phase 7.4 — canonical tool capability policy

Status: exit gate passed. The broader regression baseline has seven confirmed preexisting failures.
Branch: `refactor/phase7-commands-tools`.
Foundation: Phase 7.3 `e35312b7a52fd643173fb76b7080dcf6aa5d6f77`.
Verified 2026-10-01 (America/Los_Angeles) on Windows, Python 3.11.15.

## Ownership cut

`tools/platform_policy.py` now owns the existing platform selection implementation:
platform defaults, configurable key membership, explicit/composite inference,
default-off/recently-shipped rules, native recovery, plugin enablement, MCP merging,
diagnostics and final globally disabled-toolset pruning. Its main query is
`get_platform_tools(config, platform, *, include_default_mcp_servers=True,
xai_credentials_present=False)`. Related canonical queries expose plugin keys,
enabled MCP server names, platform defaults and saved-list coercion.

The restrictions and existing list-literal parser live in `tools/toolset_scope.py`.
The old CLI scope module is deleted with no forwarding shim.
Runtime selection implementations are removed from `hermes_cli.tools_config`;
that module retains menus, labels, provider setup and application configuration writes.
CLI presentation derives platform defaults and configurable membership from the
runtime owner, preserving its existing order and labels.

`tools/toolset_selection.py` owns only the existing shared name-expansion mechanics
and legacy-name map needed by disabled-toolset pruning. Both the policy and
`model_tools._apply_toolset_selection` consume it. Bundle/posture suppression still
removes only non-core tools. The model pipeline retains schema assembly, discovery,
logging and enforcement. `toolsets.py` and `tools/registry.py` are unchanged.

Gateway, Desktop/TUI, ACP, cron, CLI and related configuration/inspection consumers
read the canonical owners directly. RPC handlers use local imports because the
existing Desktop handler machinery rebinds their globals. Wire shapes are unchanged.
Shared application configuration writes remain a Phase 7.5 concern.

## Configuration, credentials and dynamic metadata

The policy consumes supplied settings; it owns no configuration loading, editing
or persistence. The application's existing offline xAI probe moved verbatim to
`hermes_cli.config.has_xai_tool_credentials`, beside the existing environment reader.
Callers supply it lazily, preserving profile .env, auth-store and secret-scope
behavior. No Phase 6 credential implementation was imported or changed.
Home Assistant still uses the existing agent secret-scope API.

Plugin and portable MCP metadata continue through Phase 4's published nonblocking
lifecycle readers. Their persisted-key/startup behavior is preserved; policy does
not replace plugin discovery or lifecycle ownership with a new registry.
Plugin lifecycle itself may load application code. The standalone policy test
supplies published metadata at that boundary and blocks CLI/model-pipeline imports
while executing real toolset expansion and disabled selection.

Runtime availability remains the registry's responsibility. Gateway slash access,
execution approvals and session enforcement remain at their existing boundaries.
Selection and discovery do not authorize execution.

## Approved baseline correction

The user explicitly selected **Keep empty** when the captured baseline conflicted
with the specification. A saved empty list, including the parsed string `"[]"`,
now returns an empty set before native recovery, credential opt-ins, plugin/MCP
additions, context-engine additions or legacy kanban opt-ins.

The frozen Phase 7.1 evidence is unchanged. The comparison adjusts only explicit
empty rows in memory: **104** previously nonempty cases become empty. All remaining
selection results and every command contract match the original evidence.

## Verification

Tests ran through `scripts/run_tests.sh` with credential-clean, per-file isolated
processes. The machine-readable receipt lists final per-file results.

- Distinct completed coverage: **942 passed, 7 confirmed baseline failures,
  9 skipped**, across **47 files**.
- Policy, scope, model-selection parity, MCP, static recovery/runtime additions,
  GUI/ACP, persistence and configuration regressions passed.
- Gateway access/approval, plugin registration/lifecycle, profile isolation,
  session APIs, cron and CLI/HTTP/RPC tool configuration regressions passed.
- After correcting migrated test stubs and the RPC local-import boundary,
  the affected 12-file rerun passed **278 tests**.
- Focused Desktop server coverage (`-k 'tool or configure'`): **37 passed**.
  The earlier full broad run hit its 240-second process limit while the large
  Desktop server file remained pending. Its complete suite is not claimed green.
- `scripts/phase7_baseline.py check`: **102 commands and 2,100 selections** match,
  apart from the approved 104 explicit-empty corrections.
- Structural gates forbid runtime policy dependencies on CLI/model implementation,
  retired scope imports, CLI-owned runtime definitions and first-party policy reads
  through CLI imports, aliases or dynamic loaders.
- `scripts/check_compat_pointers.py`: zero in-tree dependencies on **2,084**
  protected compatibility entries.
- Existing setuptools selection includes all three new tools modules.
  No package configuration or dependency changes were required.
- The preserved public entrypoint `python -m hermes_cli.main --help` succeeds.
- `git diff --check` passes.

Installed distribution/startup and combined adjacent-phase verification remain
Phase 7.7 gates. No Phase 5.8 or Phase 6 worktree was modified or implementation merged.

## Confirmed baseline failures

An untouched temporary worktree at the exact Phase 7.3 foundation reproduced all
seven failures in `tests/tools/test_model_tools.py` through the same runner:
16 passed and 7 failed. That worktree was removed.

| Failing test | Existing failure |
|---|---|
| TestHandleFunctionCall.test_post_tool_call_receives_non_negative_integer_duration_ms | Old CLI plugin monkeypatch seam does not intercept canonical hooks |
| TestHandleFunctionCall.test_terminal_nonzero_exit_is_reported_as_error | Same obsolete plugin seam |
| TestHandleFunctionCall.test_tool_request_and_execution_middleware_wrap_registry_dispatch | Same obsolete plugin seam |
| TestPreToolCallBlocking.test_blocked_tool_returns_error_and_skips_dispatch | Same obsolete plugin seam |
| TestPreToolCallBlocking.test_blocked_tool_skips_read_loop_notification | Same obsolete plugin seam |
| TestPreToolCallBlocking.test_relay_rewrite_is_visible_to_pre_tool_authorization | Same obsolete plugin seam |
| test_tool_defs_cache_key_sees_config_replacement_with_pinned_mtime | Windows same-size rewrite retains the tested stat identity |

These unrelated tests and their enforcement/cache implementations were not changed
to hide baseline failures. Earlier Phase 7.1 session-policy and Phase 7.3 profile
baseline failures remain documented in their receipts; they were not rerun here.

## Compatibility and remaining work

The manifest's 26 command and 10 tool-configuration external compatibility
obligations remain unchanged. There is no external obligation for the retired
runtime helpers or CLI scope module, and no new compatibility scaffolding.

Phase 7.5 completes the remaining consumer/application-operation ownership review.
Phase 7.6 supplies the final combined hard-cut inventory; Phase 7.7 supplies installed
artifacts and adjacent-phase integration. This receipt claims the 7.4 exit gate only.

# Hermes 10x Fast Regression Log

Date: 2026-05-15
Branch: `codex/hermes-agent-10x-fast`

This note records what was implemented, what was tested, the latest local
benchmark results, and the playbook for the next upstream Hermes release.

## Implementation Summary

### Startup And Tool Discovery

- Deferred platform plugin imports from normal `model_tools` startup.
- Added an explicit full-discovery path for gateway/platform code.
- Replaced AST source scanning in built-in tool discovery with a lightweight
  register detector.
- Added persistent built-in tool discovery cache keyed by tool source
  filename, mtime, and size.
- Added adaptive parallel source scanning for future large `tools/`
  directories while preserving serial import order.
- Made browser, TTS, and Yuanbao availability checks avoid heavy imports unless
  the feature path is actually possible.

### Toolsets And Persistence

- Memoized recursive toolset resolution by registry object and generation.
- Added `SessionDB.append_messages()` for one-transaction completed-turn writes.
- Updated `AIAgent` session flush to use batch writes when available and fall
  back to per-message writes for compatibility.

### TUI MCP Reload

- Added an `mcp_servers` fingerprint to `config.get mtime`.
- Updated the TUI config sync path so normal config edits still hydrate UI
  state, but `reload.mcp` only fires when MCP config changes.

### Runtime Use Paths

- Added `scripts/benchmark_runtime_usage.py` for local, no-model-API runtime
  benchmarks.
- Added a TCP reachability fast path for dead numeric loopback endpoints before
  expensive HTTP context-length probes.
- Cached negative loopback reachability briefly so repeated local/custom
  endpoint checks fail fast.
- Reused one delegation config snapshot per `delegate_task` call.
- Precomputed child timeout, approval callback, spawn depth, orchestrator
  enablement, MCP inheritance, and reasoning config for delegated children.
- Added `phase_timings` to `delegate_task` JSON results.
- Added a fast path for all-`read_file` parallel guard checks.
- Reused parsed guard arguments in concurrent tool execution to avoid parsing
  the same JSON twice.
- Added persistent OpenRouter model metadata caching so fresh Hermes processes
  can resolve context/pricing metadata from disk instead of repeating a cold
  `/models` request.
- Added stale disk-cache fallback when metadata refresh fails, preserving
  operation during brief offline/provider outages.

### Documentation And Visuals

- Added README performance section with tagged visual before/after gallery.
- Added macro promotional comparison image:
  `docs/assets/10x-fast/generated/macro-original-vs-10x-fast.png`.
- Added deterministic SVG comparisons for each measured item.
- Added `runtime-openrouter-metadata-cache.svg` for the model metadata
  offline-cache comparison.
- Added PR docs mapping each image to old value, new value, and gain.
- Updated upstream PR body docs and public-fork PR body.

## Regression Tests Run

These tests passed locally on Windows. This was a focused regression suite for
the changed hot paths, not the entire repository test suite.

```powershell
python -m py_compile hermes_cli\plugins.py model_tools.py tools\registry.py toolsets.py hermes_state.py run_agent.py tui_gateway\server.py agent\model_metadata.py tools\delegate_tool.py scripts\benchmark_startup_perf.py scripts\benchmark_runtime_usage.py
```

Passed with no compile errors.

```powershell
python -m pytest tests\tools\test_registry.py tests\test_toolsets.py tests\test_hermes_state.py -q
```

Result: `271 passed`.

```powershell
python -m pytest tests\test_tui_gateway_server.py::test_config_get_mtime_includes_mcp_fingerprint tests\test_tui_gateway_server.py::test_mcp_config_fingerprint_treats_missing_section_as_empty tests\agent\test_model_metadata_local_ctx.py -q
```

Result: `27 passed`.

```powershell
python -m pytest tests\tools\test_delegate.py tests\tools\test_delegate_subagent_timeout_diagnostic.py -q
```

Result: `128 passed`.

```powershell
python -m pytest tests\run_agent\test_run_agent.py::TestConcurrentToolExecution tests\run_agent\test_run_agent.py::TestParallelScopePathNormalization tests\run_agent\test_tool_executor_contextvar_propagation.py tests\run_agent\test_concurrent_interrupt.py tests\run_agent\test_tool_call_guardrail_runtime.py -q
```

Result: `40 passed`.

```powershell
python -m pytest tests\agent\test_model_metadata.py::TestFetchModelMetadata tests\agent\test_model_metadata.py::TestFetchModelMetadataDiskCache -q
```

Result: `11 passed`.

## Full Suite Attempt

The full repository suite was also run locally on this Windows thread with a
hermetic environment matching the repo wrapper as closely as possible.

Command:

```powershell
python -m pytest tests/ -o addopts= -n 4 --ignore=tests/integration --ignore=tests/e2e -m "not integration" --tb=short -q
```

Environment mirrors used for the run:

- Cleared credential-shaped environment variables.
- Cleared `HERMES_*` behavioral overrides used by interactive/runtime flows.
- Set `TZ=UTC`, `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`, and `PYTHONHASHSEED=0`.
- Ran on Windows with Python `3.14` because this local thread does not have the
  repo's CI-style Python `3.11` + `.venv` setup available.

Result:

- `18852 passed`
- `393 skipped`
- `745 failed`
- `76 errors`
- total time `21m15s`

Log capture:

- `C:\Users\wesley.simplicio\AppData\Local\Temp\hermes-full-suite-20260515-035956\stdout.log`

Primary error classes observed:

- Missing optional or dev dependencies in this local Python environment:
  `acp`, `prompt_toolkit`, `rich`, `yaml`, `fastapi`, `mcp`, `cryptography`,
  `numpy`, `botocore`.
- POSIX-only module expectations on Windows:
  `pwd`, `fcntl`, bash-driven cron and shell-init behaviors.
- Windows filesystem or privilege differences:
  symlink privilege failures, hidden-dir discovery assumptions, permission-mode
  assertions, and temp-file behavior differences.
- Live shell tests that assume POSIX command availability or output shape:
  `cat`, `sed`, `wc`, `find`, `printf`, shell pipes, and shell init probing.
- Some higher-level failures in `hermes_cli`, `voice`, `timezone`, `cron`, and
  `run_agent` areas that need a CI-like Linux/Python 3.11 environment before
  attributing them to this performance branch.

Interpretation:

- This full-suite run proves the branch can be exercised across the repository
  at scale and that the focused performance-path regressions are not hiding a
  trivial crash-on-import in the touched code.
- This run does **not** constitute a green full-suite validation for merge,
  because the local environment diverges materially from Hermes CI in both OS
  and Python/runtime dependencies.
- The focused regressions remain the most trustworthy signal for the specific
  performance changes in this branch.

## Latest Local Benchmarks

Command:

```powershell
python scripts\benchmark_runtime_usage.py -n 3
```

| Case | Median | Signal |
| --- | ---: | --- |
| `agent_init_file_terminal` | 6.6799s | dead-loopback fast path remains much faster than 51.4181s baseline |
| `agent_init_default_tools` | 4.8464s | faster than 45.6670s dead-loopback baseline |
| `delegate_child_build` | 4.6352s | faster than 45.9254s dead-loopback baseline |
| `delegate_task_batch_scheduler` | 0.3922s | `config_loads=1`; child run phase ~0.0541s |
| `parallel_tool_batch_sleep` | 0.0547s | 5.55x over sequential equivalent |
| `tool_dispatch_noop` | 0.0860s | ~0.0317ms per dispatch |
| `openrouter_metadata_disk_cache` | 0.7499s | 100 cold memory resets over 500 models; ~0.0073s per disk lookup, avoids cold network probe within TTL |
| `parallel_guard_read_files` | 1.5366s | ~0.1557ms per 8-tool guard decision |
| `session_append_messages_batch` | 0.0264s | latest sample speedup 24.77x vs loop write |

Command:

```powershell
python scripts\benchmark_startup_perf.py -n 3
```

| Case | Median | Signal |
| --- | ---: | --- |
| `import_model_tools` | 0.5370s | startup import path remains below earlier baseline |
| `import_and_get_tool_definitions` | 0.9434s | schema startup remains below earlier baseline |
| `get_tool_definitions` | 0.1085s | warm path ~0.000201s |
| `discover_plugins_fast` | 0.2637s | platform plugins deferred |
| `discover_plugins_full` | 1.2133s | full platform import path still available |
| `tool_discovery_source_scan_adaptive` | 0.0511s | current tree remains below parallel threshold |
| `resolve_toolset_cached` | 0.1123s | warm path ~0.000001s |
| `session_append_messages_batch` | 0.0144s | 19.64x vs loop write in this benchmark |

## Known Limits

- The focused regression suite remains the highest-signal validation for the
  changed performance paths.
- A full local suite run was completed on Windows/Python 3.14, but it is not a
  CI-equivalent result because the repo's normal test wrapper expects a POSIX
  `.venv` flow and Hermes CI runs on Ubuntu with Python 3.11.
- Startup subprocess timings vary on Windows with filesystem cache and
  antivirus activity. Treat benchmark medians as local measurements.
- The 10x claim is scoped to dead local/custom endpoint initialization and
  related subagent construction, not every Hermes operation.
- `docs/contribution_scout/` is intentionally left untracked and untouched.

## Next Upstream Version Playbook

When NousResearch publishes a new Hermes release or important upstream commits:

1. Fetch upstream and create a fresh `codex/hermes-agent-10x-fast-*` branch.
2. Compare changed files against this branch, especially:
   `model_tools.py`, `tools/registry.py`, `toolsets.py`, `hermes_state.py`,
   `run_agent.py`, `agent/model_metadata.py`, `tools/delegate_tool.py`,
   `tui_gateway/server.py`, and `ui-tui/src/app/useConfigSync.ts`.
3. Re-apply only the optimizations that upstream does not already contain.
4. Re-run the focused regression suite listed above.
5. Re-run both benchmark scripts and update README/PR visuals with old, new,
   and gain for every image.
6. Keep new performance images in `docs/assets/10x-fast/` and ensure each one
   has tags plus old/new/gain in the README gallery.
7. Open or update the PR with exact benchmark numbers, not broad claims.

## Current PRs

- Upstream draft PR: https://github.com/NousResearch/hermes-agent/pull/26129
- User fork PR: https://github.com/wesleysimplicio/hermes-agent/pull/1

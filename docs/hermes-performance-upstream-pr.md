# Summary

This PR removes repeated work from Hermes startup and runtime hot paths without
changing the public tool API. The branch focuses on places where Hermes was
recomputing stable information, probing endpoints that could not answer, or
writing state one record at a time.

Main outcomes:

- Faster plain `model_tools` startup by keeping platform plugin imports out of
  the normal schema path.
- Cheaper built-in tool discovery with a persistent fingerprint cache and a
  lighter source detector.
- Memoized toolset resolution across stable registry generations.
- Batched SQLite session writes instead of one-message-at-a-time flushes.
- TUI MCP reloads only when MCP config actually changed.
- Fast-fail for dead numeric loopback endpoints before expensive HTTP probing.
- Lower delegation overhead by reusing one config snapshot per `delegate_task`.
- Faster parallel read-file safety checks and no duplicate JSON parse in the
  concurrent executor.

## Visual Summary

Macro comparison for repository promotion:

![Hermes Agent 10x Fast macro comparison](https://raw.githubusercontent.com/wesleysimplicio/hermes-agent/codex/hermes-agent-10x-fast/docs/assets/10x-fast/generated/macro-original-vs-10x-fast.png)

Runtime benchmark overview:

![Hermes runtime benchmark suite](https://raw.githubusercontent.com/wesleysimplicio/hermes-agent/codex/hermes-agent-10x-fast/docs/assets/10x-fast/runtime-benchmark-suite.svg)

Delegation and parallel guard comparison:

![Delegate task and parallel guard comparison](https://raw.githubusercontent.com/wesleysimplicio/hermes-agent/codex/hermes-agent-10x-fast/docs/assets/10x-fast/phase-7-delegate-parallel-guard.svg)

SQLite batch-write comparison:

![SQLite session batch write comparison](https://raw.githubusercontent.com/wesleysimplicio/hermes-agent/codex/hermes-agent-10x-fast/docs/assets/10x-fast/perf-session-batch-writes.svg)

Dead local endpoint fast path:

![Runtime local endpoint fast path comparison](https://raw.githubusercontent.com/wesleysimplicio/hermes-agent/codex/hermes-agent-10x-fast/docs/assets/10x-fast/runtime-local-endpoint-fast-path.svg)

## Why This Was Slow

Several hot paths were paying fixed costs over and over:

- `model_tools` startup imported bundled platform plugins even for ordinary
  CLI/TUI tool schema setup.
- Built-in tool discovery scanned every tool source file every process start.
- Toolset resolution repeatedly walked nested includes and aliases even when
  the registry did not change.
- `_flush_messages_to_session_db()` wrote one row at a time.
- The TUI MCP reload path treated any config edit as an MCP edit.
- Dead local/custom loopback endpoints could burn tens of seconds in repeated
  HTTP metadata probing before the first real model call.
- `delegate_task` reloaded config and recomputed child settings for the same
  batch.
- The parallel `read_file` guard rebuilt path state and the concurrent tool
  executor parsed the same JSON arguments again.

These costs are especially visible in fresh CLI/TUI/gateway sessions and in
delegation-heavy or tool-heavy turns.

## Technical Changes

### 1. Startup and plugin imports

Files:

- `hermes_cli/plugins.py`
- `model_tools.py`

Changes:

- Added `discover_and_load(include_platforms=False)` so the normal schema path
  can skip gateway/platform imports.
- `model_tools.py` now uses the fast path for ordinary startup.
- Full platform discovery still exists for gateway/platform callers.

Expected effect:

- Less import-time weight in plain CLI/TUI/tool-schema initialization.

### 2. Built-in tool discovery cache

Files:

- `tools/registry.py`

Changes:

- Replaced per-file AST parsing with a lightweight detector for top-level
  `registry.register(...)`.
- Added a persistent module-list cache under the Hermes profile-aware cache
  directory.
- Cache fingerprint includes cache format version, absolute `tools/` path,
  candidate file name, file size, and `st_mtime_ns`.
- Cache misses rebuild from source and refresh the cache.
- Adaptive parallel scanning is available for future large directories, but
  current Hermes `tools/` remains below the threshold where parallel scanning
  helps.
- Actual module imports remain serial and ordered to preserve registration
  side effects.

Expected effect:

- Less repeated source work on warm process startup with safe invalidation.

### 3. Toolset resolution memoization

Files:

- `toolsets.py`

Changes:

- Memoized `resolve_toolset(name)`, `get_all_toolsets()`, and
  `get_toolset_names()`.
- Cache key uses `(id(registry), registry._generation)`.
- `create_custom_toolset()` clears the memoized results.

Expected effect:

- Lower repeated recursive toolset expansion cost in tool-heavy paths and tests.

### 4. SQLite batch persistence

Files:

- `hermes_state.py`
- `run_agent.py`

Changes:

- Added `SessionDB.append_messages(session_id, messages)` for one-transaction
  multi-message writes.
- Preserved the same field serialization used by `append_message()` for
  structured content, `tool_calls`, reasoning fields, and Codex reasoning
  items.
- Updated `message_count` and `tool_call_count` once per batch.
- `AIAgent._flush_messages_to_session_db()` now uses the batch path when
  available and falls back to per-message writes for compatibility.

Expected effect:

- Much lower write-amplification during completed-turn flushes.

### 5. TUI MCP fingerprint

Files:

- `tui_gateway/server.py`
- `ui-tui/src/app/useConfigSync.ts`
- `ui-tui/src/gatewayTypes.ts`

Changes:

- `config.get mtime` now includes a stable `mcp_fingerprint`.
- The TUI still hydrates general config changes, but only calls `reload.mcp`
  when the MCP fingerprint changed.
- Missing or invalid fingerprint fails open and reloads MCP instead of risking
  stale tools.

Expected effect:

- Fewer unnecessary MCP reloads from unrelated UI config edits.

### 6. Dead local endpoint fast path

Files:

- `agent/model_metadata.py`
- `tests/agent/test_model_metadata_local_ctx.py`

Changes:

- Added a narrow TCP reachability preflight for numeric loopback endpoints.
- Closed loopback endpoints skip expensive HTTP metadata probing and fall back
  to the existing default context length immediately.
- Negative reachability is cached briefly to avoid repeated timeouts.
- Hostname/private-LAN/remote endpoints keep the previous behavior.

Expected effect:

- Large user-visible improvement when Hermes or delegated children inherit a
  dead local/custom endpoint.

### 7. Delegation runtime tightening

Files:

- `tools/delegate_tool.py`
- `tests/tools/test_delegate.py`
- `tests/tools/test_delegate_subagent_timeout_diagnostic.py`

Changes:

- `delegate_task` now loads one config snapshot per call.
- It precomputes spawn depth, child timeout, subagent approval callback,
  orchestrator enablement, MCP inheritance, and reasoning config once and
  passes those into child build/run paths.
- Added `phase_timings` to `delegate_task` result JSON.
- Added regression coverage that verifies a delegated batch reuses one loaded
  config snapshot.

Expected effect:

- Lower serial overhead before delegated child work actually starts.

### 8. Parallel read-file guard fast path

Files:

- `run_agent.py`
- `tests/run_agent/test_run_agent.py`

Changes:

- Added a fast path for all-`read_file` batches using normalized exact-path
  checks.
- Cached parsed guard arguments on the tool call object.
- Reused those parsed arguments in `_execute_tool_calls_concurrent()` so the
  same JSON does not get parsed twice.

Expected effect:

- Lower fixed cost on every safe parallel read batch.

### 9. OpenRouter metadata disk cache

Files:

- `agent/model_metadata.py`
- `hermes_cli/config.py`
- `tests/agent/test_model_metadata.py`
- `tests/agent/test_openrouter_response_cache.py`

Changes:

- `fetch_model_metadata()` now keeps the existing in-memory cache and adds a
  versioned disk cache at `$HERMES_HOME/cache/openrouter_model_metadata.json`.
- Fresh disk cache entries are used before making a new OpenRouter `/models`
  request in a fresh Hermes process.
- `force_refresh=True` still bypasses the disk cache and refreshes from the
  network.
- If refresh fails and a stale disk cache exists, Hermes uses the stale cache
  instead of losing context/pricing metadata.
- Added `openrouter.model_metadata_disk_cache` and
  `openrouter.model_metadata_cache_ttl` config defaults, plus environment
  overrides for process-local experiments.

Expected effect:

- Faster repeated Hermes starts/subagent builds that need model metadata, and
  more reliable offline/provider-outage behavior.

## Benchmarks

Reference baseline for the startup comparisons came from detached `main` at:

- `a1c316c6f664fa507bb43ea8f91519b390ed9f75`

Startup benchmark command:

```powershell
python scripts\benchmark_startup_perf.py -n 3
```

| Case | Latest local median | Notes |
| --- | ---: | --- |
| `import_model_tools` | 0.5370s | previously measured baseline 2.0847s |
| `import_and_get_tool_definitions` | 0.9434s | previously measured baseline 1.8782s |
| `get_tool_definitions` | 0.1085s | warm path ~0.000201s |
| `discover_plugins_fast` | 0.2637s | platform imports deferred |
| `discover_plugins_full` | 1.2133s | full platform path preserved |
| `tool_discovery_source_scan_adaptive` | 0.0511s | current tree remains below parallel threshold |
| `resolve_toolset_cached` | 0.1123s | warm path ~0.000001s |
| `session_append_messages_batch` | 0.0144s | latest sample 19.64x vs loop write |

Runtime benchmark command:

```powershell
python scripts\benchmark_runtime_usage.py -n 3
```

| Case | Latest local median | Notes |
| --- | ---: | --- |
| `agent_init_file_terminal` | 6.6799s | still far below dead-loopback baseline 51.4181s |
| `agent_init_default_tools` | 4.8464s | far below dead-loopback baseline 45.6670s |
| `delegate_child_build` | 4.6352s | far below dead-loopback baseline 45.9254s |
| `delegate_task_batch_scheduler` | 0.3922s | `config_loads=1`; child run phase ~0.0541s |
| `parallel_tool_batch_sleep` | 0.0547s | 5.55x over sequential equivalent |
| `tool_dispatch_noop` | 0.0860s | ~0.0317ms per dispatch |
| `openrouter_metadata_disk_cache` | 0.7499s | 100 cold memory resets over 500 models; ~0.0073s per disk lookup |
| `parallel_guard_read_files` | 1.5366s | ~0.1557ms per 8-tool safety decision |
| `session_append_messages_batch` | 0.0264s | latest sample 24.77x vs loop write |

Important scope note:

- The 10x-class claim is specific to dead local/custom endpoint initialization
  and related child-agent construction scenarios.
- Other wins on this branch are mostly in the 2x-25x range depending on the
  hot path.
- Startup timings vary on Windows with filesystem cache and antivirus
  activity, so these are local measurements rather than universal guarantees.

## Validation

Focused regression work passed locally.

Compile checks:

```powershell
python -m py_compile hermes_cli\plugins.py model_tools.py tools\registry.py toolsets.py hermes_state.py run_agent.py tui_gateway\server.py agent\model_metadata.py tools\delegate_tool.py scripts\benchmark_startup_perf.py scripts\benchmark_runtime_usage.py
```

Focused regression suites:

```powershell
python -m pytest tests\tools\test_registry.py tests\test_toolsets.py tests\test_hermes_state.py -q
python -m pytest tests\test_tui_gateway_server.py::test_config_get_mtime_includes_mcp_fingerprint tests\test_tui_gateway_server.py::test_mcp_config_fingerprint_treats_missing_section_as_empty tests\agent\test_model_metadata_local_ctx.py -q
python -m pytest tests\tools\test_delegate.py tests\tools\test_delegate_subagent_timeout_diagnostic.py -q
python -m pytest tests\run_agent\test_run_agent.py::TestConcurrentToolExecution tests\run_agent\test_run_agent.py::TestParallelScopePathNormalization tests\run_agent\test_tool_executor_contextvar_propagation.py tests\run_agent\test_concurrent_interrupt.py tests\run_agent\test_tool_call_guardrail_runtime.py -q
python -m pytest tests\agent\test_model_metadata.py tests\agent\test_openrouter_response_cache.py::TestDefaultConfig -q
```

Latest focused results:

- Registry/toolsets/state: `271 passed`
- TUI MCP fingerprint + model metadata local context: `27 passed`
- Delegation + timeout diagnostics: `128 passed`
- Concurrent tool execution + guardrails + interrupts: `40 passed`
- Model metadata + OpenRouter default config: `99 passed`

Total focused regression count in this pass:

- `565 passed`

This was a targeted hot-path regression run, not the full repository suite.

## Risks and Mitigations

- Risk: stale built-in tool discovery cache.
  Mitigation: fingerprint uses cache version, absolute path, file names, sizes,
  and nanosecond mtimes.

- Risk: stale toolset results after registry/plugin mutation.
  Mitigation: cache key includes registry identity and `_generation`; custom
  toolset creation clears caches.

- Risk: batch persistence changes message order or counters.
  Mitigation: batch path preserves ordering and uses the same serialization
  semantics as the one-message path.

- Risk: TUI misses a needed MCP reload.
  Mitigation: missing fingerprint fails open and reloads MCP.

- Risk: local endpoint fast-fail is too broad.
  Mitigation: it only applies to numeric loopback endpoints and uses a short
  negative cache TTL.

- Risk: parallel guard fast path changes behavior.
  Mitigation: added focused tests for path normalization and cached-arg reuse.

- Risk: stale OpenRouter model metadata.
  Mitigation: cache is versioned, TTL-bound for normal reads, bypassed by
  `force_refresh=True`, and stale reads are only used after refresh failure.

## Reviewer Guide

Suggested review order:

1. `tools/registry.py`
2. `toolsets.py`
3. `hermes_state.py` and `run_agent.py`
4. `tui_gateway/server.py` and `ui-tui/src/app/useConfigSync.ts`
5. `agent/model_metadata.py`
6. `tools/delegate_tool.py`
7. `scripts/benchmark_startup_perf.py` and `scripts/benchmark_runtime_usage.py`

## Follow-Up Work

This PR does not attempt to implement:

- full tool schema manifests that avoid importing tool modules entirely
- plugin manifest disk caching
- external skill snapshot caching
- adaptive `/goal` cadence
- CI performance budgets
- broader runtime telemetry for session log rewrite and queue wait

Those remain good follow-up optimization targets once this safer hot-path pass
lands.

## Supporting Docs

- [Regression log and next-upstream-release playbook](https://github.com/wesleysimplicio/hermes-agent/blob/codex/hermes-agent-10x-fast/docs/hermes-10x-fast-regression-log.md)
- [Detailed implementation notes](https://github.com/wesleysimplicio/hermes-agent/blob/codex/hermes-agent-10x-fast/docs/hermes-agent-10x-fast-pr.md)
- [Runtime research mapping](https://github.com/wesleysimplicio/hermes-agent/blob/codex/hermes-agent-10x-fast/docs/runtime-performance-investigation-2026-05-15.md)

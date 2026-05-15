# Upstream PR Draft - Hermes Hot Path Performance

Date: 2026-05-15

Suggested title:

```text
perf: cache tool hot paths and speed up runtime probes
```

Suggested create command:

```powershell
gh pr create --repo NousResearch/hermes-agent --base main --head wesleysimplicio:codex/hermes-agent-10x-fast --title "perf: cache tool hot paths and speed up runtime probes" --body-file docs/hermes-performance-upstream-pr.md
```

## Summary

This PR reduces avoidable startup, tool-discovery, session-persistence, TUI
MCP reload, and runtime local-endpoint probe overhead without changing the
public tool API.

It does six main things:

1. Keeps platform plugin imports out of normal `model_tools` startup unless a
   gateway/platform path needs them.
2. Makes built-in tool discovery cheaper with a lightweight register detector,
   a persistent fingerprint cache, and adaptive parallel source scanning for
   large future tool directories.
3. Memoizes recursive toolset resolution by registry object and generation.
4. Batches completed-turn session message writes into one SQLite transaction.
5. Makes the TUI reload MCP tools only when the `mcp_servers` config
   fingerprint changes.
6. Makes dead numeric loopback endpoints fail fast during context-length
   discovery, which avoids repeated HTTP connect timeouts when local/custom
   providers are down.

The branch also adds reproducible startup/runtime benchmark harnesses, visual
PR documentation, and focused regression tests for the new cache, batch,
fingerprint, and endpoint-fast-path behavior.

Image rule for this PR: every generated or diagrammed image used in the README
and PR documentation must show or be mapped to old behavior, new behavior, and
the measured gain for that item. The canonical mapping is documented in
`docs/hermes-agent-10x-fast-pr.md` under "Image Comparison Contract".

## Problem

Several hot paths were doing repeated work:

- `model_tools` startup loaded bundled platform plugins even for plain tool
  schema setup.
- Built-in tool discovery scanned every tool source file each process startup.
- Toolset resolution repeatedly walked nested toolset includes and plugin
  aliases even when the registry had not changed.
- `_flush_messages_to_session_db()` wrote one message at a time, causing one
  SQLite write path and counter update per message.
- The TUI mtime poller called `reload.mcp` for any config change, including
  display/voice edits that do not affect MCP tools.
- Agent/subagent initialization could spend tens of seconds probing a dead
  local custom endpoint for context metadata before the first model call.

These costs are especially visible in fresh CLI/TUI/gateway processes and in
tool-heavy or delegation-heavy agent turns.

## Implementation

### Startup and Plugin Imports

- `hermes_cli/plugins.py` supports `discover_and_load(include_platforms=False)`.
- `model_tools.py` uses the fast path so normal tool-schema setup does not
  import gateway/platform stacks.
- Platform plugins can still be loaded later through the full discovery path.

### Built-In Tool Discovery

- `tools/registry.py` replaces per-file AST parsing with a top-level
  `registry.register(...)` regex detector.
- Built-in discovery now stores a module list cache under Hermes' profile-aware
  cache directory.
- The cache key includes:
  - cache format version
  - absolute `tools/` path
  - each candidate file name
  - each candidate file size
  - each candidate file `st_mtime_ns`
- Cache misses rebuild the module list and refresh the cache.
- Source scanning can use `ThreadPoolExecutor` for large directories, but only
  after a size threshold. Local benchmarking showed the current Hermes `tools/`
  directory is small enough that sequential scanning is faster.
- Tool module imports remain ordered and serial to preserve registration side
  effects.

### Toolset Resolution Cache

- `toolsets.py` now memoizes:
  - `resolve_toolset(name)`
  - `get_all_toolsets()`
  - `get_toolset_names()`
- The cache key is `(id(registry), registry._generation)`, so tests/plugins that
  replace or mutate the registry invalidate safely.
- `create_custom_toolset()` clears the cache.

### SQLite Batch Persistence

- `SessionDB.append_messages(session_id, messages)` inserts multiple messages
  in one write transaction.
- It preserves the same serialization logic as `append_message()` for:
  - structured content
  - `tool_calls`
  - reasoning fields
  - Codex reasoning/message items
- It updates `message_count` and `tool_call_count` once per batch.
- `AIAgent._flush_messages_to_session_db()` builds pending message records and
  uses `append_messages()` when available, with a per-message fallback for
  compatibility.

### TUI MCP Fingerprint

- `tui_gateway/server.py` now includes `mcp_fingerprint` in `config.get mtime`.
- The fingerprint is a stable JSON serialization of the `mcp_servers` config
  section.
- `ui-tui/src/app/useConfigSync.ts` still hydrates full config on any mtime
  change, but calls `reload.mcp` only if the MCP fingerprint changed.
- If the fingerprint is unavailable, the TUI fails open and reloads MCP rather
  than risking stale tools.

### Runtime Local Endpoint Fast Path

- `agent/model_metadata.py` now performs a narrow TCP reachability preflight
  for numeric loopback endpoints before expensive HTTP metadata probes.
- If a local custom endpoint is closed, Hermes caches that negative
  reachability result briefly and falls back to the existing default context
  length immediately.
- Hostname/private-LAN/remote endpoints keep the previous HTTP metadata
  behavior, preserving LM Studio/Ollama/vLLM discovery semantics.
- Regression tests cover `detect_local_server_type()`,
  `fetch_endpoint_model_metadata()`, and `get_model_context_length()` so dead
  loopback endpoints skip both `httpx.Client` and `requests.get`.

### Benchmark Harness

`scripts/benchmark_startup_perf.py` measures fresh subprocess timings for:

- `import model_tools`
- `import model_tools` + `get_tool_definitions`
- warm/cold `get_tool_definitions`
- fast/full plugin discovery
- adaptive source scanning
- cached toolset resolution
- looped vs batched session message inserts

`scripts/benchmark_runtime_usage.py` measures runtime hot paths without model
API calls:

- agent initialization with focused/default tools
- delegated child construction
- mocked `delegate_task` scheduling
- parallel tool-call execution
- no-op tool dispatch overhead
- parallel safety checks
- looped vs batched session message inserts

The runtime pass now also instruments `delegate_task` phase timing and verifies
that a delegated batch reuses one loaded config snapshot for child build/run
settings.

## Benchmarks

Command:

```powershell
python scripts\benchmark_startup_perf.py -n 7
```

Phase 1 baseline was measured from detached `main` at
`a1c316c6f664fa507bb43ea8f91519b390ed9f75` in a separate worktree.

| Case | main median | branch median | Speedup | Change |
| --- | ---: | ---: | ---: | ---: |
| `import_model_tools` | 2.0847s | 0.8419s | 2.48x | 59.6% faster |
| `import_and_get_tool_definitions` | 1.8782s | 0.8741s | 2.15x | 53.5% faster |
| `get_tool_definitions` | 0.0918s | 0.0898s | 1.02x | 2.2% faster |
| platform plugin discovery fast path | 0.5571s full baseline | 0.1930s fast path | 2.89x | 65.4% faster |

Later local Windows runs were noisier, so the added Phase 2 cases are more
useful as microbenchmarks:

```powershell
python scripts\benchmark_startup_perf.py -n 3
```

| Case | Median | Notes |
| --- | ---: | --- |
| `tool_discovery_source_scan_adaptive` | 0.0987s | `parallel_eligible=False`; adaptive guard kept local scan sequential |
| `resolve_toolset_cached` | 0.1610s cold | warm path about 0.000002s/call |
| `session_append_messages_batch` | 0.0240s batch | loop about 0.6329s, about 24.21x faster for 180 messages |

The SQLite batch-write result is the strongest Phase 2 win. Startup subprocess
benchmarks on Windows can vary with filesystem and antivirus activity, so the
PR treats those numbers as local measurements, not universal guarantees.

Runtime benchmark:

```powershell
python scripts\benchmark_runtime_usage.py -n 3
```

| Case | Median | Notes |
| --- | ---: | --- |
| `agent_init_file_terminal` | 5.5563s | 9.25x faster than dead-loopback preflight baseline 51.4181s |
| `agent_init_default_tools` | 5.2897s | 8.63x faster than dead-loopback preflight baseline 45.6670s |
| `delegate_child_build` | 5.0907s | 9.02x faster than dead-loopback preflight baseline 45.9254s |
| `delegate_task_batch_scheduler` | 0.3971s | mocked scheduler; `config_loads=1`; child run phase ~0.0535s |
| `parallel_tool_batch_sleep` | 0.0590s | 5.14x faster than sequential equivalent |
| `tool_dispatch_noop` | 0.0992s | 0.0308ms per dispatch over 3000 calls |
| `parallel_guard_read_files` | 1.6403s | 0.1673ms per 8-tool safety decision; 4.26x lower median than prior 6.9878s |
| `session_append_messages_batch` | 0.0192s | 22.10x faster than loop writes for 240 messages |

The runtime 10x-class result is scoped to dead local/custom endpoint
initialization. It fixes a real "Hermes appears stuck before the first model
call" failure mode, especially visible when subagents inherit a down local
endpoint.

## Validation

Commands that passed locally:

```powershell
python -m py_compile hermes_cli\plugins.py model_tools.py tools\registry.py tools\browser_tool.py tools\tts_tool.py tools\yuanbao_tools.py scripts\benchmark_startup_perf.py
python -m py_compile tools\registry.py toolsets.py hermes_state.py run_agent.py tui_gateway\server.py scripts\benchmark_startup_perf.py

python -m pytest tests\hermes_cli\test_plugins.py tests\tools\test_registry.py -q -k "platform_plugins_can_be_deferred_then_loaded or imports_only_self_registering_modules or ignores_indented_register_calls or skips_mcp_tool"
python -m pytest tests\tools\test_browser_cloud_fallback.py tests\tools\test_browser_cdp_override.py tests\tools\test_browser_content_none_guard.py -q
python -m pytest tests\tools\test_tts_gemini.py tests\tools\test_tts_mistral.py tests\tools\test_tts_piper.py tests\tools\test_tts_dotenv_fallback.py -q
python -m pytest tests\tools\test_yuanbao_tools.py -q

python -m pytest tests\tools\test_registry.py tests\test_toolsets.py tests\test_hermes_state.py -q
python -m pytest tests\test_tui_gateway_server.py::test_config_get_mtime_includes_mcp_fingerprint tests\test_tui_gateway_server.py::test_mcp_config_fingerprint_treats_missing_section_as_empty -q

python -m py_compile agent\model_metadata.py scripts\benchmark_runtime_usage.py
python -m py_compile run_agent.py tools\delegate_tool.py scripts\benchmark_runtime_usage.py
python -m pytest tests\agent\test_model_metadata_local_ctx.py -q
python -m pytest tests\tools\test_delegate.py tests\tools\test_delegate_subagent_timeout_diagnostic.py -q
python -m pytest tests\run_agent\test_run_agent.py::TestConcurrentToolExecution tests\run_agent\test_run_agent.py::TestParallelScopePathNormalization -q
python scripts\benchmark_runtime_usage.py -n 3

cd ui-tui
npm ci
npm test -- useConfigSync.test.ts
npm run type-check
```

Latest focused results:

- `tests/tools/test_registry.py tests/test_toolsets.py tests/test_hermes_state.py`: 271 passed.
- Gateway fingerprint tests: 2 passed.
- Model metadata local context tests: 25 passed.
- `ui-tui` `useConfigSync.test.ts`: 33 passed.
- `ui-tui` type-check: passed.

Known local test environment note:

- `python -m pytest tests\test_tui_gateway_server.py -q -n 0` still has 7
  unrelated failures in this local Python environment because `prompt_toolkit`
  is unavailable to the test process. The new fingerprint tests pass
  independently.

## Reviewer Guide

Recommended review order:

1. `tools/registry.py`
   - Confirm cache fingerprinting invalidates on file changes.
   - Confirm imports remain serial and ordered.
   - Confirm adaptive parallelism is gated by size and does not affect current
     small directories.

2. `toolsets.py`
   - Confirm cache keys include registry identity and generation.
   - Confirm custom toolsets clear memoized results.

3. `hermes_state.py` and `run_agent.py`
   - Compare `append_messages()` field serialization with `append_message()`.
   - Confirm message ordering and session counters are preserved.
   - Confirm fallback to `append_message()` remains available.

4. `tui_gateway/server.py` and `ui-tui/src/app/useConfigSync.ts`
   - Confirm non-MCP config edits still hydrate UI state.
   - Confirm MCP reload happens only when the fingerprint changes or is missing.

5. `scripts/benchmark_startup_perf.py`
   - Confirm benchmark cases are small, local, and do not require network.

6. `agent/model_metadata.py`
   - Confirm numeric loopback fast-fail is narrow.
   - Confirm hostname/private-LAN endpoints still use existing metadata probes.
   - Confirm the fallback context length remains unchanged when probes are down.

7. `scripts/benchmark_runtime_usage.py`
   - Confirm runtime cases avoid model API calls and isolate Hermes home state
     per subprocess.

## Risk Assessment

Risk: stale built-in tool discovery cache.

Mitigation: cache version, absolute path, file names, sizes, and nanosecond
mtimes are checked before use. Cache misses rebuild from source.

Risk: stale toolset resolution after plugin or registry changes.

Mitigation: cache key includes registry identity and `_generation`; tests cover
registry mutation. Custom toolsets clear the cache.

Risk: SQLite batch write changes message ordering or counters.

Mitigation: batch inserts preserve message order, apply monotonic timestamps
within the batch, and update counters with the same tool-call counting semantics
as `append_message()`. Tests cover message order and counter behavior.

Risk: TUI misses an MCP reload.

Mitigation: missing/invalid fingerprints fail open and trigger reload. Backend
fingerprint is stable JSON of `mcp_servers`.

Risk: parallel scanning changes registration side effects.

Mitigation: only source detection is parallel. Module imports remain serial and
ordered.

Risk: local endpoint preflight skips a server that is about to start.

Mitigation: the fast path only applies to numeric loopback endpoints, caches
negative reachability for a short 30-second TTL, and falls back to the same
default context length Hermes already used when metadata probing failed.

## Rollback Plan

Each optimization can be reverted independently:

- Disable persistent tool discovery cache by removing cache read/write calls in
  `discover_builtin_tools()`.
- Disable toolset memoization by bypassing `_RESOLVED_TOOLSET_CACHE`,
  `_ALL_TOOLSETS_CACHE`, and `_TOOLSET_NAMES_CACHE`.
- Disable SQLite batch writes by making `_flush_messages_to_session_db()` call
  `append_message()` in the loop again.
- Disable TUI MCP fingerprinting by restoring unconditional `reload.mcp` on
  config mtime changes.
- Disable adaptive parallel scanning by forcing
  `_should_parallel_scan_tool_sources()` to return `False`.
- Disable local endpoint preflight by making `_local_endpoint_reachable()` return
  `True`.

## Out Of Scope

This PR does not implement:

- a full tool schema manifest that avoids importing tool modules entirely
- plugin manifest disk caching
- skill snapshot caching for external skill dirs
- denormalized session-list preview fields
- adaptive `/goal` judge cadence
- CI performance budgets
- full runtime telemetry for message build, JSON log rewrite, and delegation
  phase timings

Those are good follow-up PRs after this safer hot-path pass lands.

## Visual Explainers

The branch includes diagrams under `docs/assets/10x-fast/`:

- `phase-2-tool-discovery-cache.svg`
- `phase-3-toolset-cache.svg`
- `phase-4-sqlite-batch-writes.svg`
- `phase-5-tui-mcp-fingerprint.svg`
- `phase-6-adaptive-parallel-scan.svg`
- `runtime-local-endpoint-fast-path.svg`
- `runtime-benchmark-suite.svg`
- `phase-7-delegate-parallel-guard.svg`
- `research-principles-map.svg`
- generated PNGs under `docs/assets/10x-fast/generated/`
  including `macro-original-vs-10x-fast.png` for public repository promotion.

They are documentation aids only; runtime behavior is covered by tests and the
benchmark harness.

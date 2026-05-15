# Hermes 100X Fast Regression Log

Date: 2026-05-15
Branch: `codex/hermes-agent-100x-fast`

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
  `docs/assets/100x-fast/generated/macro-original-vs-100x-fast.png`.
- Added deterministic SVG comparisons for each measured item.
- Added `runtime-openrouter-metadata-cache.svg` for the model metadata
  offline-cache comparison.
- Added three GPT-image generated comparison images for README/social use.
- Added a Remotion 90-second launch video with generated stereo soundtrack:
  `docs/assets/100x-fast/video/hermes-100x-fast-launch.mp4`.
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

```powershell
cd docs\remotion\100x-fast
npm run audio
npx tsc --noEmit
npm run still
npm run render
```

Result: Remotion typecheck passed, poster rendered, and a 90-second H.264/AAC
MP4 rendered at `docs/assets/100x-fast/video/hermes-100x-fast-launch.mp4`.

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

## Media Refresh Validation

After adding the GPT-image comparison images and the 90-second Remotion video,
the following checks were rerun on 2026-05-15:

```powershell
npx tsc --noEmit
```

Result: Remotion TypeScript check passed.

```powershell
python -m py_compile agent\model_metadata.py hermes_cli\config.py scripts\benchmark_runtime_usage.py
```

Result: passed with no compile errors.

```powershell
python -m pytest tests\agent\test_model_metadata_local_ctx.py tests\agent\test_model_metadata.py tests\agent\test_openrouter_response_cache.py::TestDefaultConfig -q
```

Result: `124 passed`.

```powershell
python -m pytest tests\tools\test_delegate.py tests\tools\test_delegate_subagent_timeout_diagnostic.py -q
```

Result: `128 passed`.

```powershell
python -m pytest tests\run_agent\test_run_agent.py::TestConcurrentToolExecution tests\run_agent\test_run_agent.py::TestParallelScopePathNormalization tests\run_agent\test_tool_executor_contextvar_propagation.py tests\run_agent\test_concurrent_interrupt.py tests\run_agent\test_tool_call_guardrail_runtime.py -q
```

Result: `40 passed`.

```powershell
python scripts\benchmark_runtime_usage.py --case openrouter_metadata_disk_cache --samples 5
```

Result: median `0.4211s` for 100 cold memory resets over 500 models; sample
reported ~`7.4728ms` per disk lookup and no network use.

```powershell
python scripts\benchmark_runtime_usage.py -n 1
python scripts\benchmark_startup_perf.py -n 1
```

Result: benchmark suites completed. Notable current samples:

- `parallel_tool_batch_sleep`: `5.20x` over sequential equivalent.
- `session_append_messages_batch`: `37.74x` in runtime benchmark and `35.96x`
  in startup benchmark.
- `resolve_toolset_cached`: warm path around `0.000004s`.
- `openrouter_metadata_disk_cache`: `0.4955s` for 100 cold memory resets in
  the full runtime benchmark sample.

## Known Limits

- The focused regression suite remains the highest-signal validation for the
  changed performance paths.
- A full local suite run was completed on Windows/Python 3.14, but it is not a
  CI-equivalent result because the repo's normal test wrapper expects a POSIX
  `.venv` flow and Hermes CI runs on Ubuntu with Python 3.11.
- Startup subprocess timings vary on Windows with filesystem cache and
  antivirus activity. Treat benchmark medians as local measurements.
- The 100x branding is scoped to specific avoided-wait/runtime bottlenecks and
  the repo's optimization track, not every Hermes operation.
- `docs/contribution_scout/` is intentionally left untracked and untouched.

## Next Upstream Version Playbook

The detailed reapply guide lives in
`docs/hermes-100x-fast-reapply-playbook.md`. Use that document as the
source of truth when porting this branch onto a new upstream Hermes release.

When NousResearch publishes a new Hermes release or important upstream commits:

1. Fetch upstream and create a fresh `codex/hermes-agent-100x-fast-*` branch.
2. Compare changed files against this branch, especially:
   `model_tools.py`, `tools/registry.py`, `toolsets.py`, `hermes_state.py`,
   `run_agent.py`, `agent/model_metadata.py`, `tools/delegate_tool.py`,
   `tui_gateway/server.py`, and `ui-tui/src/app/useConfigSync.ts`.
3. Re-apply only the optimizations that upstream does not already contain.
4. Re-run the focused regression suite listed above.
5. Re-run both benchmark scripts and update README/PR visuals with old, new,
   and gain for every image.
6. Keep new performance images in `docs/assets/100x-fast/` and ensure each one
   has tags plus old/new/gain in the README gallery.
7. Open or update the PR with exact benchmark numbers, not broad claims.

Minimum carry-forward checklist:

- Start from `upstream/main`, not from the old performance branch.
- Use `origin/codex/hermes-agent-100x-fast` as the reference branch.
- Port one optimization group at a time and run its focused tests immediately.
- Preserve profile-aware cache paths via `get_hermes_home()`.
- Keep `force_refresh` and fail-open/fallback behavior intact.
- Update README, PR docs, benchmark tables, and image paths in the same commit
  as any visual rename or measurement change.

## Current PRs

- Upstream draft PR: https://github.com/NousResearch/hermes-agent/pull/26129
- User fork PR: https://github.com/wesleysimplicio/hermes-agent/pull/1

## Post-Merge Continuation Cycle

Date: 2026-05-15

After syncing the performance branch with `upstream/main`, the continuation
loop now uses this default cadence:

1. Run a short `codex exec --enable goals "/goal ..."` pass to summarize the
   current PR state and next objective.
2. Use a parallel read-only agent for the next performance-scouting pass when
   agent slots are available.
3. Pick the smallest high-signal optimization from that scout report.
4. Implement it with focused tests before broadening the regression surface.
5. Update this log and the PR narrative before committing.

Latest `/goal` result:

- PR #26129 is mergeable but still draft/blocked on GitHub state.
- `git diff --check upstream/main...HEAD` passed.
- `docs/contribution_scout/` remains intentionally untracked and outside this
  PR scope.
- The Codex CLI subprocess could not see the Python/venv environment, so its
  local test warning was treated as an environment limitation, not as a Hermes
  regression.

Parallel scout selected two low-risk runtime follow-ups for immediate action:

- Reuse parsed tool-call args for mixed safe concurrent batches, not only
  all-`read_file` batches.
- Apply `normcase` to write/patch parallel scope paths so case-insensitive
  filesystems do not accidentally parallelize the same target with different
  casing.

Implemented follow-up:

- `run_agent.py` now routes the general parallel-safety path through
  `_parse_parallel_guard_args()`, so `_execute_tool_calls_concurrent()` can
  reuse cached args for mixed safe batches.
- `_extract_parallel_scope_path()` now applies `os.path.normcase()` after
  absolute path normalization for write/patch overlap checks.
- Added regression tests for mixed safe arg reuse and case-normalized path
  overlap.

Validation:

```powershell
python -m py_compile run_agent.py tests\run_agent\test_run_agent.py
```

Result: passed.

```powershell
python -m pytest tests\run_agent\test_run_agent.py::TestConcurrentToolExecution::test_concurrent_reuses_parallel_guard_parsed_args tests\run_agent\test_run_agent.py::TestConcurrentToolExecution::test_concurrent_reuses_guard_args_for_mixed_safe_batches tests\run_agent\test_run_agent.py::TestParallelScopePathNormalization -q
```

Result: `6 passed`.

Broader concurrent-tool guardrail validation:

```powershell
python -m pytest tests\run_agent\test_run_agent.py::TestConcurrentToolExecution tests\run_agent\test_run_agent.py::TestParallelScopePathNormalization tests\run_agent\test_tool_executor_contextvar_propagation.py tests\run_agent\test_concurrent_interrupt.py tests\run_agent\test_tool_call_guardrail_runtime.py -q
```

Result: `42 passed`.

## CI Follow-Up: Discord E2E Mock Compatibility

Date: 2026-05-15

After pushing the post-merge continuation commit, GitHub Actions surfaced an
`e2e` failure in `tests/e2e/test_discord_adapter.py`. The failure was not in
the performance code; the Discord e2e mock exposes `discord.Forbidden` as a
mock object rather than a real exception class, so `_fetch_channel_context()`
raised `TypeError` while trying to catch it after a mocked channel without
`history()` failed.

Implemented follow-up:

- `gateway/platforms/discord.py` now detects whether `discord.Forbidden` is a
  real exception class inside the generic exception path before treating it as
  a missing-permission case.
- Real Discord behavior is preserved; mocked Discord modules no longer crash
  the e2e command-dispatch path.

Validation:

```powershell
python -m py_compile gateway\platforms\discord.py
```

Result: passed.

```powershell
python -m pytest tests\e2e\test_discord_adapter.py::TestMentionStrippedCommandDispatch::test_mention_then_command tests\e2e\test_discord_adapter.py::TestMentionStrippedCommandDispatch::test_nickname_mention_then_command tests\e2e\test_discord_adapter.py::TestMentionStrippedCommandDispatch::test_text_before_command_not_detected tests\e2e\test_discord_adapter.py::TestAutoThreadingPreservesCommand::test_command_detected_after_auto_thread -q
```

Result: `4 passed`.

```powershell
python -m pytest tests\e2e -q
```

Result: `56 passed, 7 skipped`.

## CI Follow-Up: Setup, Browser Cloud, and Provider Parity

Date: 2026-05-15

After the Discord fix, the GitHub Actions `test` job exposed thirteen focused
failures outside the runtime parallelism changes:

- setup/gateway tests tried to enter an interactive checklist in captured CI
  output or misread plugin-disabled platforms as configured;
- browser cloud provider cache tests expected monkeypatchable provider
  factories;
- Nous provider-parity tests instantiated an empty model and should not probe
  a live endpoint for minimum-context validation;
- transcription dotenv fallback tests were revalidated as part of the same
  failure batch.

Implemented follow-up:

- `setup_gateway()` now continues past an empty checklist when configured
  platforms were pre-selected, avoiding captured-stdin prompts while preserving
  the existing user configuration path.
- Gateway migration summaries now distinguish real user configuration from
  `plugin disabled` sentinels, and they still recognise built-in platform env
  markers such as Matrix even when the local setup menu hides that platform on
  Windows.
- `tools/browser_tool.py` restores small provider factory wrappers plus a
  registry so tests and plugins can monkeypatch provider construction without
  importing optional provider packages eagerly.
- Browser cloud auto-detection now attempts Browser Use and Browserbase
  providers directly and only caches a positive configured provider.
- `get_model_context_length()` returns the default fallback context for an
  empty model string instead of probing a configured provider endpoint.

Validation:

```powershell
python -m py_compile tools\browser_tool.py agent\model_metadata.py hermes_cli\setup.py hermes_cli\gateway.py
```

Result: passed.

```powershell
python -m pytest tests\tools\test_browser_cloud_provider_cache.py tests\tools\test_browser_cloud_fallback.py -q
```

Result: `14 passed`.

```powershell
python -m pytest tests\run_agent\test_provider_parity.py::TestDeveloperRoleSwap::test_developer_role_via_nous_portal tests\run_agent\test_provider_parity.py::TestBuildApiKwargsNousPortal::test_includes_nous_product_tags tests\run_agent\test_provider_parity.py::TestBuildApiKwargsNousPortal::test_uses_chat_completions_format -q
```

Result: `3 passed`.

```powershell
python -m pytest tests\tools\test_transcription_dotenv_fallback.py -q
```

Result: `9 passed`.

```powershell
python -m pytest tests\hermes_cli\test_setup.py tests\hermes_cli\test_setup_irc.py tests\hermes_cli\test_setup_openclaw_migration.py -q
```

Result: `60 passed, 2 warnings`.

```powershell
python -m pytest tests\run_agent\test_provider_parity.py tests\tools\test_transcription_dotenv_fallback.py tests\tools\test_browser_cloud_provider_cache.py tests\tools\test_browser_cloud_fallback.py -q
```

Result: `116 passed`.

Operational note: `codex exec --enable goals` was attempted for this cycle, but
the local Codex CLI reported a usage-limit error. Work continued locally with
the same staged objective loop.

## CI Follow-Up: Fork Audit Noise

Date: 2026-05-15

The fork-side CI also reported two non-runtime failures after the regression
fix commit:

- contributor attribution detected older upstream author emails that were not
  present as exact keys in `AUTHOR_MAP`;
- the supply-chain scanner flagged `hermes_cli/setup.py` as if it were a
  package install hook named `setup.py`.

Implemented follow-up:

- Added exact author mappings for the fork-reported emails:
  `amethystani@users.noreply.github.com`, `nightcityblade@gmail.com`, and
  `robin@soal.org`.
- Narrowed the install-hook scanner so root-level `setup.py` is still critical,
  while non-root application modules such as `hermes_cli/setup.py` are not
  treated as packaging install hooks.

## CI Follow-Up: Full Test Job Residuals

Date: 2026-05-15

After the audit jobs were green, the upstream `test` job exposed ten residual
failures:

- non-interactive gateway setup still fell through to an input prompt under
  captured pytest output;
- Matrix needed to stay hidden from the Windows setup picker while still being
  recognised as existing configuration for migration/service summaries;
- plugin platform status treated installed dependencies as configured state;
- systemd PATH generation could raise on inaccessible user-local directories;
- xAI STT credential resolution could hold a stale imported config helper in a
  long-lived test process.

Implemented follow-up:

- `curses_checklist()` now returns the cancel/default selection whenever stdin
  or stdout is non-interactive, and avoids the numbered input fallback under
  pytest capture.
- Gateway setup summaries and service checks now use setup-visible platforms
  plus built-in platforms hidden by host gating, so existing Matrix config can
  be counted without showing Matrix in the Windows picker.
- Plugin platform status now treats `is_connected`/required env vars as
  configuration state, not dependency-only `check_fn()` success.
- User-local PATH probing ignores inaccessible directories instead of raising.
- `tools.xai_http.get_env_value()` resolves the live config helper at call
  time, matching the STT dotenv fallback contract.

Validation:

```powershell
python -m py_compile hermes_cli\setup.py hermes_cli\gateway.py hermes_cli\curses_ui.py tools\xai_http.py
```

Result: passed.

```powershell
python -m pytest tests\hermes_cli\test_setup.py tests\hermes_cli\test_setup_irc.py tests\hermes_cli\test_setup_openclaw_migration.py tests\hermes_cli\test_gateway_platform_gating.py tests\hermes_cli\test_gateway_service.py tests\tools\test_transcription_dotenv_fallback.py -q
```

Result: `73 passed, 1 skipped, 2 warnings`.

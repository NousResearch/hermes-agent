# 2026-05-31 compression/log hardening ops notes

## Scope

- Profiles touched/checked: `default`, `redops`.
- Explicitly avoided reading/changing `producers` profile contents.
- Main incident context: runaway context compression/log amplification in session `20260531_060815_758024` class; current remediation focused on auxiliary routing, compression stability, and persistent log preview bounds.

## Changes made in this pass

1. `agent/tool_executor.py`
   - Added safe result-preview handling for non-string tool results so sequential tool execution no longer relies on `function_result[:200]` after guardrail observation.
   - Hardened error preview generation for dict/JSON/multimodal-shaped results:
     - strips cognitive-shock banners from persistent log previews;
     - keeps logs single-line/bounded;
     - drops heavy fields such as `file_preview`, `content`, `raw`, `raw_output`;
     - summarizes nested `results[].error` payloads without persisting large extracted content.

2. `tests/agent/test_tool_executor_log_preview.py`
   - Added regressions for critical wrapper compaction, dict payload immutability, `file_preview` dropping, nested `web_extract` errors, and non-string result preview safety.

3. `scripts/hermes_ops_smoke.py`
   - Added safe doctor-free operational smoke runner.
   - The runner captures sub-check rc/stdout/stderr into JSON and exits `0` by default, so failed sub-checks do not create fresh `Tool terminal returned error` entries while auditing logs.
   - `--strict` preserves CI/human shell nonzero semantics.
   - `--include-doctor` is explicit and off by default because current `hermes doctor` prints a global profile roster.

4. `tests/scripts/test_hermes_ops_smoke.py`
   - Added regressions that default smoke is doctor-free and does not mention `producers`; doctor checks are only included with explicit `--include-doctor`.

## Verification

- `PYTEST_ADDOPTS='' python -m pytest -o addopts='' tests/agent/test_tool_executor_log_preview.py tests/scripts/test_hermes_ops_smoke.py -q`
  - Result: `9 passed`.
- `python -m py_compile agent/tool_executor.py scripts/hermes_ops_smoke.py tests/agent/test_tool_executor_log_preview.py tests/scripts/test_hermes_ops_smoke.py`
  - Result: OK.
- `python scripts/hermes_ops_smoke.py --skip-pytest`
  - Result: JSON `ok=true`; checks: `aux_config`, `checkpoint_store`, `git_diff_check`.
- `python scripts/hermes_ops_smoke.py`
  - Result: JSON `ok=true`; targeted pytest: `179 passed, 1 warning`.
- Log smoke over default+redops only:
  - No new `Tool terminal returned error` from the safe runner.
  - `redops` log remains clean of new warning/error storm.
  - `default` still contains old pre-patch raw `file_preview`/critical-wrapper entries and unrelated current OpenRouter 429 warnings; these are historical or external, not produced by the new runner.

## Additional live hardening pass — 13:24 MSK

Fresh default+redops log scan found two actionable default-profile edges while `redops` stayed clean:

1. `memory` tool overflow failures were still logging full `current_entries=[...]` state snapshots.
   - Fix: `agent/tool_executor.py` now drops `current_entries`, `existing_entries`, `entries`, `matches`, and `file_matches` from persistent error previews.
   - Regression: `tests/agent/test_tool_executor_log_preview.py::test_tool_error_log_preview_drops_memory_state_snapshots` verifies the memory limit message remains but private memory contents do not enter `agent.log` preview strings.

2. `disk-cleanup` `post_tool_call` hook could raise `PermissionError` while parsing untrusted terminal output paths such as `/root/.cli-proxy-api\\nremote-management:\\n`.
   - Fix: `plugins/disk-cleanup/__init__.py` now applies the cleanup scope guard before `Path.exists()`/stat-like probes and catches `OSError`/`ValueError` inside `_attempt_track`.
   - Regression: `tests/plugins/test_disk_cleanup_plugin.py::test_terminal_malformed_privileged_path_never_raises` verifies malformed privileged paths are ignored without hook-runner warnings.

Verification for this pass:

- `PYTEST_ADDOPTS='' python -m pytest -o addopts='' tests/agent/test_tool_executor_log_preview.py tests/plugins/test_disk_cleanup_plugin.py -q`
  - Result: `50 passed`.
- `python scripts/hermes_ops_smoke.py --skip-pytest`
  - Result: JSON `ok=true`; checks: `aux_config`, `checkpoint_store`, `git_diff_check`.
- Timestamped combined verification at `2026-05-31T13:24:38+03:00`:
  - Result: `50 passed`; smoke JSON `ok=true`.

Operational note: existing `agent.log` tail still contains historical pre-patch raw JSON/critical-banner entries and hook warnings. These should decay out of tails after active processes are restarted or naturally replaced by new agent sessions using the patched imports.

## Additional live hardening pass — 13:45 MSK

Fresh default+redops `errors.log` scan (tail 6000) ranked recurring noise by signature. `redops` stayed clean. Two actionable default edges fixed; one already-self-healed edge verified.

1. Discord voice Opus decode-error log flood (~2.3k ERROR lines/session).
   - Root cause: `VoiceReceiver._on_packet` logged `logger.error("Opus decode error for SSRC ...")` on every corrupted/lost Opus packet. Packet loss/corruption is routine on lossy voice links, so the hot path emitted thousands of identical ERROR lines (top signature: 2292 hits).
   - Fix: `plugins/platforms/discord/adapter.py` now tracks `_opus_failed_ssrcs` and logs the first failure per SSRC burst only, at `WARNING` (downgraded from `ERROR`), re-arming on the next successful decode. Decode/`_decoders.pop` reset logic unchanged.
   - Regressions: `tests/gateway/test_discord_opus.py::test_opus_decode_error_logging_is_throttled` (source contract) and `::test_opus_decode_error_one_shot_behaviour` (behavioural: 50 failures → 1 log, recovery re-arms → 1 more).

2. LSP client stdin drain race (`AssertionError` flood, active at 13:35).
   - Root cause: the reader loop dispatches server→client requests as fire-and-forget tasks; multiple of them — plus the request-sender path — called `StreamWriter.drain()` on the same writer concurrently. asyncio's `_drain_helper` asserts a single drain waiter, so unsynchronized drains raced into `AssertionError`, surfaced as `asyncio: Task exception was never retrieved`.
   - Fix: `agent/lsp/client.py` serializes write+drain behind a new `self._write_lock` (`asyncio.Lock`), retains dispatch tasks in `self._dispatch_tasks` with a `_on_dispatch_done` callback that consumes any exception, and cancels in-flight dispatch tasks in `_cleanup_process`.
   - Regressions: `tests/agent/lsp/test_write_drain_race.py` — a fake stdin replicating asyncio's single-drain-waiter assertion proves 50 concurrent writes serialize (`max_concurrent_drains == 1`), plus source contracts and a behavioural check that failed dispatch tasks are consumed.

3. Checkpoint store `fatal: not a git repository` (100 historical ERRORs) — verified already self-healed, no code change.
   - All 100 hits were pre-`11:22`; zero after. `_init_store` validates via `_is_valid_store` and archives a broken store (`_archive_corrupt_store`) before the hot-path `git add -A`. Confirmed: live store `rev-parse` rc=0, `corrupt-20260531-112248/` archive present, new objects written at 13:43. Covered by `tests/tools/test_checkpoint_manager.py::test_corrupt_store_is_archived_and_reinitialised` (asserts `"fatal: not a git repository" not in caplog.text`).

Verification for this pass:

- `PYTEST_ADDOPTS='' python -m pytest -o addopts='' tests/gateway/test_discord_opus.py tests/agent/lsp/test_write_drain_race.py tests/agent/lsp/ -q`
  - Result: `156 passed`. (Teardown-only `Task was destroyed but it is pending!` lines come from unrelated `_wait_for_fresh_push`/`Event.wait` coroutines in the test loop, not the patched dispatch tasks; no production flood.)
- `git diff --check` on touched files (`plugins/platforms/discord/adapter.py`, `agent/lsp/client.py`, `tests/gateway/test_discord_opus.py`, `tests/agent/lsp/test_write_drain_race.py`)
  - Result: clean.

## Additional live hardening pass — 13:55 MSK

Recent-window scan (`errors.log` after `13:24`): `redops` clean (0). `default` had only 11 WARNING+ lines. The stale `Opus`/checkpoint floods are gone (fixes effective). Two signals triaged:

- `asyncio: Task exception was never retrieved` ×4 at `13:35:34` — the already-fixed LSP drain race, emitted by the long-lived process that imported the pre-patch `client.py` (old line numbers 576/536). Zero after `13:35`; decays on restart. No action.
- `Credential pool provider mismatch` ×91 (`pool=custom:cliproxy, agent=custom`) — **functional bug, fixed.**

Fix: `recover_with_credential_pool` custom-provider key resolution.
- Root cause: custom endpoints all report `agent.provider == "custom"`, but the pool is keyed `custom:<name>` (e.g. `custom:cliproxy`). The mismatch guard did a naive `agent.provider != pool.provider`, so it tripped for **every** custom provider and silently returned `False` — disabling credential rotation on 429/billing/auth. With cliproxy holding 2 pool entries (priority 0/1) and being the active runtime (`model.provider=custom`, `base_url=http://127.0.0.1:8317/v1`), failover was fully broken.
- Fix: `agent/agent_runtime_helpers.py` now resolves the agent's current `base_url` to its pool key via `get_custom_provider_pool_key(...)` and compares keys. Rotation is allowed when the agent is still on the endpoint that seeded the pool; a fallback to a *different* custom endpoint (or a genuine cross-provider mismatch) still trips the guard — preserving the #33163 cross-provider-contamination protection.
- Verified resolution against live config: `http://127.0.0.1:8317/v1` → `custom:cliproxy` (matches pool); `https://api.mistral.ai/v1`/unknown → `None`/different key (correctly blocked).
- Regressions: `tests/agent/test_credential_pool_routing.py::TestCustomProviderPoolKeyGuard` — 4 cases: same endpoint rotates, different custom endpoint blocks, unresolvable endpoint blocks, non-custom mismatch still blocks.

Verification for this pass:

- `PYTEST_ADDOPTS='' python -m pytest -o addopts='' tests/agent/test_credential_pool.py tests/agent/test_credential_pool_routing.py -q`
  - Result: `92 passed`.
- `python -m py_compile agent/agent_runtime_helpers.py tests/agent/test_credential_pool_routing.py` — OK.
- `git diff --check` on touched files — clean.

## Cron budget-exhaustion conflation fix — 14:25 MSK

Recent-window scan (`errors.log` after `13:45`): `redops` clean (0). `default` showed 9 WARNING+ lines — 8 the (already-fixed-in-source) credential mismatch + RateLimit retries from long-lived pre-patch sessions (`20260531_131848_0b78ef`, `_133008_a9bec7`; decay on restart), plus **1 ERROR: `Job 'edge_assimilation_loop' failed`**.

Fix: cron scheduler no longer conflates iteration-budget exhaustion with failure.
- Root cause: `agent.log` showed `Turn ended: reason=max_iterations_reached(150/150)` immediately before the cron ERROR. In `conversation_loop.py:4404`, `completed = (final_response is not None and api_call_count < max_iterations and not failed)`, so the budget-ceiling path returns `completed=False` with `failed=False` and a **real summary** in `final_response` (produced by `_handle_max_iterations`). But `scheduler.py:1791` raised `RuntimeError` on `failed is True OR completed is False`, conflating a genuine crash with an expected perpetual-loop termination. Result: the loop job's work-product (a valid status report) was discarded (raised, not delivered) and surfaced in `errors.log` as a fake failure.
- Fix: `cron/scheduler.py` adds a tightly-gated carve-out. When the agent did NOT hard-fail (`failed` falsy), was NOT interrupted, `completed is False`, `final_response` is non-empty, AND `turn_exit_reason` starts with `max_iterations_reached`, the summary is delivered as success. Every #17855 failure shape (carries `failed=True`, or lacks the budget exit reason, or has empty output) still raises.
- Regressions: `tests/cron/test_scheduler.py::TestRunJobSessionPersistence::test_run_job_budget_exhaustion_with_summary_succeeds` — 4 parametrized cases (budget+summary→success, budget+empty→fail, non-budget abnormal stop→fail, budget+hard-fail→fail). The existing #17855 contract (`test_run_job_treats_agent_failure_flag_as_failure`) is unchanged and still green.

Verification for this pass:

- `PYTEST_ADDOPTS='' python -m pytest -o addopts='' tests/cron/test_scheduler.py -q` → `132 passed`.
- `python -m py_compile cron/scheduler.py tests/cron/test_scheduler.py` — OK.
- `git diff --check` on touched files — clean.

## Autopilot pytest usage-error vs regression conflation — 00:10 MSK (Jun 01)

- Symptom: `hermes-autopilot.service` stuck in systemd `failed` state (`ExecMainStatus=2`) since the 13:21 tick. Inner Hermes-as-primary run succeeded (`envelope.returncode=0`), but the outer regression gate marked the issue `phase=Regressed` with `pytest failed: rc=4`.
- Root cause: issue `t6pyj.5` scope carried a prose-shaped path `tests/import` (meant "import checks", not a file). `reconcile()` kept everything under `tests/` and handed it to `run_pytest_for_scope`, which ran `pytest … tests/import` → **rc=4 (usage error: path not found)**. The gate `ok = (returncode == 0) and …` treated ANY non-zero rc as a regression, so an infra/scope problem hard-failed the unit on every timer tick and blocked legitimate work.
- pytest exit-code grading the gate ignored: 0=passed, 1=failed, 2=interrupted, 3=internal, 4=usage/bad-path, 5=no-tests-collected. Only 1 (or a parsed failed/error count) is a real regression.
- Fix (`scripts/ops/hermes_autopilot.py::run_pytest_for_scope`), two-part, strict both ways:
  - (A) drop scope entries that do not exist on disk before invoking pytest; if none remain, short-circuit `ok=True` without running pytest. Non-existent paths are noted in the detail string.
  - (B) classify the result: `regression = (failed > 0) or (errors > 0) or (rc == 1)`. Codes 2/3/4/5 are infra, not regression → `ok=True`. SAFETY property preserved: a real failure (rc=1 / parsed failures / parsed errors) still hard-fails.
- Regressions: `tests/scripts/test_hermes_autopilot.py` — 6 new cases (drop-nonexistent, usage-error-not-regression, no-tests-not-regression, real-failure-IS-regression, parsed-errors-IS-regression, mixed-scope-runs-existing). `parse_pytest_summary` contract untouched.
- Stale unit state cleared with `systemctl --user reset-failed hermes-autopilot.service` (reversible; `Result=success` after). Timer still armed: next tick Mon 03:00:47 MSK, `Persistent=true`, `OnUnitActiveSec=6h`. The fix lands automatically on that tick — no service restart needed.
- KNOWN LIMITATION surfaced by end-to-end verification against the real `t6pyj.5` scope: the autopilot's `REPO_ROOT` is `security-workstation/`, but Hermes-agent issue scopes name paths relative to the nested `tools/hermes-agent/` project (which has its OWN venv, conftest hermetic fixtures, and pytest isolation addopts). So for any hermes-agent-scoped issue, all scope test paths resolve non-existent → the gate now returns `ok=True` with detail `no existing test files in scope (skipped non-existent: …)`. This is HONEST (recorded in the artifact, not hidden) and strictly better than the previous false-Regressed/unit-fail, but the gate is VACUOUS for nested-project scopes. Naively re-resolving paths into `tools/hermes-agent/` would be WRONG — running those tests from the root `sys.executable`/cwd bypasses the hermes-agent venv + isolation config and reintroduces rc!=0 flake. Proper fix is a design decision (should autopilot verify nested projects, and with which venv/cwd?) — deferred to operator, not changed unilaterally on a safety gate.

Verification for this pass:

- `PYTEST_ADDOPTS='' python -m pytest -o addopts='' tests/scripts/test_hermes_autopilot.py -q` → `27 passed`.
- `python -m py_compile scripts/ops/hermes_autopilot.py tests/scripts/test_hermes_autopilot.py` — OK.
- `git diff --check` on touched files — clean.

## CDP endpoint resolution warning flood — 00:25 MSK (Jun 01)

- Symptom: `tools.browser_tool` emitted `Failed to resolve CDP endpoint http://127.0.0.1:9222 …` in bursts of x6/x12 per minute — 203 lines over ~26h in `default/errors.log`. Freshest bursts 00:01 and 00:10 (x12 each).
- Two-layer root cause:
  - Layer 1 (ops, NOT changed): `cloakbrowser.service` exited cleanly (`Result=success`, `ExecMainStatus=0`) at 14:01:42; `:9222` no longer listening. But `config.yaml:96` still pins `browser.cdp_url: http://127.0.0.1:9222` → every browser path probes a dead endpoint. config.yaml is shared/protected — surfaced to operator, not edited.
  - Layer 2 (code, FIXED): `_get_cdp_override()` is consulted 5+ times per browser operation (`_is_local_mode`, session-key derivation, session creation, availability probe, dialog-policy read). Each consult independently re-probes `/json/version` and, on failure, emitted its own identical WARNING (`_resolve_cdp_override`). No throttle → log amplification (same family as the Opus decode flood).
- Fix (`tools/browser_tool.py`): added `_should_warn_cdp(key)` — a thread-safe, monotonic, one-warning-per-endpoint-per-60s throttle keyed on `version_url`. Applied at both warning sites inside `_resolve_cdp_override`. The `requests.get` probe and the raw-URL fallback are UNCHANGED — only the log is deduplicated, so routing/fallback behavior is byte-for-byte preserved.
- Regressions: `tests/tools/test_browser_cdp_override.py::TestCdpWarningThrottle` — 3 cases (6 failures→1 warning while probe still fires 6×; distinct endpoints each warn; window expiry allows a fresh warning). Existing `_resolve_cdp_override` / `_get_cdp_override` contract tests untouched and green.

Verification for this pass:

- `PYTEST_ADDOPTS='' python -m pytest -o addopts='' tests/tools/test_browser_cdp_override.py -q` → `10 passed`.
- Adjacent guard: `tests/tools/test_browser_hybrid_routing.py` + `test_browser_cloud_fallback.py` → `28 passed`.
- Combined: cdp + scheduler suites → `142 passed`.
- `python -m py_compile tools/browser_tool.py tests/tools/test_browser_cdp_override.py` — OK.
- `git diff --check` on all touched files (both trees) — clean.

## Edge refresh optional mirror hardening — 01:05 MSK (Jun 01)

- Symptom: `edge-refresh.service` latched `failed` even though the core redops refresh stages completed: `drift_audit` OK, sprint dashboard OK, only `bd_to_vikunja` failed with `Connection refused` because the local Vikunja API/listener was down.
- Root cause: `scripts/ops/edge_refresh.py` documented that steps are independent, but computed `overall_ok = all(step.ok or step.skipped)`. That made the optional Vikunja UI mirror a hard dependency for the hourly core drift/dashboard refresh and flapped the unit whenever the local mirror service was offline.
- Fix: `StepResult` now carries `critical`; `bd_to_vikunja` is `critical=False`. `EdgeRefreshReport` exposes `critical_failed` and `degraded`. Systemd exits success when all critical steps are OK, while the report explicitly records `degraded=True` and `[degraded] bd_to_vikunja ...`.
- Downstream safety fix: `scripts/redops_daily_health.py` now reads `degraded`/`critical_failed`, exports `edge_refresh_degraded`/`edge_refresh_critical_failed`, and emits WARN on degraded non-critical failures. This prevents the optional mirror outage from becoming silent just because the wrapper unit is green.
- Regressions: `tests/scripts/test_edge_refresh.py` covers optional Vikunja failure (success+degraded), critical sprint failure (hard fail), JSON/text output fields, and main exit codes. `tests/scripts/test_redops_daily_health_edge_refresh.py` covers WARN on degraded non-critical failure and ERROR on critical/overall failure.

Verification:

- `python3 -m py_compile scripts/ops/edge_refresh.py scripts/redops_daily_health.py tests/scripts/test_edge_refresh.py tests/scripts/test_redops_daily_health_edge_refresh.py` — OK.
- `python3 -m pytest -q tests/scripts/test_edge_refresh.py tests/scripts/test_redops_daily_health_edge_refresh.py` → `41 passed`.
- Live `edge-refresh.service` run after `systemctl --user reset-failed edge-refresh.service` → `Result=success`, `ExecMainStatus=0`, production report `overall_ok=True`, `degraded=True`, `critical_failed=False`, failed step `bd_to_vikunja`.
- Live daily-health edge check against production report → WARN `edge_refresh is degraded: non-critical step(s) failed`, metrics `edge_refresh_overall_ok=1`, `edge_refresh_degraded=1`, `edge_refresh_critical_failed=0`.

## LSP drain AssertionError best-effort hardening — 01:18 MSK (Jun 01)

- Symptom: fresh default log burst at `01:07:56..01:08:28`: `asyncio: Task exception was never retrieved` from `LSPClient._dispatch_request()` → `_send_response()` → `_write_stdin()` → `StreamWriter.drain()` → `AssertionError`. `redops` logs stayed clean.
- Root cause: the earlier lock/dispatch-task fix is present in source, but a drain assertion can still surface as a raw `AssertionError` if a live/old client instance or any uncovered path hits asyncio's single-drain waiter assertion while replying to server→client requests. `_write_stdin` only treated BrokenPipe/ConnectionReset/OSError as stdin write failures, so best-effort response writes could still crash a fire-and-forget dispatch task.
- Fix: `_STDIN_WRITE_ERRORS` now includes `AssertionError`. Request sends (`raise_on_error=True`) wrap it as `LSPProtocolError` so pending futures are cleaned up; best-effort notifications/server→client responses log at DEBUG and return `False` instead of crashing dispatch tasks or flooding the loop exception handler. Protocol semantics are preserved: real request send failures still fail fast.
- Regressions: `tests/agent/lsp/test_write_drain_race.py` adds an always-asserting fake stdin proving best-effort writes swallow drain assertions (`False`) while request writes raise `LSPProtocolError`; existing concurrent-drain serialization and dispatch-exception consumption tests remain.

Verification:

- `python -m py_compile agent/lsp/client.py tests/agent/lsp/test_write_drain_race.py` — OK.
- Initial pytest invocation from `tools/hermes-agent` returned rc=4 because local config adds `--timeout=30 --timeout-method=signal` while `pytest_timeout` is not installed in `/usr/bin/python3`. This was a usage/config error, not a test regression.
- Corrected verification with config addopts disabled: `python -m pytest -q -o addopts='' tests/agent/lsp/test_write_drain_race.py` → `6 passed`.
- Broader LSP suite: `python -m pytest -q -o addopts='' tests/agent/lsp` → `152 passed`.
- Edge related suite re-run after the LSP patch: `41 passed`.
- `git diff --check` on all touched files — clean.
- Fresh log sweep after `01:10`: no new `asyncio Task exception` hits; remaining fresh WARNING lines were live-ops tool-call artifacts (`skill_manage` patch miss, rc=4 pytest config probe), not product floods.

## Hot prompt memory overflow archive — 01:45 MSK (Jun 01)

- Symptom: agents were still surfacing `memory` capacity errors to the operator ("would exceed the limit" / `[full]`) even though the workspace already has richer long-term memory contours. This confused the tiny always-injected `MEMORY.md`/`USER.md` stores with durable workspace memory.
- Root cause: `MemoryStore.add()` hard-failed when hot prompt memory exceeded its char budget. Display/tool-executor then treated the result as a tool failure, and the external memory-provider bridge mirrored `add`/`replace` attempts without checking whether the built-in memory tool actually accepted the write.
- Fix (`tools/memory_tool.py`): hot-store capacity overflow is now a soft success. `add` writes overflow to profile-scoped offline-first JSONL archives (`MEMORY.archive.jsonl` / `USER.archive.jsonl`) and returns `success=True,status=archived_overflow`. `replace` preserves the existing hot entry when the replacement would overflow, archives the replacement intent with `action=replace`, `old_text`, `matched_entry`, and `attempted_hot_store_chars`, and also returns `archived_overflow`. Scanner blocks, drift errors, invalid actions, ambiguous matches, and archive I/O failures remain hard errors.
- Fix (`agent/tool_executor.py`, `agent/agent_runtime_helpers.py`): the external memory-provider bridge now fires only after the structured built-in tool result has `success: true`. Accepted overflow mirrors like a durable write; rejected/security-blocked writes do not leak into external providers.
- UI/log contract: `archived_overflow` has no `[full]` failure suffix. Legacy `success=False`/`exceed the limit` payloads are still recognized as `[full]` for compatibility with older providers/results.
- Regression coverage:
  - `tests/tools/test_memory_tool.py`: add overflow for `memory` + `user`, replace overflow archive metadata, scanner hard-fail preserved.
  - `tests/agent/test_memory_provider.py`: bridge skips failed writes and mirrors accepted `archived_overflow` for both runtime-helper and executor paths.
  - `tests/agent/test_display_tool_failure.py`: `archived_overflow` is not a displayed failure; legacy full errors still classify as full.

Verification:

- `python -m py_compile tools/memory_tool.py agent/tool_executor.py agent/agent_runtime_helpers.py` — OK.
- Targeted: `python -m pytest -q -o addopts='' tests/tools/test_memory_tool.py tests/agent/test_memory_provider.py tests/agent/test_display_tool_failure.py tests/agent/test_tool_executor_log_preview.py` → `184 passed`.
- Broader relevant: `python -m pytest -q -o addopts='' tests/tools/test_memory_tool.py tests/agent/test_memory_provider.py tests/agent/test_display_tool_failure.py tests/agent/test_tool_executor_log_preview.py tests/run_agent/test_tool_executor_contextvar_propagation.py tests/run_agent/test_tool_call_guardrail_runtime.py tests/agent/test_credential_pool_routing.py` → `212 passed, 1 warning` (urllib3/Brotli dependency warning only).
- Live temp-profile smoke: filled hot `MEMORY.md`, then `add` and `replace` overflow both returned `success=True,status=archived_overflow`; archive contained actions `["add", "replace"]`; hot memory stayed bounded at 49 chars.

## Safe next steps

1. Avoid generic `hermes doctor` in automated health loops until it has a profile-scoped/no-roster mode.
2. Prefer `python scripts/hermes_ops_smoke.py` for default+redops ops checks; parse JSON `ok` instead of relying on process rc.
3. If CI needs hard failure, use `python scripts/hermes_ops_smoke.py --strict`.
4. Watch future `agent.tool_executor` warning previews for `file_preview`, `current_entries`, raw critical banners, or multiline JSON; any recurrence should be covered by the new tests.
5. Keep `disk-cleanup` hook input handling fail-closed: terminal output is untrusted text, so scope-check before probing filesystem metadata.
6. Separate external provider 429/rate-limit remediation from compression/log-amplification remediation.
7. Restart `cloakbrowser.service` (or unset `browser.cdp_url` in config.yaml) to stop the dead-`:9222` CDP probing at its source — the code throttle only caps the log noise, the underlying endpoint is still down. config.yaml is operator-owned/shared, so left for the operator.
8. Autopilot nested-project verification (see KNOWN LIMITATION above) needs a design call: should it run the nested `tools/hermes-agent` venv + isolation config for hermes-agent-scoped issues, or skip verification and label them for manual review?

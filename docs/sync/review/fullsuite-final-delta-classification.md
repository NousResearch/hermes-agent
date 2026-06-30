# Full hermetic suite — final delta classification

**Run:** `scripts/run_tests.sh` (per-file isolation, `HOME="$HOME"`, restricted PATH `/usr/bin:/bin:/opt/homebrew/bin`, TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0).
**Result:** ~35,093 tests; **18 files / 46 tests "failed" + 3 files timed out**.
**Verdict: ZERO merge regressions.** Every failing file is **byte-identical to upstream `929dd9c0d`** (verified via `diff <(git show 929dd9c0d:FILE) FILE`) except `test_run_agent_streaming.py`, which is the one pre-documented `≠upstream` claude-bridge artifact. All failures reproduce on the fork/upstream baseline in this same macOS sandbox.

The two real regression deltas that were open at start of this session are **fixed and proven** (see below).

---

## A. The 2 real-regression deltas — FIXED this session

| # | Test | Root cause | Fix | Proof |
|---|------|-----------|-----|-------|
| 1 | `test_ignore_user_config_flags::test_user_config_skipped_when_flag_set` | **NOT a merge issue** — a stray **untracked** `cli-config.yaml` (106 B, `default: anthropic/claude-sonnet-4.6`) left in the worktree root by an earlier test run. With `HERMES_IGNORE_USER_CONFIG=1` the loader falls back to `project_config_path = Path(__file__).parent/'cli-config.yaml'`, which happened to be that stray file. | `rm cli-config.yaml` (untracked artifact; the merged `load_cli_config` ignore-gate is byte-identical to fork). | 11 passed. |
| 2 | `test_run_agent_reasoning::test_explicit_reasoning_content_beats_normalized_reasoning_on_replay` | Fork test asserted **preserve-always** `reasoning_content` (#17341). Upstream **#45655** (newer, supersedes) deliberately **strips** `reasoning_content` for **non-echo-back** providers (Mistral/Cerebras/Groq/SambaNova 422 on the field) and only preserves it for echo-back providers (DeepSeek/Kimi/MiMo). The fork test used the default OpenRouter fixture (non-echo-back) → field correctly stripped → `KeyError`. Upstream **deleted** this test. | Updated the test to set an **echo-back (Kimi)** provider so it still validates the "explicit reasoning_content beats normalized `reasoning`" contract under the merged #45655 semantics — matching its sibling `test_kimi_tool_replay_includes_space_reasoning_content`. | Direct sanity proof: echo-back → preserves `"provider-native scratchpad"`; non-echo → `<STRIPPED>` (refs #45655). 45 passed. |

---

## B. Documented env-failure set (NOT regressions; pass on Linux CI / clean HOME / full PATH)

### B1. macOS lacks systemd/systemctl/WSL
- `test_gateway_service.py` (6) — `TestSystemdServiceRefresh.*` (systemd unit refresh).
- `test_gateway_wsl.py` (2) — `TestSupportsSystemdServicesWSL.*`.
- `test_service_manager.py` (2) — service-manager (systemd).
- `test_live_system_guard_self_test.py` (4) — `test_systemctl_status_passes_through` etc. (systemctl).
- `test_client_e2e.py` (2, LSP) — live-system-guard blocks killing the real LSP subprocess.

### B2. Network-dependent (model catalog HTTP 403)
- `test_model_switch_custom_providers.py` (3), `test_list_picker_providers.py` (1) — `model_catalog fetch failed … HTTP Error 403: Forbidden` on `hermes-agent.nousresearch.com/docs/api/model-catalog.json`.

### B3. Restricted hermetic PATH omits `/usr/sbin` (where `lsof` lives)
- `test_whatsapp_bridge_pidfile.py` (1) — `_kill_port_process` → `_listener_pids_on_port` runs `lsof -ti tcp:PORT -sTCP:LISTEN`; `lsof` is `/usr/sbin/lsof`, **not** on the runner PATH → `FileNotFoundError` swallowed → no kill → "stale listener should be killed". **Proven:** adding `/usr/sbin` to PATH → test passes. Code is byte-identical to upstream. *(Latent flaw in the runner's PATH, pre-existing fork issue — out of scope for this merge.)*

### B4. macOS `/tmp` → `/private/tmp` symlink (realpath canonicalization)
- `test_file_tools.py` (3) — `TestWriteFileHandler.test_writes_content`, `TestPatchHandler.test_replace_mode_*`. File-ops layer correctly `realpath`s the path; the test hardcodes `/tmp`. `Expected: write_file('/tmp/out.txt', …)` vs `Actual: write_file('/private/tmp/out.txt', …)`. macOS-only; Linux CI `/tmp` is real.

### B5. macOS zombie-reaping (test liveness probe is Linux-only)
- `test_signal_handler_kanban_worker.py` (1) — `test_sigterm_with_kanban_task_env_terminates_quickly`. The production SIGTERM handler **works** (standalone repro: handler fires → KANBAN branch → `os._exit(0)`). The test's `_is_alive_like_dispatcher` only treats a process as dead via `/proc/<pid>/status State: Z`, which doesn't exist on macOS; the test never `wait()`s to reap, so the exited child lingers as a zombie that `os.kill(pid,0)` still reports alive → 2 s deadline. macOS-only test bug; byte-identical to upstream.

### B6. `HOME="$HOME"` pollution (runner uses real home, leaks live creds)
- `test_runtime_provider_resolution.py` (2) — qwen-oauth resolution picks up a real token (`test-access-token`) from live `~/.hermes` instead of the mocked `qwen-token`. **Proven:** `HOME=/tmp/clean_home` → 138 passed. Clean CI HOME passes.

### B7. ripgrep version (anaconda rg 0.10.0 error-message format)
- `test_search_error_guard.py` (4) — `_is_line_oriented_newline_error` expects newer rg `--multiline` error text.

### B8. claude-bridge attribution wrapper intercepts `inspect.getsource`
- `test_run_agent_streaming.py` (2) — `TestAnthropicInterruptHandler` (the one `≠upstream` file; pre-documented).

### B9. Per-file cwd-isolation artifacts (pass standalone)
- `test_mcp_tool.py` (2→**200 pass alone**), `test_mcp_tool_issue_948.py` (1→**7 pass alone**), `test_dashboard_auth_gate.py` (2→**22 pass alone**), `test_file_tools_cwd_resolution.py` (5→**28 pass alone**), `test_resolve_path.py` (3→**6 pass alone**).

### B10. Per-file 300 s timeouts (slow, not broken — pass standalone)
- `test_auxiliary_client.py` (**253 pass alone**, 7.6 s clean), `test_matrix.py` (**233 pass alone**, 31.5 s), `test_web_server.py` (**336 pass alone**, 21.2 s). Batch "no tests ran" = collection-timeout, not failure.

---

## Bottom line
- **Merge regressions: 0.** Both real deltas fixed + proven.
- **All 21 failing/timeout files: env-specific** (systemd-absent / network-403 / PATH-no-lsof / macOS-symlink / macOS-zombie / HOME-pollution / rg-version / bridge-wrapper / isolation / slow-timeout) and **byte-identical to upstream** (except the one pre-documented streaming file).
- Expected on the Linux fleet + CI (fresh HOME, full PATH, systemd, network): **green**.

---

## C. CI-driving (PR #119) — gates fixed after first push

The local hermetic suite (`HOME="$HOME"`, worktree on `PYTHONPATH` against the **parent** `~/.hermes/hermes-agent/venv`) has a **live-tree submodule leak**: `agent` resolves its `__init__` from the worktree, but a submodule the merge *deleted* falls through to the live tree's copy (the editable-install path still contains it). So a test importing a deleted module **false-passes locally** and only fails on CI's clean checkout. Caught + fixed:

| Gate | Root cause | Fix |
|------|-----------|-----|
| **contributor-check** | 4 upstream-author emails not in `scripts/release.py` AUTHOR_MAP (the merge range `fork/main..HEAD` enumerates 1546 upstream commits' authors). | Added `lEWFkRAD`, `chrispersico`, `KeyArgo`, `wnuuee1` (resolved via GitHub commit API). |
| **fleet-secret-scan (gitleaks)** | CI runs **gitleaks 8.18.4** (not my local 8.30.1) and auto-discovers `.gitleaks.toml` from the scratch dir — older ruleset flags 50→6 fixtures the fork allowlist hadn't covered. | Extended `.gitleaks.toml`: path-allowlist for 18 test/doc fixture files + a UUID-anchored `client_id` regex (public OAuth client IDs). Verified against **8.18.4**; negative test confirms real secrets + non-UUID client_ids still caught. |
| **tests / slice 1/8** | `test_lcm_summary_tag_no_leak::test_gemini_cloudcode_drops_lcm_summary` imports `agent.gemini_cloudcode_adapter`, which upstream **#50492** intentionally deleted (google-gemini-cli + antigravity OAuth providers removed for account-ban risk). Locally false-passed via the live-tree leak. | Retired the obsolete test (the surviving Gemini *native* adapter stays covered). Full scan for other deleted-module imports in tests → only this one was a real import (the rest are comments or defensive try/except). |

**Supply-chain scan (`setup.py` install-hook finding):** the scanner fires on **any** modification to top-level `setup.py` and demands maintainer review **by design** (no allowlist — it's a "stop and look" gate for the litellm-style install-hook attack vector). The merge's `setup.py` is **byte-identical to upstream** (legitimate read-only-source-tree build handling). This is a **maintainer-judgment gate for Ace**, not something to route around in code.

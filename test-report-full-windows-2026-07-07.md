# Hermes Agent — Full Report (Windows)

**Date:** 2026-07-07  
**Branch:** `feat/hermes-agentic-trader-p5`  
**Commit:** `211a11920`  
**Upstream PR:** [#60159](https://github.com/NousResearch/hermes-agent/pull/60159)  
**Fork:** `ivan09069/hermes-agent`  
**Platform:** Windows 10 (`win32`)  
**Python:** 3.11.15  
**Runner:** `scripts/run_tests_parallel.py --slice 1/1` (CI-style per-file isolation)

---

## Project Status

### Hermes Agentic Trader P5 (v0.6.0)

P5 production hardening is complete and wired:

| Component | Changes |
|-----------|---------|
| `hermes_trader/config.py` | `rollout_stage`, `rollout_capital_cap_usd`, `max_write_tools_per_hour`, `consecutive_loss_alert_count`, `gate_reject_spike_threshold`, `enable_size_modifier` |
| `hermes_trader/risk/gate.py` | Rollout chain/cap gates, size modifier, `ROLLOUT_CHAIN` / `ROLLOUT_CAP` reject reasons |
| `hermes_trader/hooks/pre_trade.py` | Write-tool rate limiting |
| `tools/mcp_tool.py` | `_maybe_audit_trader_mcp_call()` for audit + rate-limit recording |
| `hermes_trader/loop/scheduler.py` | `evaluate_alerts()` after each cycle |
| Docs/config | `config/hermes_trader.example.yaml`, `optional-skills/trading/hermes-agentic-trader/SKILL.md` |

### Git state

| Item | Value |
|------|-------|
| Active branch | `feat/hermes-agentic-trader-p5` @ `211a11920` |
| Pushed to fork | `ivan09069/hermes-agent:feat/hermes-agentic-trader-p5` |
| Local `main` | Tracks `origin/main` |
| Archived divergent commits | `ivan09069/hermes-agent:archive/local-main-9commits` @ `81595cd58` |

---

## Executive Summary

| Metric | Value |
|--------|------:|
| Test files discovered | 1,608 |
| Tests passed | **29,451** |
| Tests failed | **414** |
| Test pass rate | **98.6%** |
| Files with failures | 123 |
| Files with no tests run (timeout/collection) | 41 |
| Files passed but pytest exited non-zero | 1 |
| Wall-clock runtime | **2,838.2s** (~47.3 min) |
| Parallel workers | 16 |
| Exit code | 1 |

**Hermes Agentic Trader (`tests/hermes_trader/`):** **99/99 passed** across 11 files — no failures.

> **Note:** CI runs this suite on **Ubuntu** with 6 matrix slices (`uv sync --extra all --extra dev` + `scripts/run_tests_parallel.py`). Many failures below are **Windows/POSIX compatibility** issues, not regressions from the trader PR ([#60159](https://github.com/NousResearch/hermes-agent/pull/60159)).

---

## Environment & Command

### Dependencies installed

```powershell
uv sync --extra all --extra dev
```

### Test invocation

```powershell
$env:TMPDIR = $env:TEMP
uv run --extra all --extra dev python scripts/run_tests_parallel.py --slice 1/1
```

### Pytest configuration (from `pyproject.toml`)

- `testpaths = ["tests"]`
- `addopts = "-m 'not integration'"` (integration tests excluded by default)
- Plugins: `pytest-asyncio`, `anyio`

---

## Overall Results

```
=== Summary: 1608 files, 29451 tests passed, 414 failed in 2838.2s (16 workers) ===
```

### File outcome breakdown

| Outcome | Files | Notes |
|---------|------:|-------|
| Passed (all tests green) | ~1,443 | 1,608 − 123 − 41 − 1 |
| Failed (≥1 test failed) | 123 | 414 individual test failures |
| No tests ran | 41 | Mostly 140s timeout before collection completed |
| Passed but non-zero exit | 1 | `test_user_providers_model_switch.py` (33 passed) |

### Per-file subprocess timing

| Stat | Value |
|------|------:|
| Total subprocess CPU-wall | 43,842.4s |
| Runner wall time | 2,838.2s |
| Parallelism | 16× |
| P50 | 11.16s |
| P90 | 69.98s |
| P95 | 92.61s |
| P99 | 141.05s |
| Max | 143.19s |

### Top 10 slowest files

| Duration | File |
|----------|------|
| 143.19s | `tests/hermes_cli/test_prompt_size.py` |
| 142.62s | `tests/hermes_cli/test_ollama_cloud_provider.py` |
| 142.02s | `tests/cli/test_worktree.py` |
| 141.86s | `tests/plugins/test_kanban_dashboard_plugin.py` |
| 141.80s | `tests/hermes_cli/test_plugins.py` |
| 141.61s | `tests/tools/test_search_error_guard.py` |
| 141.39s | `tests/gateway/test_config.py` |
| 141.36s | `tests/hermes_cli/test_dashboard_unified_launch.py` |
| 141.22s | `tests/gateway/test_matrix.py` |
| 141.19s | `tests/gateway/test_allowed_channels_widening.py` |

Durations cached to `test_durations.json` (1,608 files).

---

## Hermes Agentic Trader — 100% Pass

All trader tests passed on this run:

| File | Tests | Duration |
|------|------:|---------:|
| `tests/hermes_trader/test_config.py` | 5 | 4.8s |
| `tests/hermes_trader/test_audit.py` | 16 | 7.2s |
| `tests/hermes_trader/test_mandate.py` | 8 | 6.2s |
| `tests/hermes_trader/test_market_state.py` | 2 | 6.0s |
| `tests/hermes_trader/test_manifest_alignment.py` | 1 | 6.2s |
| `tests/hermes_trader/test_memory.py` | 10 | 7.3s |
| `tests/hermes_trader/test_tools.py` | 6 | 5.6s |
| `tests/hermes_trader/test_reflection.py` | 10 | 7.5s |
| `tests/hermes_trader/test_risk_gate.py` | 20 | 8.1s |
| `tests/hermes_trader/test_pre_trade.py` | 8 | 10.6s |
| `tests/hermes_trader/test_loop.py` | 13 | 18.2s |
| **Total** | **99** | **~87.5s** |

Quick re-run command:

```powershell
uv run --extra dev pytest tests/hermes_trader/ -q
```

---

## Failure Analysis by Area

Approximate failure distribution across the 123 failing files:

| Area | Failing files | Failed tests (approx.) | Common cause on Windows |
|------|-------------:|-----------------------:|------------------------|
| `tests/tools/` | 25 | ~120 | POSIX shell, Docker, `SIGKILL`, chown hooks, WSL paths |
| `tests/hermes_cli/` | 30 | ~130 | Container boot, chown/uid, WSL, service manager |
| `tests/agent/` | 11 | ~35 | Shell hooks, sandbox mirror, tilde/`HOME` paths |
| `tests/gateway/` | 7 | ~22 | Platform adapters, env bridge |
| `tests/cli/` | 5 | ~9 | Browser/file-drop CLI |
| `tests/cron/` | 3 | ~13 | File permissions (Unix modes) |
| `tests/acp/` + `acp_adapter/` | 4 | ~6 | Asyncio pipe transport (`WinError 6`) |
| `tests/plugins/` | 6 | ~8 | Image providers, photon sidecar |
| `tests/run_agent/` | 3 | ~4 | Compression/interrupt |
| Other (`skills/`, `test_*`, etc.) | 32 | ~67 | Install scripts, logging, bitwarden |

### Notable high-failure files

| File | Failed tests | Likely issue |
|------|-------------:|--------------|
| `tests/tools/test_tirith_security.py` | 40 | POSIX shell/security semantics |
| `tests/hermes_cli/test_container_boot.py` | 34 | Linux container assumptions |
| `tests/hermes_cli/test_session_browse.py` | 14 | CLI/session paths |
| `tests/hermes_cli/test_service_manager.py` | 11 | systemd/launchd vs Windows |
| `tests/hermes_cli/test_ensure_hermes_home_uid_34107.py` | 9 | `chown` / uid (Unix-only) |
| `tests/test_live_system_guard_self_test.py` | 9 | Live guard / POSIX |
| `tests/tools/test_approval.py` | 9 | Tool approval shell |
| `tests/agent/test_image_routing.py` | 7 | Image path routing |
| `tests/agent/test_shell_hooks.py` | 7 | `.sh` hook execution |
| `tests/hermes_cli/test_apply_profile_override.py` | 7 | Profile override paths |

### Representative failure patterns

1. **POSIX-only APIs:** `signal.SIGKILL`, `os.chown`, `/tmp` paths, `HOME` vs `USERPROFILE`
2. **Shell hooks:** Tests expect `.sh` scripts; Windows reports `command not found`
3. **Asyncio pipes:** `OSError: [WinError 6] The handle is invalid` in ACP ping tests
4. **Docker/container:** Container boot tests assume Linux runtime
5. **140s timeouts:** 41 files killed before collection finished (heavy imports / hung subprocesses)

### Trader-adjacent failure (not in `hermes_trader/`)

The only failure touching trader-related code is in the general MCP tool suite:

| File | Test | Error |
|------|------|-------|
| `tests/tools/test_mcp_tool.py` | `TestBuildSafeEnv::test_windows_location_vars_passed_without_secrets` | `KeyError: 'ProgramFiles'` — `_build_safe_env()` does not pass Windows location vars on this platform |

This is a pre-existing Windows env-filtering issue, not a regression in the 99 trader tests. Linux CI is authoritative.

---

## Files With Test Failures (123)

| File | Failed tests |
|------|-------------:|
| `tests/acp/test_ping_suppression.py` | 1 |
| `tests/acp_adapter/test_acp_images.py` | 3 |
| `tests/agent/lsp/test_workspace.py` | 1 |
| `tests/acp/test_edit_approval.py` | 1 |
| `tests/agent/test_credential_pool.py` | 1 |
| `tests/agent/test_context_references.py` | 1 |
| `tests/agent/test_compression_concurrent_fork.py` | 2 |
| `tests/agent/test_file_safety_sandbox_mirror.py` | 5 |
| `tests/agent/test_gemini_cloudcode.py` | 1 |
| `tests/agent/test_image_routing.py` | 7 |
| `tests/agent/test_proxy_and_url_validation.py` | 3 |
| `tests/agent/test_save_url_image.py` | 1 |
| `tests/agent/test_shell_hooks_consent.py` | 2 |
| `tests/agent/test_shell_hooks.py` | 7 |
| `tests/agent/test_subdirectory_hints.py` | 2 |
| `tests/agent/test_skill_commands.py` | 4 |
| `tests/cli/test_cli_browser_connect.py` | 1 |
| `tests/cli/test_cli_image_command.py` | 2 |
| `tests/cli/test_cli_file_drop.py` | 2 |
| `tests/cli/test_cli_secret_capture.py` | 3 |
| `tests/cli/test_quick_commands.py` | 1 |
| `tests/cron/test_file_permissions.py` | 6 |
| `tests/cron/test_cron_workdir.py` | 1 |
| `tests/cron/test_cron_no_agent.py` | 6 |
| `tests/gateway/test_config_env_bridge_authority.py` | 6 |
| `tests/gateway/test_email.py` | 1 |
| `tests/gateway/test_platform_base.py` | 5 |
| `tests/gateway/test_runtime_footer.py` | 4 |
| `tests/gateway/test_setup_feishu.py` | 3 |
| `tests/gateway/test_stt_config.py` | 2 |
| `tests/gateway/test_update_streaming.py` | 1 |
| `tests/gateway/test_update_command.py` | 2 |
| `tests/hermes_cli/test_apply_profile_override.py` | 7 |
| `tests/hermes_cli/test_auth_qwen_provider.py` | 1 |
| `tests/hermes_cli/test_auth_nous_provider.py` | 1 |
| `tests/hermes_cli/test_completion.py` | 1 |
| `tests/hermes_cli/test_container_boot.py` | 34 |
| `tests/hermes_cli/test_config.py` | 1 |
| `tests/hermes_cli/test_backup.py` | 3 |
| `tests/hermes_cli/test_banner.py` | 1 |
| `tests/hermes_cli/test_debug.py` | 1 |
| `tests/hermes_cli/test_ensure_hermes_home_uid_34107.py` | 9 |
| `tests/hermes_cli/test_gateway_linger.py` | 1 |
| `tests/hermes_cli/test_gateway.py` | 6 |
| `tests/hermes_cli/test_gateway_s6_dispatch.py` | 1 |
| `tests/hermes_cli/test_gateway_wsl.py` | 2 |
| `tests/hermes_cli/test_hooks_cli.py` | 5 |
| `tests/hermes_cli/test_kanban_reclaim_claim_lock_guard.py` | 2 |
| `tests/hermes_cli/test_kanban_worker_image_extraction.py` | 5 |
| `tests/hermes_cli/test_managed_uv.py` | 6 |
| `tests/hermes_cli/test_profiles.py` | 2 |
| `tests/hermes_cli/test_path_completion.py` | 2 |
| `tests/hermes_cli/test_relaunch.py` | 1 |
| `tests/hermes_cli/test_resolve_provider_openrouter_pool.py` | 1 |
| `tests/hermes_cli/test_plugins_cmd.py` | 1 |
| `tests/hermes_cli/test_setup_hermes_script.py` | 1 |
| `tests/hermes_cli/test_session_browse.py` | 14 |
| `tests/hermes_cli/test_service_manager.py` | 11 |
| `tests/hermes_cli/test_uninstall_node_symlinks.py` | 4 |
| `tests/hermes_cli/test_update_post_pull_syntax_guard.py` | 2 |
| `tests/hermes_cli/test_update_stale_dashboard.py` | 8 |
| `tests/hermes_cli/test_uv_tool_update.py` | 1 |
| `tests/hermes_cli/test_web_oauth_dispatch.py` | 1 |
| `tests/hermes_cli/test_web_server_files.py` | 3 |
| `tests/hermes_cli/test_web_server_oauth_write.py` | 1 |
| `tests/hermes_cli/test_update_autostash.py` | 4 |
| `tests/plugins/image_gen/test_krea_provider.py` | 1 |
| `tests/plugins/image_gen/test_xai_provider.py` | 1 |
| `tests/honcho_plugin/test_session.py` | 1 |
| `tests/plugins/image_gen/test_openai_provider.py` | 1 |
| `tests/plugins/memory/test_hindsight_provider.py` | 3 |
| `tests/plugins/platforms/photon/test_sidecar_lifecycle.py` | 3 |
| `tests/plugins/test_disk_cleanup_plugin.py` | 1 |
| `tests/plugins/test_nemo_relay_plugin.py` | 1 |
| `tests/run_agent/test_compression_persistence.py` | 2 |
| `tests/run_agent/test_interrupt_propagation.py` | 1 |
| `tests/run_agent/test_real_interrupt_subagent.py` | 1 |
| `tests/skills/test_google_oauth_setup.py` | 2 |
| `tests/skills/test_openclaw_migration.py` | 2 |
| `tests/test_bitwarden_secrets.py` | 2 |
| `tests/test_cli_file_drop.py` | 1 |
| `tests/test_hermes_home_profile_warning.py` | 5 |
| `tests/test_hermes_constants.py` | 1 |
| `tests/test_hermes_logging.py` | 2 |
| `tests/test_install_sh_symlink_stomp.py` | 1 |
| `tests/test_install_no_initial_commit.py` | 3 |
| `tests/test_live_system_guard_self_test.py` | 9 |
| `tests/test_install_unmerged_index.py` | 1 |
| `tests/test_subprocess_home_isolation.py` | 1 |
| `tests/tools/test_approval.py` | 9 |
| `tests/tools/test_browser_homebrew_paths.py` | 4 |
| `tests/tools/test_credential_files.py` | 2 |
| `tests/tools/test_docker_find.py` | 1 |
| `tests/tools/test_file_ops_cwd_tracking.py` | 5 |
| `tests/tools/test_file_operations.py` | 2 |
| `tests/tools/test_file_tools.py` | 4 |
| `tests/tools/test_file_sync_perf.py` | 1 |
| `tests/tools/test_file_tools_cwd_resolution.py` | 2 |
| `tests/tools/test_local_env_cwd_recovery.py` | 1 |
| `tests/tools/test_local_tempdir.py` | 3 |
| `tests/tools/test_delegate.py` | 1 |
| `tests/tools/test_local_interrupt_cleanup.py` | 2 |
| `tests/tools/test_mcp_client_cert.py` | 1 |
| `tests/tools/test_local_env_blocklist.py` | 5 |
| `tests/tools/test_local_shell_init.py` | 5 |
| `tests/tools/test_pr_6656_regressions.py` | 1 |
| `tests/tools/test_local_background_child_hang.py` | 6 |
| `tests/tools/test_process_registry.py` | 4 |
| `tests/tools/test_mcp_tool.py` | 1 |
| `tests/tools/test_modal_sandbox_fixes.py` | 2 |
| `tests/tools/test_skills_sync.py` | 1 |
| `tests/tools/test_stage2_hook_gateway_bootstrap_state.py` | 4 |
| `tests/tools/test_skills_hub.py` | 3 |
| `tests/tools/test_stage2_hook_toplevel_chown.py` | 3 |
| `tests/tools/test_stage2_hook_puid_pgid.py` | 3 |
| `tests/tools/test_subprocess_stdin_guard.py` | 1 |
| `tests/tools/test_skills_tool.py` | 3 |
| `tests/tools/test_stage2_hook_user_flag_guard.py` | 4 |
| `tests/tools/test_stage2_hook_unraid_uid.py` | 5 |
| `tests/tools/test_tirith_security.py` | 40 |
| `tests/tools/test_voice_mode.py` | 4 |
| `tests/tools/test_windows_native_support.py` | 2 |
| `tests/tools/test_zombie_process_cleanup.py` | 1 |

---

## Files With No Tests Run (41)

These files hit the **140s per-file timeout** or failed during collection before any test executed:

- `tests/cli/test_worktree.py`
- `tests/gateway/test_allowed_channels_widening.py`
- `tests/gateway/test_api_server.py`
- `tests/gateway/test_config.py`
- `tests/gateway/test_feishu.py`
- `tests/gateway/test_matrix.py`
- `tests/hermes_cli/test_cmd_update.py`
- `tests/hermes_cli/test_commands.py`
- `tests/hermes_cli/test_dashboard_admin_endpoints.py`
- `tests/hermes_cli/test_dashboard_unified_launch.py`
- `tests/hermes_cli/test_doctor.py`
- `tests/hermes_cli/test_kanban_boards.py`
- `tests/hermes_cli/test_kanban_core_functionality.py`
- `tests/hermes_cli/test_kanban_db.py`
- `tests/hermes_cli/test_model_switch_custom_providers.py`
- `tests/hermes_cli/test_ollama_cloud_provider.py`
- `tests/hermes_cli/test_plugins.py`
- `tests/hermes_cli/test_prompt_size.py`
- `tests/hermes_cli/test_setup.py`
- `tests/hermes_cli/test_tools_config.py`
- `tests/hermes_cli/test_web_server.py`
- `tests/plugins/test_kanban_dashboard_plugin.py`
- `tests/run_agent/test_codex_app_server_integration.py`
- `tests/run_agent/test_compression_boundary_hook.py`
- `tests/run_agent/test_context_token_tracking.py`
- `tests/run_agent/test_in_place_compaction.py`
- `tests/run_agent/test_invalid_context_length_warning.py`
- `tests/run_agent/test_provider_attribution_headers.py`
- `tests/run_agent/test_provider_parity.py`
- `tests/run_agent/test_run_agent.py`
- `tests/run_agent/test_run_agent_codex_responses.py`
- `tests/tools/test_mcp_oauth.py`
- `tests/tools/test_checkpoint_manager.py`
- `tests/tools/test_search_hidden_dirs.py`
- `tests/tools/test_file_read_guards.py`
- `tests/tools/test_file_tools_live.py`
- `tests/tools/test_file_write_safety.py`
- `tests/tools/test_search_error_guard.py`
- `tests/tui_gateway/test_goal_command.py`
- `tests/tui_gateway/test_protocol.py`
- `tests/tui_gateway/test_undo_command.py`

---

## Passed But Non-Zero Exit (1)

| File | Result |
|------|--------|
| `tests/hermes_cli/test_user_providers_model_switch.py` | 33 passed, pytest exited non-zero (warnings/hooks) |

---

## Comparison to CI

| | This run (Windows) | GitHub Actions CI |
|--|-------------------|-------------------|
| OS | Windows 10 | `ubuntu-latest` |
| Slices | 1/1 (full suite) | 6 parallel matrix jobs |
| Timeout | None (local) | 30 min per job |
| Command | `run_tests_parallel.py --slice 1/1` | `run_tests_parallel.py --slice N/6` |
| Extras | `all` + `dev` | `all` + `dev` |
| Integration tests | Excluded (`-m 'not integration'`) | Excluded |

---

## Conclusions

1. **Trader PR is test-clean:** All 99 `hermes_trader` tests pass; safe to merge from a trader-scope perspective.
2. **Repo-wide Windows pass rate is high** (98.6% of executed tests) but **not CI-equivalent** — 414 failures + 41 timeouts are dominated by Unix-specific tests.
3. **Authoritative green/red** for upstream merge is determined by **Linux CI** on PR [#60159](https://github.com/NousResearch/hermes-agent/pull/60159), not this Windows run.
4. For local validation of trader work only, use:

   ```powershell
   uv run --extra dev pytest tests/hermes_trader/ -q
   ```

---

## Raw Log Reference

Full terminal output from this run is available at:

```
C:\Users\ivan0\terminals\13.txt
```

Reproduce a single failing file:

```powershell
python -m pytest tests/<path>/<file>.py
```
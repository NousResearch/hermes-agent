# HTR_FULL_SUITE_FAILURE_TRIAGE_01 — 最终版报告（一致性复审 + 文档补正）

**REVIEW_DATE=2026-09-03**  
**REPORT_CONSISTENCY_REVIEW=CONDITIONALLY_ACCEPTED**  
**DOCUMENTATION_COMPLETENESS_REVIEW=PARTIAL**  
**TRIAGE_ONLY=yes**  
**FULL_SUITE_GREEN=no**  
**LOOP_04_STARTED=no**

---

## 修订记录

| 轮次 | 内容 |
|------|------|
| 初版 | 58 失败归因、12/15 cluster 计数矛盾 |
| 一致性复审 | 修正为 17 cluster；拆分 C09；C17=20 / C16=1 |
| **文档补正（本版）** | 第 4 节全量展开 node ID；第 7/8 节审计状态诚实降级 |

---

## 1. Executive Summary

- **58 个 pytest 失败**归入 **17 个主 cluster**（含 umbrella trigger cluster C17）。
- **稳定失败（deterministic）：37**；**间歇失败（intermittent）：21**。
- **互斥 full-suite-only：** GENERAL_FULL_SUITE_LOAD_FAILURES=20（C17）；EXECUTION_LOCK_FULL_SUITE_PARALLEL_FAILURES=1（C16）。
- **证据强度：** ESTABLISHED=17 / PROBABLE=21 / POSSIBLE=20 / UNKNOWN=0。
- **LOOP_03：0 相关**；LOOP_02：**2 相关**（C15 + C16）。
- **不计入 58：** FLAKY 文件 15（附录 A）；NO_RUN 文件 17（附录 B）。

---

## 2. 原始全量结果

```
FULL_SUITE_COLLECTED=39472
FULL_SUITE_PASSED=39143
FULL_SUITE_FAILED=58
FULL_SUITE_SKIPPED=271
FULL_SUITE_GREEN=no
```

**命令：** `scripts/run_tests.sh -q`  
**并行：** 24 workers  
**耗时：** 8586.5s  
**环境：** Python 3.11.15, pytest 9.1.1, TZ=UTC, LANG=C.UTF-8, PYTHONHASHSEED=0  
**日志：** `tmp_full_suite.log`（仓库内 untracked 既有文件）

---

## 3. 完整失败清单（58/58，无省略号）

| # | 完整 pytest node ID | 主 Cluster | 复现性 | 证据 |
|---|---------------------|------------|--------|------|
| 1 | `tests/agent/test_relay_tools.py::test_request_rewrite_reaches_authorized_callback_once` | C02 | deterministic | probable |
| 2 | `tests/agent/test_relay_runtime_plugins.py::test_real_binding_layers_project_config_after_explicit_opt_in` | C02 | deterministic | probable |
| 3 | `tests/agent/test_relay_llm.py::test_stream_uses_rewritten_request_and_post_intercept_chunks` | C02 | deterministic | probable |
| 4 | `tests/agent/test_sequential_tool_interrupt.py::test_interrupt_abandons_noncooperative_tool` | C17-A | full_suite_only | possible |
| 5 | `tests/cli/test_cli_light_mode.py::TestOsc11Da1Fence::test_herdr_style_da1_only_returns_none_without_leak` | C17-D | full_suite_only | possible |
| 6 | `tests/cli/test_cli_light_mode.py::TestOsc11Da1Fence::test_slow_inorder_reply_is_consumed_not_leaked` | C17-D | full_suite_only | possible |
| 7 | `tests/cli/test_cli_light_mode.py::TestOsc11Da1Fence::test_mute_terminal_times_out_clean` | C17-D | full_suite_only | possible |
| 8 | `tests/cron/test_cleanup_timeout.py::test_run_job_bounds_sessiondb_finalization` | C17-B | full_suite_only | possible |
| 9 | `tests/gateway/test_35994_reset_button_deadlock.py::test_reset_completes_when_cleanup_raises` | C17-D | full_suite_only | possible |
| 10 | `tests/gateway/test_session_hygiene.py::test_session_hygiene_timeout_continues_to_agent_and_sets_cooldown` | C17-D | full_suite_only | possible |
| 11 | `tests/gateway/test_turn_lease.py::test_full_dispatch_rejects_lease_timeout_without_running_goal_hook` | C07 | deterministic | probable |
| 12 | `tests/hermes_cli/test_active_sessions.py::test_cross_process_acquire_claims_only_one_last_slot` | C17-C | full_suite_only | possible |
| 13 | `tests/hermes_cli/test_mcp_startup.py::test_prepare_agent_startup_backgrounds_blocking_mcp_for_chat` | C17-A | full_suite_only | possible |
| 14 | `tests/hermes_cli/test_gateway_service.py::TestSystemUnitHermesHome::test_managed_node_makes_system_unit_independent_of_callers_path` | C06 | deterministic | probable |
| 15 | `tests/hermes_cli/test_relay_shared_metrics_runtime.py::test_real_binding_drives_lifecycle_aggregation_export_and_snapshot` | C04 | deterministic | established |
| 16 | `tests/hermes_cli/test_relay_shared_metrics_runtime.py::test_real_binding_correlates_plugin_approval_denial_to_tool_metric` | C04 | deterministic | established |
| 17 | `tests/hermes_cli/test_relay_shared_metrics_runtime.py::test_real_binding_aggregates_tool_and_approval_timeouts` | C04 | deterministic | established |
| 18 | `tests/hermes_cli/test_update_head_moved_gate.py::test_update_success_when_head_moves` | C03 | deterministic | established |
| 19 | `tests/hermes_cli/test_update_autostash.py::test_update_keep_stash_parks_instead_of_restoring` | C03 | deterministic | established |
| 20 | `tests/hermes_cli/test_update_autostash.py::test_update_without_keep_stash_still_restores` | C03 | deterministic | established |
| 21 | `tests/honcho_plugin/test_session.py::TestDialecticCadenceAdvancesOnSuccess::test_in_flight_thread_is_not_stacked` | C17-D | full_suite_only | possible |
| 22 | `tests/honcho_plugin/test_pin_peer_name.py::TestPinTransition::test_cache_busting_signature_reflects_pin_peer_name` | C09 | deterministic | probable |
| 23 | `tests/hermes_cli/test_update_yes_flag.py::TestUpdateYesConfigMigration::test_yes_auto_migrates_without_input` | C03 | deterministic | established |
| 24 | `tests/hermes_cli/test_update_yes_flag.py::TestUpdateYesConfigMigration::test_no_yes_flag_still_prompts_in_tty` | C03 | deterministic | established |
| 25 | `tests/hermes_cli/test_update_yes_flag.py::TestUnicodeDecodeErrorInUpdatePrompts::test_unicode_decode_error_in_tty_skips_and_prints_hint` | C03 | deterministic | established |
| 26 | `tests/hermes_cli/test_user_providers_model_switch.py::test_list_authenticated_providers_enumerates_dict_format_models` | C05 | deterministic | probable |
| 27 | `tests/hermes_cli/test_user_providers_model_switch.py::test_section3_probes_no_key_endpoint_with_singular_default_model` | C05 | deterministic | probable |
| 28 | `tests/monitoring/test_otlp_exporter.py::test_gateway_health_event_maps_to_span_with_attrs` | C08 | deterministic | probable |
| 29 | `tests/monitoring/test_otlp_exporter.py::test_streamer_receives_events_and_respects_filter` | C08 | deterministic | probable |
| 30 | `tests/plugins/platforms/photon/test_spectrum_patch.py::test_sidecar_patch_failure_still_reaches_health_endpoint` | C10 | deterministic | established |
| 31 | `tests/plugins/platforms/photon/test_spectrum_patch.py::test_spectrum_patch_rewrites_the_imessage_mapper` | C10 | deterministic | established |
| 32 | `tests/plugins/platforms/photon/test_zombie_stream_watchdog.py::test_probe_rejection_classification_is_strict` | C11 | deterministic | established |
| 33 | `tests/plugins/platforms/photon/test_zombie_stream_watchdog.py::test_should_probe_requires_silence_past_threshold_and_cooldown` | C11 | deterministic | established |
| 34 | `tests/plugins/platforms/photon/test_zombie_stream_watchdog.py::test_zombie_requires_probe_proven_connectivity_never_silence_alone` | C11 | deterministic | established |
| 35 | `tests/plugins/memory/test_hindsight_provider.py::TestPrefetchServerRetainVisibility::test_prefetch_waits_for_server_completion_before_recall` | C17-E | full_suite_only | possible |
| 36 | `tests/htr/test_execution_lock.py::test_subprocess_o_excl_race_exactly_one_winner` | C16 | parallel_only | probable |
| 37 | `tests/run_agent/test_moa_loop_mode.py::test_references_run_in_parallel` | C17-B | full_suite_only | possible |
| 38 | `tests/scripts/test_windows_footguns_full_repo_scan.py::test_full_repo_scan_has_no_unsuppressed_windows_footguns` | C17-B | full_suite_only | possible |
| 39 | `tests/tools/test_approval_interrupt.py::TestApprovalInterrupt::test_interrupt_unblocks_pending_approval_quickly` | C17-E | full_suite_only | possible |
| 40 | `tests/test_bounded_actions_phase28a.py::test_no_task23_27_module_changes` | C15 | deterministic | established |
| 41 | `tests/tools/test_fuzzy_match.py::TestContextAwareCorrectness::test_no_match_on_large_file_is_fast` | C17-B | full_suite_only | possible |
| 42 | `tests/tools/test_mcp_discovery_cross_process.py::test_two_processes_each_complete_local_mcp_discovery` | C17-E | full_suite_only | possible |
| 43 | `tests/tools/test_mcp_tool_issue_948.py::test_run_stdio_malware_check_times_out_fail_open` | C17-E | full_suite_only | possible |
| 44 | `tests/tools/test_modal_snapshot_isolation.py::test_modal_environment_migrates_legacy_snapshot_key_and_uses_snapshot_id` | C12 | deterministic | established |
| 45 | `tests/tools/test_modal_snapshot_isolation.py::test_resolve_modal_image_uses_snapshot_ids_and_registry_images` | C12 | deterministic | established |
| 46 | `tests/tools/test_search_auto_multiline.py::TestAutoMultiline::test_newline_regex_matches_across_lines` | C01 | deterministic | probable |
| 47 | `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_case_mismatch_gets_hint` | C01 | deterministic | probable |
| 48 | `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_case_mismatch_hint_names_the_files` | C01 | deterministic | probable |
| 49 | `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_regex_metachar_literal_hint` | C01 | deterministic | probable |
| 50 | `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_hidden_only_match_gets_hint` | C01 | deterministic | probable |
| 51 | `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_probe_path_list_is_capped` | C01 | deterministic | probable |
| 52 | `tests/tools/test_termux_api_detection.py::TestDetectAudioEnvironmentTermuxFallback::test_inconclusive_probes_with_binary_does_not_emit_app_warning` | C13 | deterministic | probable |
| 53 | `tests/tools/test_voice_wsl_pipewire.py::test_wsl_without_forwarding_still_blocks` | C13 | deterministic | probable |
| 54 | `tests/tools/test_transcription_tools.py::TestRunCommandSttIdleTimeout::test_stderr_progress_extends_beyond_timeout` | C17-E | full_suite_only | possible |
| 55 | `tests/tools/test_transcription_tools.py::TestRunCommandSttIdleTimeout::test_silent_stall_still_times_out` | C17-E | full_suite_only | possible |
| 56 | `tests/tools/test_voice_mode.py::TestDetectAudioEnvironment::test_wsl_without_pulse_blocks_voice` | C13 | deterministic | probable |
| 57 | `tests/tools/test_zombie_process_cleanup.py::TestDelegationCleanup::test_timed_out_child_keeps_relay_session_until_its_turn_exits` | C14 | deterministic | probable |
| 58 | `tests/tui_gateway/test_slash_worker_mcp_discovery.py::test_profile_local_mcp_tool_is_visible_in_slash_worker` | C17-E | full_suite_only | possible |

**静态校验：**

```
FAILURE_NODE_IDS_ACCOUNTED_FOR=58/58
FAILURE_NODE_IDS_FULLY_EXPANDED=yes
FAILURE_ASSIGNMENT_DUPLICATES=0
FAILURE_ASSIGNMENT_OMISSIONS=0
CLUSTER_COUNTS_SUM=58
C17_SUBCLASS_COUNTS_SUM=20
```

---

## 4. Failure Clusters（17 个，含完整 node ID）

### C01 — Search 零匹配 hint 缺失（6 tests）

- **Table #:** 46, 47, 48, 49, 50, 51
- **ROOT_CAUSE_CLASS:** `product_regression`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #46 `tests/tools/test_search_auto_multiline.py::TestAutoMultiline::test_newline_regex_matches_across_lines`
  - #47 `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_case_mismatch_gets_hint`
  - #48 `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_case_mismatch_hint_names_the_files`
  - #49 `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_regex_metachar_literal_hint`
  - #50 `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_hidden_only_match_gets_hint`
  - #51 `tests/tools/test_search_zero_match_and_multipath.py::TestZeroMatchProbe::test_probe_path_list_is_capped`
- **后续任务:** `HTR_TOOLS_SEARCH_ZERO_MATCH_HINTS_01` (P1)

### C02 — Agent relay asyncio 绑定（3 tests）

- **Table #:** 1, 2, 3
- **ROOT_CAUSE_CLASS:** `product_regression`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #1 `tests/agent/test_relay_tools.py::test_request_rewrite_reaches_authorized_callback_once`
  - #2 `tests/agent/test_relay_runtime_plugins.py::test_real_binding_layers_project_config_after_explicit_opt_in`
  - #3 `tests/agent/test_relay_llm.py::test_stream_uses_rewritten_request_and_post_intercept_chunks`
- **后续任务:** `HTR_AGENT_RELAY_ASYNCIO_BINDING_01` (P1)

### C03 — hermes_cli update 集成/隔离（6 tests）

- **Table #:** 18, 19, 20, 23, 24, 25
- **ROOT_CAUSE_CLASS:** `test_isolation_failure`
- **EVIDENCE:** `established`
- **Node IDs:**
  - #18 `tests/hermes_cli/test_update_head_moved_gate.py::test_update_success_when_head_moves`
  - #19 `tests/hermes_cli/test_update_autostash.py::test_update_keep_stash_parks_instead_of_restoring`
  - #20 `tests/hermes_cli/test_update_autostash.py::test_update_without_keep_stash_still_restores`
  - #23 `tests/hermes_cli/test_update_yes_flag.py::TestUpdateYesConfigMigration::test_yes_auto_migrates_without_input`
  - #24 `tests/hermes_cli/test_update_yes_flag.py::TestUpdateYesConfigMigration::test_no_yes_flag_still_prompts_in_tty`
  - #25 `tests/hermes_cli/test_update_yes_flag.py::TestUnicodeDecodeErrorInUpdatePrompts::test_unicode_decode_error_in_tty_skips_and_prints_hint`
- **后续任务:** `HTR_CLI_TEST_UPDATE_ISOLATION_01` (P2)

### C04 — Relay shared metrics native API（3 tests）

- **Table #:** 15, 16, 17
- **ROOT_CAUSE_CLASS:** `product_regression`
- **EVIDENCE:** `established`
- **Node IDs:**
  - #15 `tests/hermes_cli/test_relay_shared_metrics_runtime.py::test_real_binding_drives_lifecycle_aggregation_export_and_snapshot`
  - #16 `tests/hermes_cli/test_relay_shared_metrics_runtime.py::test_real_binding_correlates_plugin_approval_denial_to_tool_metric`
  - #17 `tests/hermes_cli/test_relay_shared_metrics_runtime.py::test_real_binding_aggregates_tool_and_approval_timeouts`
- **后续任务:** `HTR_CLI_RELAY_METRICS_NATIVE_01` (P1)

### C05 — User providers model discovery（2 tests）

- **Table #:** 26, 27
- **ROOT_CAUSE_CLASS:** `product_regression`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #26 `tests/hermes_cli/test_user_providers_model_switch.py::test_list_authenticated_providers_enumerates_dict_format_models`
  - #27 `tests/hermes_cli/test_user_providers_model_switch.py::test_section3_probes_no_key_endpoint_with_singular_default_model`
- **后续任务:** `HTR_CLI_USER_PROVIDERS_PROBE_01` (P1)

### C06 — Gateway service systemd unit 路径（1 test）

- **Table #:** 14
- **ROOT_CAUSE_CLASS:** `test_isolation_failure`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #14 `tests/hermes_cli/test_gateway_service.py::TestSystemUnitHermesHome::test_managed_node_makes_system_unit_independent_of_callers_path`
- **后续任务:** `HTR_CLI_GATEWAY_SERVICE_UNIT_01` (P2)

### C07 — Gateway turn lease 超时（1 test）

- **Table #:** 11
- **ROOT_CAUSE_CLASS:** `timeout_budget`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #11 `tests/gateway/test_turn_lease.py::test_full_dispatch_rejects_lease_timeout_without_running_goal_hook`
- **后续任务:** `HTR_TEST_INFRA_GATEWAY_TIMEOUT_01` (P1)

### C08 — OTLP exporter 空事件（2 tests）

- **Table #:** 28, 29
- **ROOT_CAUSE_CLASS:** `product_regression`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #28 `tests/monitoring/test_otlp_exporter.py::test_gateway_health_event_maps_to_span_with_attrs`
  - #29 `tests/monitoring/test_otlp_exporter.py::test_streamer_receives_events_and_respects_filter`
- **后续任务:** `HTR_MONITORING_OTLP_EXPORTER_01` (P2)

### C09 — Honcho pin cache-busting 签名（1 test）

- **Table #:** 22
- **ROOT_CAUSE_CLASS:** `product_regression`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #22 `tests/honcho_plugin/test_pin_peer_name.py::TestPinTransition::test_cache_busting_signature_reflects_pin_peer_name`
- **后续任务:** `HTR_HONCHO_PIN_CACHE_BUST_01` (P2)

### C10 — Photon spectrum patch 缺 node（2 tests）

- **Table #:** 30, 31
- **ROOT_CAUSE_CLASS:** `environment_dependency`
- **EVIDENCE:** `established`
- **Node IDs:**
  - #30 `tests/plugins/platforms/photon/test_spectrum_patch.py::test_sidecar_patch_failure_still_reaches_health_endpoint`
  - #31 `tests/plugins/platforms/photon/test_spectrum_patch.py::test_spectrum_patch_rewrites_the_imessage_mapper`
- **后续任务:** `HTR_PHOTON_NODE_PATH_01` (P2)

### C11 — Photon zombie watchdog 缺 node（3 tests）

- **Table #:** 32, 33, 34
- **ROOT_CAUSE_CLASS:** `environment_dependency`
- **EVIDENCE:** `established`
- **Node IDs:**
  - #32 `tests/plugins/platforms/photon/test_zombie_stream_watchdog.py::test_probe_rejection_classification_is_strict`
  - #33 `tests/plugins/platforms/photon/test_zombie_stream_watchdog.py::test_should_probe_requires_silence_past_threshold_and_cooldown`
  - #34 `tests/plugins/platforms/photon/test_zombie_stream_watchdog.py::test_zombie_requires_probe_proven_connectivity_never_silence_alone`
- **后续任务:** `HTR_PHOTON_NODE_PATH_01` (P2)

### C12 — Modal snapshot lazy SDK（2 tests）

- **Table #:** 44, 45
- **ROOT_CAUSE_CLASS:** `missing_optional_dependency`
- **EVIDENCE:** `established`
- **Node IDs:**
  - #44 `tests/tools/test_modal_snapshot_isolation.py::test_modal_environment_migrates_legacy_snapshot_key_and_uses_snapshot_id`
  - #45 `tests/tools/test_modal_snapshot_isolation.py::test_resolve_modal_image_uses_snapshot_ids_and_registry_images`
- **后续任务:** `HTR_TOOLS_MODAL_SNAPSHOT_MOCK_01` (P2)

### C13 — Voice/WSL/Termux 环境检测（3 tests）

- **Table #:** 52, 53, 56
- **ROOT_CAUSE_CLASS:** `environment_dependency`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #52 `tests/tools/test_termux_api_detection.py::TestDetectAudioEnvironmentTermuxFallback::test_inconclusive_probes_with_binary_does_not_emit_app_warning`
  - #53 `tests/tools/test_voice_wsl_pipewire.py::test_wsl_without_forwarding_still_blocks`
  - #56 `tests/tools/test_voice_mode.py::TestDetectAudioEnvironment::test_wsl_without_pulse_blocks_voice`
- **后续任务:** `HTR_TOOLS_VOICE_WSL_DETECTION_01` (P2)

### C14 — Zombie delegation relay session 时序（1 test）

- **Table #:** 57
- **ROOT_CAUSE_CLASS:** `subprocess_race`
- **EVIDENCE:** `probable`
- **Node IDs:**
  - #57 `tests/tools/test_zombie_process_cleanup.py::TestDelegationCleanup::test_timed_out_child_keeps_relay_session_until_its_turn_exits`
- **后续任务:** `HTR_TEST_INFRA_SUBPROCESS_ISOLATION_01` (P2)

### C15 — LOOP_02 bounded-actions git-blob 守卫（1 test）

- **Table #:** 40
- **ROOT_CAUSE_CLASS:** `pre_existing_failure`（守卫触发）
- **EVIDENCE:** `established`
- **Node IDs:**
  - #40 `tests/test_bounded_actions_phase28a.py::test_no_task23_27_module_changes`
- **机制:** 未提交工作区 `htr/execution_lock.py` bytes 与 git HEAD blob 不一致；守卫预期行为。
- **处置:** 当前失败存在；**若** LOOP_02 变更经授权提交，**必须**在提交后验收中重跑该守卫测试确认；**不得**写成「commit 后应自动消失」。

### C16 — execution_lock 子进程 Queue 超时（1 test）

- **Table #:** 36
- **ROOT_CAUSE_CLASS:** `parallel_resource_contention`
- **EVIDENCE:** `probable`
- **性质:** 独立于 C17；`parallel_only` + full-suite-observed
- **Node IDs:**
  - #36 `tests/htr/test_execution_lock.py::test_subprocess_o_excl_race_exactly_one_winner`
- **EXECUTION_LOCK_PRODUCT_REGRESSION:** `not_established`（无 CPU starvation 直接证据；无多/zero winner 语义错误）
- **后续任务:** `HTR_EXECUTION_LOCK_STABILITY_01` (P2)

### C17 — Full-suite 并行负载 umbrella trigger cluster（20 tests）

- **性质:** 共同触发条件簇，**非**已建立的单一代码根因簇
- **ROOT_CAUSE_CLASS（主类）:** `parallel_resource_contention`（触发条件）
- **EVIDENCE:** `possible`
- **与 C16 互斥计数**

#### C17-A — per-file 300s SIGKILL（2 tests）

- **Node IDs:**
  - #4 `tests/agent/test_sequential_tool_interrupt.py::test_interrupt_abandons_noncooperative_tool`
  - #13 `tests/hermes_cli/test_mcp_startup.py::test_prepare_agent_startup_backgrounds_blocking_mcp_for_chat`

#### C17-B — wall-clock 性能预算（4 tests）

- **Node IDs:**
  - #8 `tests/cron/test_cleanup_timeout.py::test_run_job_bounds_sessiondb_finalization`
  - #37 `tests/run_agent/test_moa_loop_mode.py::test_references_run_in_parallel`
  - #38 `tests/scripts/test_windows_footguns_full_repo_scan.py::test_full_repo_scan_has_no_unsuppressed_windows_footguns`
  - #41 `tests/tools/test_fuzzy_match.py::TestContextAwareCorrectness::test_no_match_on_large_file_is_fast`

#### C17-C — subprocess 同步（1 test）

- **Node IDs:**
  - #12 `tests/hermes_cli/test_active_sessions.py::test_cross_process_acquire_claims_only_one_last_slot`

#### C17-D — gateway/async/terminal 调度（6 tests）

- **Node IDs:**
  - #5 `tests/cli/test_cli_light_mode.py::TestOsc11Da1Fence::test_herdr_style_da1_only_returns_none_without_leak`
  - #6 `tests/cli/test_cli_light_mode.py::TestOsc11Da1Fence::test_slow_inorder_reply_is_consumed_not_leaked`
  - #7 `tests/cli/test_cli_light_mode.py::TestOsc11Da1Fence::test_mute_terminal_times_out_clean`
  - #9 `tests/gateway/test_35994_reset_button_deadlock.py::test_reset_completes_when_cleanup_raises`
  - #10 `tests/gateway/test_session_hygiene.py::test_session_hygiene_timeout_continues_to_agent_and_sets_cooldown`
  - #21 `tests/honcho_plugin/test_session.py::TestDialecticCadenceAdvancesOnSuccess::test_in_flight_thread_is_not_stacked`

#### C17-E — discovery/visibility 间歇（7 tests）

- **Node IDs:**
  - #35 `tests/plugins/memory/test_hindsight_provider.py::TestPrefetchServerRetainVisibility::test_prefetch_waits_for_server_completion_before_recall`
  - #39 `tests/tools/test_approval_interrupt.py::TestApprovalInterrupt::test_interrupt_unblocks_pending_approval_quickly`
  - #42 `tests/tools/test_mcp_discovery_cross_process.py::test_two_processes_each_complete_local_mcp_discovery`
  - #43 `tests/tools/test_mcp_tool_issue_948.py::test_run_stdio_malware_check_times_out_fail_open`
  - #54 `tests/tools/test_transcription_tools.py::TestRunCommandSttIdleTimeout::test_stderr_progress_extends_beyond_timeout`
  - #55 `tests/tools/test_transcription_tools.py::TestRunCommandSttIdleTimeout::test_silent_stall_still_times_out`
  - #58 `tests/tui_gateway/test_slash_worker_mcp_discovery.py::test_profile_local_mcp_tool_is_visible_in_slash_worker`

**C17 子类校验:** 2+4+1+6+7=20

**Cluster 总计:** 6+3+6+3+2+1+1+2+1+2+3+2+3+1+1+1+20=58

---

## 5. execution_lock 专项（C16）

```
EXECUTION_LOCK_CLASSIFICATION=parallel_infrastructure_contention_probable
EXECUTION_LOCK_PRODUCT_REGRESSION=not_established
```

---

## 6. LOOP_02/03 结论

```
HTR_PRODUCT_GOAL_LOOP_03=ACCEPTED
LOOP_03_REOPENED=no
LOOP_03_CONTRACT_COVERAGE_INTACT=yes
LOOP_02_RELATED_FAILURES=2
LOOP_03_RELATED_FAILURES=0
```

---

## 7. 命令审计摘要（部分恢复）

```
COMMAND_LOG_COMPLETE=no
COMMAND_LOG_PARTIALLY_RECOVERED=yes
COMMAND_LOG_SOURCE_ARTIFACTS_AVAILABLE=yes
```

**说明:** 脚本与输出文件可证明 invocation 目标与部分 pytest 摘要；**多数 invocation 的 shell 返回码未写入输出文件**，故不能声称完整 command log。

**审计边界（2026-09-03 文档复核）：**

- 本次文档复核**未重跑**任何历史分诊命令。
- 缺失的历史退出码与 stdout 工件**未重建、未推断**。
- **脚本中存在**的 invocation 定义 ≠ **已证明执行**；**pytest 摘要存在** ≠ **shell 退出码已记录**。
- 表中 `EXIT_CODE=not_recorded` 表示现有记录无法恢复 shell 原始退出码；**不得**用 pytest 的 failed/passed 摘要代替 shell 退出码。

**环境前缀（分诊隔离命令共用）：**

```bash
source "$HOME/.hermes/hermes-agent/venv/bin/activate"
cd "$HOME/hermes-agent-task29-local"
export TZ=UTC LANG=C.UTF-8
# /tmp/htr_triage_run.sh 另设 PYTHONHASHSEED=0
```

**并行:** `scripts/run_tests.sh` 默认 24 workers；追加 `-q`。

### 7.1 原始全量（分诊输入，非本任务重跑）

| 序 | 完整命令 | 返回码 | 输出记录 | 目的 |
|----|----------|--------|----------|------|
| F1 | `scripts/run_tests.sh -q` | 非 0（58 failed） | `tmp_full_suite.log` | 原始全量基线 |

### 7.2 分诊隔离 invocation（来源：`/tmp/htr_triage_*.sh` + 输出文件）

| 序 | 完整命令 | 次数 | EXIT_CODE | 输出记录 | 目的 |
|----|----------|------|-----------|----------|------|
| I1 | `scripts/run_tests.sh tests/htr/test_execution_lock.py::test_subprocess_o_excl_race_exactly_one_winner -q` | 5 | not_recorded | `/tmp/htr_triage_isolated.txt` L71-81：5× Summary 0 failed | C16 单测 |
| I2 | `scripts/run_tests.sh tests/tools/test_search_zero_match_and_multipath.py tests/tools/test_search_auto_multiline.py tests/tools/test_fuzzy_match.py::TestContextAwareCorrectness::test_no_match_on_large_file_is_fast tests/test_bounded_actions_phase28a.py::test_no_task23_27_module_changes -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L1-7 | C01/C15 cluster |
| I3 | `scripts/run_tests.sh tests/agent/test_relay_tools.py tests/agent/test_relay_llm.py tests/agent/test_relay_runtime_plugins.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L8-13：3 files failed | C02 |
| I4 | `scripts/run_tests.sh tests/gateway/test_session_hygiene.py tests/gateway/test_turn_lease.py tests/gateway/test_35994_reset_button_deadlock.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L15-20：turn_lease failed | C07/C17-D |
| I5 | `scripts/run_tests.sh tests/hermes_cli/test_update_yes_flag.py tests/hermes_cli/test_update_autostash.py tests/hermes_cli/test_update_head_moved_gate.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L22-27：6 tests failed | C03 |
| I6 | `scripts/run_tests.sh tests/hermes_cli/test_user_providers_model_switch.py tests/hermes_cli/test_relay_shared_metrics_runtime.py tests/hermes_cli/test_active_sessions.py tests/hermes_cli/test_gateway_service.py tests/hermes_cli/test_mcp_startup.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L29-34：6 tests failed | C04/C05/C06/C17 |
| I7 | `scripts/run_tests.sh tests/cli/test_cli_light_mode.py::TestOsc11Da1Fence -q` | 2 | not_recorded | `/tmp/htr_triage_isolated2.txt` L105-107：2× 3 passed | C17-D |
| I8 | `scripts/run_tests.sh tests/tools/test_approval_interrupt.py tests/tools/test_mcp_tool_issue_948.py tests/run_agent/test_moa_loop_mode.py::test_references_run_in_parallel tests/tools/test_voice_mode.py tests/tools/test_voice_wsl_pipewire.py tests/tools/test_termux_api_detection.py tests/tools/test_zombie_process_cleanup.py tests/tui_gateway/test_slash_worker_mcp_discovery.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L43-48：timing only | C13/C14/C17 |
| I9 | `scripts/run_tests.sh tests/monitoring/test_otlp_exporter.py tests/honcho_plugin/test_session.py tests/honcho_plugin/test_pin_peer_name.py tests/plugins/platforms/photon/test_spectrum_patch.py tests/plugins/platforms/photon/test_zombie_stream_watchdog.py tests/plugins/memory/test_hindsight_provider.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L50-55：8 tests failed | C08/C09/C10/C11/C17 |
| I10 | `scripts/run_tests.sh tests/cron/test_cleanup_timeout.py tests/agent/test_sequential_tool_interrupt.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L57-62：timing only | C17-A/B |
| I11 | `scripts/run_tests.sh tests/tools/test_mcp_discovery_cross_process.py tests/tools/test_modal_snapshot_isolation.py tests/tools/test_transcription_tools.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L64-69：modal 2 failed | C12/C17-E |
| I12 | `scripts/run_tests.sh tests/htr/test_execution_lock.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L82-87 | C16 全文件 |
| I13 | `scripts/run_tests.sh tests/scripts/test_windows_footguns_full_repo_scan.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated.txt` L89-94 | C17-B |
| I14 | `scripts/run_tests.sh tests/tools/test_search_zero_match_and_multipath.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L1-7：5 failed | C01 |
| I15 | `scripts/run_tests.sh tests/tools/test_search_auto_multiline.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L9-15：1 failed | C01 |
| I16 | `scripts/run_tests.sh tests/tools/test_fuzzy_match.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L17-23 | C17-B |
| I17 | `scripts/run_tests.sh tests/hermes_cli/test_active_sessions.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L25-31 | C17-C |
| I18 | `scripts/run_tests.sh tests/hermes_cli/test_mcp_startup.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L33-39 | C17-A |
| I19 | `scripts/run_tests.sh tests/gateway/test_session_hygiene.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L41-47 | C17-D |
| I20 | `scripts/run_tests.sh tests/gateway/test_35994_reset_button_deadlock.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L49-55 | C17-D |
| I21 | `scripts/run_tests.sh tests/tools/test_mcp_discovery_cross_process.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L57-63 | C17-E |
| I22 | `scripts/run_tests.sh tests/tools/test_transcription_tools.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L65-71 | C17-E |
| I23 | `scripts/run_tests.sh tests/plugins/memory/test_hindsight_provider.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L73-79 | C17-E |
| I24 | `scripts/run_tests.sh tests/honcho_plugin/test_session.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L81-87 | C17-D |
| I25 | `scripts/run_tests.sh tests/htr/test_execution_lock.py tests/gateway/test_turn_lease.py tests/hermes_cli/test_relay_shared_metrics_runtime.py tests/tools/test_search_zero_match_and_multipath.py tests/agent/test_relay_tools.py tests/run_agent/test_moa_loop_mode.py tests/plugins/platforms/photon/test_zombie_stream_watchdog.py tests/monitoring/test_otlp_exporter.py tests/scripts/test_windows_footguns_full_repo_scan.py -q` | 1 | not_recorded | `/tmp/htr_triage_isolated2.txt` L89-104：14 tests failed | 并行 stress |

### 7.3 单 node ambiguous 运行（来源：`/tmp/htr_triage_single.sh`）

**脚本路径:** `/tmp/htr_triage_single.sh`  
**输出:** 未写入独立文件；结果摘要存在于分诊会话记录（agent transcript）。  
**EXIT_CODE:** 全部 `not_recorded`（会话记录未保存 shell 退出码）。

| 序 | 完整 pytest target | EXIT_CODE | 会话记录摘要 |
|----|-------------------|-----------|--------------|
| S1 | `tests/tools/test_fuzzy_match.py::TestContextAwareCorrectness::test_no_match_on_large_file_is_fast` | not_recorded | Summary: 1 passed |
| S2 | `tests/cron/test_cleanup_timeout.py::test_run_job_bounds_sessiondb_finalization` | not_recorded | Summary: 1 passed |
| S3 | `tests/agent/test_sequential_tool_interrupt.py::test_interrupt_abandons_noncooperative_tool` | not_recorded | Summary: 1 passed |
| S4 | `tests/tools/test_approval_interrupt.py::TestApprovalInterrupt::test_interrupt_unblocks_pending_approval_quickly` | not_recorded | Summary: 1 passed |
| S5 | `tests/tools/test_mcp_tool_issue_948.py::test_run_stdio_malware_check_times_out_fail_open` | not_recorded | Summary: 1 passed |
| S6 | `tests/tools/test_termux_api_detection.py::TestDetectAudioEnvironmentTermuxFallback::test_inconclusive_probes_with_binary_does_not_emit_app_warning` | not_recorded | pytest FAILED |
| S7 | `tests/tools/test_voice_wsl_pipewire.py::test_wsl_without_forwarding_still_blocks` | not_recorded | pytest FAILED |
| S8 | `tests/tools/test_zombie_process_cleanup.py::TestDelegationCleanup::test_timed_out_child_keeps_relay_session_until_its_turn_exits` | not_recorded | pytest FAILED |
| S9 | `tests/tui_gateway/test_slash_worker_mcp_discovery.py::test_profile_local_mcp_tool_is_visible_in_slash_worker` | not_recorded | PASS（runner 标记 1× FLAKY retry） |
| S10 | `tests/run_agent/test_moa_loop_mode.py::test_references_run_in_parallel` | not_recorded | Summary: 1 passed |
| S11 | `tests/plugins/memory/test_hindsight_provider.py::TestPrefetchServerRetainVisibility::test_prefetch_waits_for_server_completion_before_recall` | not_recorded | Summary: 1 passed |
| S12 | `tests/gateway/test_35994_reset_button_deadlock.py::test_reset_completes_when_cleanup_raises` | not_recorded | Summary: 1 passed |
| S13 | `tests/gateway/test_session_hygiene.py::test_session_hygiene_timeout_continues_to_agent_and_sets_cooldown` | not_recorded | Summary: 1 passed |
| S14 | `tests/hermes_cli/test_active_sessions.py::test_cross_process_acquire_claims_only_one_last_slot` | not_recorded | Summary: 1 passed |
| S15 | `tests/hermes_cli/test_mcp_startup.py::test_prepare_agent_startup_backgrounds_blocking_mcp_for_chat` | not_recorded | Summary: 1 passed |
| S16 | `tests/honcho_plugin/test_session.py::TestDialecticCadenceAdvancesOnSuccess::test_in_flight_thread_is_not_stacked` | not_recorded | Summary: 1 passed |
| S17 | `tests/tools/test_transcription_tools.py::TestRunCommandSttIdleTimeout::test_silent_stall_still_times_out` | not_recorded | Summary: 1 passed |
| S18 | `tests/tools/test_transcription_tools.py::TestRunCommandSttIdleTimeout::test_stderr_progress_extends_beyond_timeout` | not_recorded | Summary: 1 passed |
| S19 | `tests/tools/test_mcp_discovery_cross_process.py::test_two_processes_each_complete_local_mcp_discovery` | not_recorded | Summary: 1 passed |

**无法恢复字段:** 全部 isolation invocation 的 shell 退出码（`EXIT_CODE=not_recorded`）；S1–S19 无独立 stdout 文件路径。

**源 artifact 路径（仅证明存在，不自动等于全部 invocation 已执行）:**

- `/tmp/htr_triage_run.sh` — invocation 定义 + 部分输出摘要
- `/tmp/htr_triage_run2.sh` — invocation 定义 + 部分输出摘要
- `/tmp/htr_triage_single.sh` — invocation 定义；执行摘要仅见于 agent transcript
- `/tmp/htr_triage_isolated.txt` — I1–I13 部分 pytest 摘要
- `/tmp/htr_triage_isolated2.txt` — I14–I25 部分 pytest 摘要
- `tmp_full_suite.log` — F1 原始全量 pytest 摘要

---

## 8. 工作树审计摘要

```
WORKTREE_AUDIT_COMPLETE=no
WORKTREE_AUDIT_SUMMARY_COMPLETE=yes
WORKTREE_AUDIT_FULL_RAW_OUTPUT_AVAILABLE=no
WORKTREE_AUDIT_SOURCE=existing_record_plus_readonly_observation_2026-09-03
```

**说明:** TRIAGE_START 与 TRIAGE_END 的**完整原始 stdout 未单独落盘**；下列 HEAD 与 tracked/untracked 摘要来自分诊会话既有记录，并与 **2026-09-03 文档补正时的只读 `git` 观察**一致。当前 `git status` **不能**冒充历史 START 快照文件。

```
CURRENT_READ_ONLY_OBSERVATION_DATE=2026-09-03
CURRENT_OBSERVATION_IS_NOT_HISTORICAL_TRIAGE_RAW_OUTPUT=yes
```

```
TRIAGE_START_HEAD=8081ad24a3bf5e428fe9f2aa3f9f2b8efd19f83b
TRIAGE_END_HEAD=8081ad24a3bf5e428fe9f2aa3f9f2b8efd19f83b
TRIAGE_START_GIT_STATUS_RAW=not_persisted
TRIAGE_END_GIT_STATUS_RAW=not_persisted
TRIAGE_START_DIFF_STAT_RAW=not_persisted
TRIAGE_END_DIFF_STAT_RAW=not_persisted
```

### 8.1 tracked modified（6 files，START=END 相同）

```
M htr/__init__.py
M htr/execution_lock.py
M htr/state.py
M tests/htr/test_execution_lock.py
M tests/htr/test_state.py
M tests/test_mcp_serve.py
```

**diff --stat（START=END 相同）：**

```
 htr/__init__.py                  | 10 +++++++
 htr/execution_lock.py            | 60 ++++++++++++++++++++++++++++++++++++++++
 htr/state.py                     | 10 ++++++-
 tests/htr/test_execution_lock.py | 40 +++++++++++++++++++++++++++
 tests/htr/test_state.py          |  7 +++++
 tests/test_mcp_serve.py          |  8 ++++--
 6 files changed, 132 insertions(+), 3 deletions(-)
```

### 8.2 untracked — LOOP_02/03 intended

```
htr/failure_fingerprint.py
htr/goal_runtime.py
htr/heal_record.py
tests/htr/test_goal_loop_heal.py
tests/htr/test_goal_runtime.py
```

### 8.3 untracked — triage/closure artifacts

```
tmp_full_suite.log
tmp_closure_collect.sh
tmp_closure_results.txt
tmp_closure_run.sh
```

### 8.4 untracked — unrelated pre-existing noise

```
docs/agent-lessons.md
nul
:USERPROFILE.sshknown_hosts
```

### 8.5 2026-09-03 静态复核只读观察

- **只读命令:** `git rev-parse HEAD`, `git status --short`, `git diff --stat`, `git diff --name-status`（2026-09-03）
- **结果:** HEAD 与 §8.1–§8.4 摘要一致；**未修改** tracked 产品/测试文件
- **本报告文件:** `docs/HTR_FULL_SUITE_FAILURE_TRIAGE_01_REPORT.md` 为 untracked 文档（非产品代码）

```
CODE_CHANGES_DURING_REPORT_REVIEW=no
TESTS_RUN_DURING_REPORT_REVIEW=no
PRODUCT_OR_TEST_CODE_CHANGED=no
```

---

## 9. 附录 A — 15 FLAKY 文件

来源：`tmp_full_suite.log` L4911–5897

1. `tests/gateway/test_abandoned_turn_process_cleanup.py`
2. `tests/gateway/test_approve_deny_commands.py`
3. `tests/gateway/test_compression_concurrent_sessions.py`
4. `tests/gateway/test_platform_registry.py`
5. `tests/gateway/test_telegram_media_read_timeout.py`
6. `tests/gateway/test_telegram_network_reconnect.py`
7. `tests/hermes_cli/test_gemini_provider.py`
8. `tests/honcho_plugin/test_network_isolation.py`
9. `tests/htr/test_marker_disposition.py`
10. `tests/htr/test_reconciliation_cases.py`
11. `tests/htr/test_recovery_runs.py`
12. `tests/run_agent/test_request_client_reuse_abort_races.py`
13. `tests/test_pty_session.py`
14. `tests/tools/test_approval.py`
15. `tests/tools/test_threat_patterns.py`

---

## 10. 附录 B — 17 NO_RUN 文件

来源：`tmp_full_suite.log` L28974–28991

1. `tests/agent/test_api_content_sidecar.py`
2. `tests/agent/test_compression_concurrent_fork.py`
3. `tests/agent/test_compression_rotation_state.py`
4. `tests/gateway/test_agent_cache.py`
5. `tests/gateway/test_config.py`
6. `tests/hermes_cli/test_cmd_update.py`
7. `tests/hermes_cli/test_doctor.py`
8. `tests/hermes_cli/test_model_switch_custom_providers.py`
9. `tests/plugins/platforms/photon/test_url_send_path.py`
10. `tests/hermes_cli/test_web_server.py`
11. `tests/run_agent/test_run_agent.py`
12. `tests/run_agent/test_run_agent_codex_responses.py`
13. `tests/run_agent/test_streaming.py`
14. `tests/test_tui_gateway_server.py`
15. `tests/tools/test_browser_camofox.py`
16. `tests/tools/test_browser_extension_router_wiring.py`
17. `tests/tools/test_execution_flag_detection.py`

---

## 11. 最终判定矩阵

```
REVIEW_DATE=2026-09-03
HTR_FULL_SUITE_FAILURE_TRIAGE_01=COMPLETED
REPORT_CONSISTENCY_REVIEW=CONDITIONALLY_ACCEPTED
FAILURE_ACCOUNTING_REVIEW=PASSED
ROOT_CAUSE_CLASSIFICATION_REVIEW=PASSED
DOCUMENTATION_COMPLETENESS_REVIEW=PARTIAL
HUMAN_SIGNOFF_READY=no
TRIAGE_ONLY=yes

ORIGINAL_FULL_SUITE=39143_PASSED_58_FAILED_271_SKIPPED
ORIGINAL_FULL_SUITE_COLLECTED=39472
FAILURE_NODE_IDS_ACCOUNTED_FOR=58/58
FAILURE_NODE_IDS_FULLY_EXPANDED=yes
FAILURE_ASSIGNMENT_DUPLICATES=0
FAILURE_ASSIGNMENT_OMISSIONS=0
FAILURE_CLUSTERS_IDENTIFIED=17
CLUSTER_COUNTS_SUM=58
C17_SUBCLASS_COUNTS_SUM=20

DETERMINISTIC_FAILURES=37
INTERMITTENT_FAILURES=21
GENERAL_FULL_SUITE_LOAD_FAILURES=20
EXECUTION_LOCK_FULL_SUITE_PARALLEL_FAILURES=1
CLASSIFICATION_OVERLAP_DOCUMENTED=yes

ESTABLISHED_ROOT_CAUSE_FAILURES=17
PROBABLE_ROOT_CAUSE_FAILURES=21
POSSIBLE_ROOT_CAUSE_FAILURES=20
UNKNOWN_ROOT_CAUSE_FAILURES=0
ROOT_CAUSE_EVIDENCE_COUNTS_SUM=58
UNRESOLVED_FAILURES=20
ROOT_CAUSE_TRIAGE_COMPLETE=yes

LOOP_02_RELATED_FAILURES=2
LOOP_03_RELATED_FAILURES=0
EXECUTION_LOCK_CLASSIFICATION=parallel_infrastructure_contention_probable
EXECUTION_LOCK_PRODUCT_REGRESSION=not_established

HTR_PRODUCT_GOAL_LOOP_03=ACCEPTED
LOOP_03_REOPENED=no
LOOP_03_CONTRACT_COVERAGE_INTACT=yes
FULL_SUITE_GREEN=no

FLAKY_FILE_LIST_ACCOUNTED_FOR=yes
NO_RUN_FILE_LIST_ACCOUNTED_FOR=yes
COMMAND_LOG_COMPLETE=no
COMMAND_LOG_PARTIALLY_RECOVERED=yes
COMMAND_LOG_SOURCE_ARTIFACTS_AVAILABLE=yes
WORKTREE_AUDIT_COMPLETE=no
WORKTREE_AUDIT_SUMMARY_COMPLETE=yes
WORKTREE_AUDIT_FULL_RAW_OUTPUT_AVAILABLE=no

CODE_CHANGES_DURING_REPORT_REVIEW=no
TESTS_RUN_DURING_REPORT_REVIEW=no
TESTS_MODIFIED=no
TESTS_SKIPPED_OR_XFAILED=no
TIMEOUTS_CHANGED=no
DEPENDENCIES_CHANGED=no
REAL_HERMES_HOME_UNCHANGED=yes
PROTECTED_RUN_UNCHANGED=yes
COMMIT_CREATED=no
MERGE_PERFORMED=no
PUSH_PERFORMED=no
LOOP_04_STARTED=no
COMMIT_MERGE_PUSH_AUTHORIZED=no
FIX_IMPLEMENTATION_AUTHORIZED=no
```

**HUMAN_SIGNOFF_READY=no 原因:** 命令审计与工作树审计仅能部分恢复；failure accounting 与 root-cause 分类已完成。`HTR_FULL_SUITE_FAILURE_TRIAGE_01=COMPLETED` 与 `HUMAN_SIGNOFF_READY=no` **不矛盾**：前者表示技术分诊与证据分级已完成，后者表示历史原始审计证据不完整，尚不满足无例外的全面人工签字条件。

---

## 12. COMPLETED 与 HUMAN_SIGNOFF_READY 的关系

```
HTR_FULL_SUITE_FAILURE_TRIAGE_01=COMPLETED
HUMAN_SIGNOFF_READY=no
```

- **COMPLETED** 表示：58 失败归属、17 cluster 划分、根因证据分级、后续任务建议均已完成。
- **HUMAN_SIGNOFF_READY=no** 表示：历史 command log 与 worktree 原始审计证据不完整，尚不满足**无例外**的全面人工签字条件。
- **UNRESOLVED_FAILURES=20** 表示 C17 的 20 个失败主证据为 `possible`（触发条件已记录，单一共同代码根因未 established）；**不表示**已修复或已解决。

### 历史证据缺口规则

若要消除当前文档证据缺口，**必须找到并纳入当时已经保存的**完整 stdout、退出码及历史工作树原始输出。若这些历史证据不存在，**不能**追溯性补全。未来经授权重新运行所形成的记录属于**新的验证证据**，不得作为原始分诊记录的替代或冒充。人工可以基于已披露的证据缺口进行**例外接受**，但**不得**据此将 `COMMAND_LOG_COMPLETE`、`WORKTREE_AUDIT_COMPLETE` 或 `WORKTREE_AUDIT_FULL_RAW_OUTPUT_AVAILABLE` 改为 `yes`。

---

## 13. Human Exception Acceptance

```
HUMAN_SIGNOFF_READY=no
STANDARD_FULL_SIGNOFF_AVAILABLE=no
EXCEPTION_SIGNOFF_AVAILABLE=yes
HUMAN_EXCEPTION_DECISION_RECORDED=yes
HUMAN_EXCEPTION_ACCEPTED=yes
HUMAN_EXCEPTION_DECISION=ACCEPT_WITH_DOCUMENTED_EXCEPTIONS
HUMAN_EXCEPTION_ACCEPTANCE_DATE=2026-09-03
AUTHORIZED_HUMAN_REVIEWER=liuqiong
AUTHORIZED_HUMAN_REVIEWER_ROLE=Project Owner
AUTHORIZED_HUMAN_SIGNATURE=/s/ liuqiong

COMMAND_LOG_COMPLETE=no
WORKTREE_AUDIT_COMPLETE=no
WORKTREE_AUDIT_FULL_RAW_OUTPUT_AVAILABLE=no
DOCUMENTATION_COMPLETENESS_REVIEW=PARTIAL
REPORT_CONSISTENCY_REVIEW=CONDITIONALLY_ACCEPTED
FULL_SUITE_GREEN=no
```

**Exception scope（可例外接受的范围）:**

- failure accounting 已完成（58/58，0 duplicate，0 omission）
- root-cause classification 已完成（17 cluster，证据分级合计 58）
- command log 只能部分恢复（`COMMAND_LOG_COMPLETE=no`）
- worktree audit 只有摘要，缺少完整历史 raw output（`WORKTREE_AUDIT_COMPLETE=no`）
- full suite 仍为 red（`FULL_SUITE_GREEN=no`）
- 58 个失败**未修复**；20 个 `UNRESOLVED_FAILURES`（C17）**未解决**
- 本次**未运行测试**；**未实施产品修复**

**Decision options:**

- [ ] **REJECT** — 要求找到更多既有历史证据后重新复审。
- [x] **ACCEPT_WITH_DOCUMENTED_EXCEPTIONS** — 接受 failure accounting 与 root-cause classification，并明确接受 command log / worktree audit 的历史证据缺口。
- [ ] **AUTHORIZE_NEW_VALIDATION_RUN** — 授权未来创建新的验证记录；该记录**不得**冒充 2026-09-03 之前的原始分诊记录。

**签字字段:**

| 字段 | 值 |
|------|-----|
| Reviewer | liuqiong |
| Role | Project Owner |
| Decision | ACCEPT_WITH_DOCUMENTED_EXCEPTIONS |
| Signature | /s/ liuqiong |
| Date | 2026-09-03 |
| Exception notes | 本人接受已经完成的 failure accounting 和 root-cause classification，并明确接受已披露的 command log 与 worktree audit 历史证据缺口。本次例外接受不表示 full suite 已转绿、不表示相关失败已经修复，也不表示产品已经达到 release readiness。 |

```
EXCEPTION_SIGNOFF_DOES_NOT_CHANGE_COMMAND_LOG_COMPLETE=yes
EXCEPTION_SIGNOFF_DOES_NOT_CHANGE_WORKTREE_AUDIT_COMPLETE=yes
EXCEPTION_SIGNOFF_PRESERVES_HUMAN_SIGNOFF_READY_NO=yes
EXCEPTION_SIGNOFF_DOES_NOT_MARK_FULL_SUITE_GREEN=yes
```

**说明:** 「例外接受」仅表示人工接受已披露的文档缺口；**不代表**完整审计已恢复，也**不会**将 `HUMAN_SIGNOFF_READY` 改为 `yes`。`HUMAN_EXCEPTION_ACCEPTED=yes` 与 `HUMAN_SIGNOFF_READY=no` **不矛盾**：前者表示 Project Owner 已授权接受已披露缺口；后者表示报告仍不符合无例外的全面签字条件。

---

**人工例外接受已记录（2026-09-03）。未实施任何修复。**

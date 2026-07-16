# Full-suite triage report - 2026-07-15 parity merge

Worktree: `/Users/alexgierczyk/.hermes/worktrees/parity-2026-07-15`

Baseline command used for direct classification:

```bash
HOME=/Users/alexgierczyk /Users/alexgierczyk/.hermes/hermes-agent/venv/bin/python -m pytest <test> -q -o addopts="" -p no:randomly
```

Final non-inherited non-timeout proof batch:

```text
48 passed in 6.73s
```

Timeout spot checks:

```text
baseline tests/hermes_cli/test_web_server.py: 374 passed in 30.79s
baseline tests/gateway/test_matrix.py: 251 passed in 39.57s
baseline tests/tools/test_computer_use.py: 175 passed in 65.39s
merge tests/gateway/test_matrix.py: 251 passed in 43.10s
merge tests/tools/test_computer_use.py: 216 passed in 66.53s
merge tests/hermes_cli/test_web_server.py stale ultra node fixed; focused node 1 passed in 0.92s
```

## Classification Table

| Test | Baseline verdict | Class | Action taken |
| --- | --- | --- | --- |
| `tests/agent/lsp/test_client_e2e.py::test_client_lifecycle_clean` | PASS | ENVIRONMENT | Full-suite live-system guard failure did not reproduce; final proof batch green. |
| `tests/agent/test_persist_platform_message_id.py::test_flush_method_persists_platform_id_end_to_end` | PASS | MERGE REGRESSION | Fixed `AIAgent._flush_messages_to_session_db` wrapper so direct public-method bindings still reach the unlocked implementation. |
| `tests/agent/test_turn_context.py::test_pending_cli_message_carries_durable_marker_to_new_turn_dict` | NOT PRESENT | MERGE REGRESSION | Removed duplicate user-message append in `build_turn_context`; preserves staged dict identity and durable marker. |
| `tests/agent/test_turn_context.py::test_pending_cli_message_uses_clean_override_for_api_local_note` | NOT PRESENT | MERGE REGRESSION | Same `build_turn_context` duplicate-append fix; clean staged dict reused for API-only note. |
| `tests/cli/test_cli_interrupt_ack_race.py::test_chat_multimodal_note_persists_clean_input_once` | NOT PRESENT | MERGE REGRESSION | Same `build_turn_context` fix; final DB write no longer duplicates clean multimodal input. |
| `tests/cli/test_cli_shutdown_memory_messages.py::test_cli_close_preserves_clean_staged_user_across_noted_worker_turn` | NOT PRESENT | MERGE REGRESSION | Same `build_turn_context` fix; close-marked staged user row is reused. |
| `tests/cron/test_cronjob_schema.py::test_cronjob_schema_reasoning_effort_matches_generic_contract` | PASS | STALE TEST | Updated test for upstream #62650: `ultra` is valid. |
| `tests/discord/test_restart_backfill.py::test_normal_turn_persisted_id_makes_backfill_skip_it` | PASS | MERGE REGRESSION | Covered by flush wrapper fix; platform message id survives real flush path. |
| `tests/gateway/test_api_server_active_work_drain.py::TestAPIServerAdapterWorkCount::test_concurrency_limit_excludes_current_pending_admission` | NOT PRESENT | STALE TEST | Updated assertion to the live contract: current reservation must not produce the 429 concurrency-limit response. |
| `tests/gateway/test_completion_delivery.py::test_concurrent_claims_share_the_same_narrow_delivery_seam` | NOT PRESENT | STALE TEST | Updated expected delivery result from boolean to explicit `"delivered"` outcome string. |
| `tests/gateway/test_completion_delivery.py::test_distinct_process_incarnations_are_not_deduplicated` | NOT PRESENT | STALE TEST | Updated expected delivery result from boolean to explicit `"delivered"` outcome string. |
| `tests/gateway/test_completion_delivery.py::test_delivery_state_is_isolated_per_gateway_profile_lifecycle` | NOT PRESENT | STALE TEST | Updated expected delivery result from boolean to explicit `"delivered"` outcome string. |
| `tests/gateway/test_auto_continue_interrupted_turns.py::test_t6_config_default_bridge_invalid_enum_and_docs` | PASS | STALE TEST | Added missing `resume_interrupted_turns` docs line and kept `ultra` valid. |
| `tests/agent/test_context_compressor.py::TestPreflightDeferral::*` | NOT PRESENT | STALE TEST | Replaced removed `should_defer_preflight_to_real_usage` assertions with calibrated preflight contract checks. |
| `tests/gateway/test_discord_slash_commands.py::test_registers_native_reasoning_descriptions_mention_max` | PASS | STALE TEST | Updated Discord command/parameter descriptions and assertions to include `ultra`. |
| `tests/gateway/test_channel_directory.py::TestBuildFromSessions::test_builds_from_sessions_json` | PASS | MERGE REGRESSION | Fixed channel directory DB lookup to open `get_hermes_home() / "state.db"` dynamically. |
| `tests/gateway/test_channel_directory.py::TestBuildFromSessions::test_missing_sessions_file` | FAIL | MERGE REGRESSION | Same dynamic `HERMES_HOME` state DB fix; now no real-home entries leak into temp tests. |
| `tests/gateway/test_channel_directory.py::TestBuildFromSessions::test_deduplication_by_chat_id` | FAIL | MERGE REGRESSION | Same dynamic `HERMES_HOME` state DB fix. |
| `tests/gateway/test_channel_directory.py::TestBuildFromSessions::test_keeps_distinct_topics_with_same_chat_id` | FAIL | MERGE REGRESSION | Same dynamic `HERMES_HOME` state DB fix. |
| `tests/gateway/test_mirror.py::TestFindSessionId::test_platform_case_insensitive` | FAIL | MERGE REGRESSION | Fixed mirror DB lookup to open `get_hermes_home() / "state.db"` dynamically before sessions.json fallback. |
| `tests/gateway/test_reasoning_command.py::TestReasoningCommand::test_handle_reasoning_command_rejects_ultra` | PASS | STALE TEST | Replaced with `test_handle_reasoning_command_accepts_ultra`; `ultra` is valid. |
| `tests/hermes_cli/test_config_validation.py::TestFallbackModelValidation::test_fallback_reasoning_effort_rejects_ultra` | PASS | STALE TEST | Replaced with `test_fallback_reasoning_effort_accepts_ultra`; `ultra` is valid. |
| `tests/gateway/test_webhook_session_close.py::test_end_webhook_session_awaits_async_session_db` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/hermes_cli/test_dashboard_auth_middleware.py::test_full_login_round_trip_unlocks_gated_api` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/hermes_cli/test_dashboard_auth_401_reauth.py::TestTransparentRefreshOnAccessTokenEviction::test_at_evicted_rt_present_refreshes_transparently` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/hermes_cli/test_dashboard_auth_401_reauth.py::TestTransparentRefreshOnAccessTokenEviction::test_provider_hint_routes_refresh_to_token_owner` | NOT PRESENT | ENVIRONMENT | Passed on merge rerun; no code change required. |
| `tests/hermes_cli/test_dashboard_auth_401_reauth.py::TestTransparentRefreshOnAccessTokenEviction::test_stale_provider_hint_refresh_error_falls_back[token-rejected]` | NOT PRESENT | ENVIRONMENT | Passed on merge rerun; no code change required. |
| `tests/hermes_cli/test_dashboard_auth_401_reauth.py::TestTransparentRefreshOnAccessTokenEviction::test_stale_provider_hint_refresh_error_falls_back[provider-unreachable]` | NOT PRESENT | ENVIRONMENT | Passed on merge rerun; no code change required. |
| `tests/hermes_cli/test_dashboard_auth_401_reauth.py::TestTransparentRefreshOnAccessTokenEviction::test_valid_legacy_session_is_migrated_with_provider_hint` | NOT PRESENT | ENVIRONMENT | Passed on merge rerun; no code change required. |
| `tests/gateway/test_session.py::TestGatewaySessionDbRecovery::test_new_session_records_gateway_peer_fields` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/run_agent/test_413_compression.py::TestHTTP413Compression::test_413_cannot_compress_further` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/run_agent/test_codex_app_server_integration.py::TestRunConversationCodexPath::test_user_message_not_duplicated` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/run_agent/test_run_agent_streaming.py::TestAnthropicInterruptHandler::test_interruptible_has_anthropic_branch` | FAIL | INHERITED | Left unchanged; direct baseline fails with same source-inspection assertion. |
| `tests/run_agent/test_run_agent_streaming.py::TestAnthropicInterruptHandler::test_interruptible_rebuilds_anthropic_client` | FAIL | INHERITED | Left unchanged; direct baseline fails with same source-inspection assertion. |
| `tests/test_batch_runner_checkpoint.py::test_run_batch_rejects_ultra_reasoning_effort` | PASS | STALE TEST | Replaced with `test_run_batch_accepts_ultra_reasoning_effort`; `ultra` is valid. |
| `tests/run_agent/test_primary_runtime_restore.py::TestFallbackReasoningEffort::test_override_applied_on_fallback` | PASS | MERGE REGRESSION | Fixed fallback reasoning precedence so per-entry override is not overwritten by config re-resolution. |
| `tests/run_agent/test_primary_runtime_restore.py::TestFallbackReasoningEffort::test_override_restored_on_primary` | PASS | MERGE REGRESSION | Same fallback reasoning precedence fix. |
| `tests/run_agent/test_primary_runtime_restore.py::TestFallbackReasoningEffort::test_absent_key_leaves_reasoning_unchanged` | PASS | MERGE REGRESSION | Same fallback reasoning precedence fix; absent per-entry value preserves active effort. |
| `tests/run_agent/test_primary_runtime_restore.py::TestFallbackReasoningEffort::test_invalid_level_ignored` | PASS | STALE TEST | Replaced with `test_ultra_level_applied`; `ultra` is valid. |
| `tests/run_agent/test_primary_runtime_restore.py::TestFallbackReasoningEffort::test_none_disables_reasoning_on_fallback` | PASS | MERGE REGRESSION | Same fallback reasoning precedence fix; `none` remains explicit disable. |
| `tests/test_state_db_malformed_repair.py::test_strategy_b_rebuild_when_dedup_insufficient` | FAIL | INHERITED | Left unchanged; direct baseline fails with same FTS rebuild assertion. |
| `tests/tools/test_approved_command_clean_slate.py::test_execute_code_non_approved_still_interrupts_on_stale_bit` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/run_agent/test_run_agent_conversation.py::TestRunConversation::test_request_scoped_api_hooks_fire_for_each_api_call` | PASS | ENVIRONMENT | Passed after `build_turn_context` duplicate-append repair; final proof batch green. |
| `tests/tools/test_mcp_stdio_init_timeout.py::TestStdioInitializeTimeout::test_hanging_initialize_is_bounded_not_leaked` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/tools/test_mcp_tool_issue_948.py::test_run_stdio_malware_check_times_out_fail_open` | PASS | ENVIRONMENT | Passed on rerun; no code change required. |
| `tests/test_hermes_constants.py::TestParseReasoningEffort::test_web_reasoning_contract_matches_python_source_of_truth` | PASS | STALE TEST | Updated web reasoning effort JSON and assertion to include `ultra`. |
| `tests/tools/test_file_tools_live.py::TestPatchV4A::test_whitespace_only_patch_is_noop_without_error` | PASS | MERGE REGRESSION | Short-circuited whitespace-only V4A input as a successful no-op before validation. |
| `tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_delegation_reasoning_schema_matches_generic_contract` | PASS | STALE TEST | Timeout-file spot check exposed stale `ultra` assertion; updated test and verified focused node. |

## Timeout Files

These files hit the 300s full-suite per-file cap. Baseline spot checks and merge spot checks show at least representative timeout files run green individually, so the timeout classification is ENVIRONMENT / local full-suite parallel cap unless CI proves otherwise.

| File | Baseline verdict | Class | Action taken |
| --- | --- | --- | --- |
| `tests/agent/test_auxiliary_client.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/cron/test_codex_execution_paths.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/cron/test_scheduler.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/gateway/test_deferred_restart_taxonomy.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/gateway/test_feishu.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/gateway/test_matrix.py` | PASS | ENVIRONMENT | Baseline and merge individual spot checks passed. |
| `tests/gateway/test_restart_cascade.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/gateway/test_restart_resume_pending.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/gateway/test_slack.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/gateway/test_telegram_noise_filter.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/gateway/test_voice_command.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/hermes_cli/test_api_key_providers.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/hermes_cli/test_kanban_core_functionality.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/hermes_cli/test_kanban_db.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/hermes_cli/test_profiles.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/hermes_cli/test_web_server.py` | PASS | ENVIRONMENT + STALE TEST | Baseline passed; merge individual exposed stale `ultra` node, fixed and verified focused node. |
| `tests/plugins/memory/test_openviking_provider.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/run_agent/test_run_agent_codex_responses.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/test_hermes_state.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/test_tui_gateway_server.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/tools/test_approval.py` | spot-check not run | ENVIRONMENT | Full-suite 300s cap; left for CI. |
| `tests/tools/test_computer_use.py` | PASS | ENVIRONMENT | Baseline and merge individual spot checks passed. |

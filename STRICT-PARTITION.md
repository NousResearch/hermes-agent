# Strict single-assignment partition (no double-counting)

Every one of the 160 delta files (v0.16.0 3c231eb .. HEAD, repo root =
the user's ./src/ checkout) is assigned to EXACTLY ONE bucket. A file touched by
multiple PRs is assigned to its lowest-numbered canonical owner; co-owners are
listed for transparency but do not double-count.

| file | bucket | canonical owner | co-owners |
|---|---|---|---|
| `.project-intel/BENCHMARKS.md` | DISCARD (project-intel) | — | — |
| `.project-intel/DATAFLOW_MAP.md` | DISCARD (project-intel) | — | — |
| `.project-intel/FEATURE_MAP.md` | DISCARD (project-intel) | — | — |
| `.project-intel/FLOW_MAP.md` | DISCARD (project-intel) | — | — |
| `.project-intel/FUNNEL_MAP.md` | DISCARD (project-intel) | — | — |
| `.project-intel/HEALTH.md` | DISCARD (project-intel) | — | — |
| `.project-intel/INDEX_STATE.md` | DISCARD (project-intel) | — | — |
| `.project-intel/PROJECT_INTELLIGENCE.md` | DISCARD (project-intel) | — | — |
| `.project-intel/QUERY_ROUTING.md` | DISCARD (project-intel) | — | — |
| `.project-intel/RESOURCE_INVENTORY.md` | DISCARD (project-intel) | — | — |
| `.project-intel/UI_FLOW_MAP.md` | DISCARD (project-intel) | — | — |
| `.project-intel/indexes/project-index.sqlite` | DISCARD (project-intel) | — | — |
| `agent/agent_init.py` | PR | #48065 | #49184, #49917, #50073, #50296 |
| `agent/agent_runtime_helpers.py` | PR | #49184 | — |
| `agent/anthropic_adapter.py` | PR | #48024 | #50064 |
| `agent/anthropic_adapter.py.bak.20260603100620` | DISCARD (bak) | — | — |
| `agent/auto_router.py` | PR | #50031 | — |
| `agent/autopilot/__init__.py` | PR | #49917 | — |
| `agent/autopilot/council_gate.py` | PR | #49917 | — |
| `agent/autopilot/driver.py` | PR | #49917 | — |
| `agent/auxiliary_client.py` | PR | #49184 | #50064 |
| `agent/background_review.py` | PR | #50296 | — |
| `agent/chat_completion_helpers.py` | PR | #50055 | — |
| `agent/codex_responses_adapter.py.bak.20260603100620` | DISCARD (bak) | — | — |
| `agent/codex_version.py` | PR | #50038 | — |
| `agent/context_engine.py` | PR | #50053 | — |
| `agent/context_engine.py.bak.20260607_231325` | DISCARD (bak) | — | — |
| `agent/conversation_loop.py` | PR | #49184 | #49917, #50073, #50155 |
| `agent/conversation_loop.py.bak.20260607_231325` | DISCARD (bak) | — | — |
| `agent/copilot_acp_client.py` | PR | #50064 | — |
| `agent/gemini_cloudcode_adapter.py` | PR | #49644 | — |
| `agent/model_metadata.py` | PR | #50064 | — |
| `agent/model_metadata.py.bak.20260603100620` | DISCARD (bak) | — | — |
| `agent/models_dev.py` | PR | #49449 | — |
| `agent/prompt_builder.py` | PR | #49917 | — |
| `agent/subdirectory_hints.py` | SUPERSEDED | #29433 | upstream PR ships same fix + test |
| `agent/system_prompt.py` | PR | #48101 | #49917 |
| `agent/system_prompt.py.bak.20260614_131421` | DISCARD (bak) | — | — |
| `agent/system_prompt_prelude.py` | PR | #48101 | — |
| `agent/tool_executor.py` | PR | #49917 | #50021 |
| `agent/tool_trace_sidecar.py` | PR | #50021 | — |
| `agent/transports/chat_completions.py` | PR | #49644 | — |
| `agent/transports/codex.py` | PR | #50038 | — |
| `agent/transports/codex_app_server.py` | PR | #50038 | — |
| `agent/transports/codex_app_server_session.py` | PR | #50038 | — |
| `agent/usage_pricing.py` | PR | #48024 | — |
| `batch_runner.py` | PR | #49644 | — |
| `cli.py` | PR | #49917 | — |
| `gateway/platforms/api_server.py` | PR | #48024 | — |
| `gateway/run.py` | PR | #49644 | #50146 |
| `gateway/status.py` | PR | #50047 | — |
| `hermes_cli/_parser.py` | PR | #49917 | — |
| `hermes_cli/auth.py` | PR | #49644 | — |
| `hermes_cli/banner.py` | PR | #50046 | — |
| `hermes_cli/codex_models.py` | PR | #50038 | — |
| `hermes_cli/commands.py` | PR | #49644 | #49917 |
| `hermes_cli/config.py` | PR | #49917 | #50046, #50073 |
| `hermes_cli/copilot_auth.py` | PR | #50064 | — |
| `hermes_cli/doctor.py` | PR | #50041 | — |
| `hermes_cli/gateway.py` | PR | #50047 | — |
| `hermes_cli/inventory.py` | PR | #50064 | — |
| `hermes_cli/kanban_db.py` | PR | #50056 | — |
| `hermes_cli/main.py` | PR | #49644 | #49917 |
| `hermes_cli/models.py` | PR | #49644 | — |
| `hermes_cli/models.py.bak.20260603100620` | DISCARD (bak) | — | — |
| `hermes_cli/plugins.py` | PR | #50054 | — |
| `hermes_cli/prompt_size.py` | PR | #48101 | — |
| `hermes_cli/runtime_provider.py` | PR | #49644 | — |
| `hermes_cli/send_cmd.py` | PR | #50048 | — |
| `hermes_cli/skills_hub.py` | PR | #50045 | — |
| `hermes_cli/source.py` | PR | #50032 | — |
| `hermes_cli/stable_update.py` | PR | #50046 | — |
| `hermes_cli/web_server.py` | PR | #50086 | — |
| `hermes_constants.py` | PR | #49644 | — |
| `hermes_state.py` | PR | #50056 | #50296 |
| `plugins/model-providers/copilot/__init__.py` | PR | #50064 | — |
| `plugins/platforms/discord/adapter.py` | PR | #50078 | — |
| `run_agent.py` | PR | #49644 | #50073, #50296 |
| `tests/agent/test_anthropic_adapter.py` | PR | #50064 | — |
| `tests/agent/test_autopilot_council_gate.py` | PR | #49917 | — |
| `tests/agent/test_autopilot_driver.py` | PR | #49917 | — |
| `tests/agent/test_autopilot_e2e_loop.py` | PR | #49917 | — |
| `tests/agent/test_auxiliary_client.py` | PR | #50064 | — |
| `tests/agent/test_auxiliary_main_first.py` | PR | #50064 | — |
| `tests/agent/test_copilot_claude_anthropic_routing.py` | PR | #50064 | — |
| `tests/agent/test_copilot_opus_context_fix_2026_06_04.py` | PR | #50664 | — |
| `tests/agent/test_model_metadata.py` | PR | #50078 | — |
| `tests/agent/test_p2_p3_oversized_handling.py` | PR | #50073 | — |
| `tests/agent/test_subdirectory_hints.py` | SUPERSEDED | #29433 | upstream PR ships same fix + test |
| `tests/agent/transports/test_codex_app_server_session.py` | PR | #50038 | — |
| `tests/cli/test_autopilot_kick.py` | PR | #49917 | — |
| `tests/cli/test_reasoning_command.py` | PR | #50078 | — |
| `tests/gateway/test_api_server.py` | PR | #48024 | — |
| `tests/hermes_cli/test_banner_git_state.py` | PR | #50046 | — |
| `tests/hermes_cli/test_bedrock_model_picker.py` | PR | #50066 | — |
| `tests/hermes_cli/test_copilot_auth.py` | PR | #50064 | — |
| `tests/hermes_cli/test_copilot_catalog_oauth_fallback.py` | PR | #50064 | — |
| `tests/hermes_cli/test_copilot_context.py` | PR | #50064 | — |
| `tests/hermes_cli/test_copilot_token_exchange.py` | PR | #50064 | — |
| `tests/hermes_cli/test_doctor.py` | PR | #50041 | — |
| `tests/hermes_cli/test_inventory.py` | PR | #50064 | — |
| `tests/hermes_cli/test_kanban_db.py` | PR | #50056 | #50056 |
| `tests/hermes_cli/test_model_switch_copilot_api_mode.py` | PR | #50064 | — |
| `tests/hermes_cli/test_model_validation.py` | PR | #50064 | — |
| `tests/hermes_cli/test_send_cmd.py` | PR | #50048 | — |
| `tests/hermes_cli/test_stable_update.py` | PR | #50046 | — |
| `tests/hermes_cli/test_update_check.py` | PR | #50046 | — |
| `tests/hermes_cli/test_web_server.py` | PR | #50066 | #50086 |
| `tests/probe_prelude_e2e.py` | PR | #50078 | — |
| `tests/run_agent/test_agent_guardrails.py` | PR | #50078 | — |
| `tests/run_agent/test_compression_feasibility.py` | PR | #50080 | — |
| `tests/run_agent/test_copilot_native_vision_headers.py` | PR | #50064 | — |
| `tests/run_agent/test_provider_attribution_headers.py` | PR | #50064 | — |
| `tests/run_agent/test_run_agent.py` | PR | #50078 | — |
| `tests/run_agent/test_run_agent_codex_responses.py` | PR | #50064 | — |
| `tests/test_anthropic_thinking_display.py` | PR | #48024 | — |
| `tests/test_auto_router_live.py` | PR | #50031 | — |
| `tests/test_context_engine_tool_wrap.py` | PR | #50080 | — |
| `tests/test_hermes_constants.py` | PR | #50078 | — |
| `tests/test_hermes_state.py` | PR | #50056 | — |
| `tests/test_hermes_state_wal_fallback.py` | PR | #50056 | — |
| `tests/test_system_prompt_prelude.py` | PR | #50078 | — |
| `tests/test_tui_gateway_server.py` | PR | #50078 | — |
| `tests/tools/test_file_read_guards.py` | PR | #50042 | — |
| `tests/tools/test_file_tools.py` | PR | #50042 | — |
| `tests/tools/test_hermes_source_accelerator.py` | PR | #50032 | — |
| `tests/tools/test_skills_guard.py` | PR | #50045 | — |
| `tests/tools/test_skills_hub.py` | PR | #50045 | — |
| `tests/tools/test_skills_tool.py` | PR | #50045 | — |
| `tests/tui_gateway/test_autopilot_command.py` | PR | #49917 | — |
| `tests/tui_gateway/test_notify_autodispatch.py` | PR | #49917 | — |
| `tools/delegate_tool.py` | PR | #50040 | — |
| `tools/file_tools.py` | PR | #50042 | — |
| `tools/hermes_source.py` | PR | #50032 | — |
| `tools/mcp_tool.py` | PR | #48069 | — |
| `tools/mcp_tool.py.bak.keepalivefix.20260610122428` | DISCARD (bak) | — | — |
| `tools/model_router.py` | PR | #50022 | — |
| `tools/project_source.py` | PR | #50032 | — |
| `tools/schema_sanitizer.py` | PR | #48057 | — |
| `tools/schema_sanitizer.py.bak.20260614_165117` | DISCARD (bak) | — | — |
| `tools/send_message_tool.py` | PR | #50048 | — |
| `tools/skills_guard.py` | PR | #50045 | — |
| `tools/skills_hub.py` | PR | #50045 | — |
| `tools/skills_tool.py` | PR | #50045 | — |
| `toolsets.py` | PR | #50032 | — |
| `tui_gateway/server.py` | PR | #49916 | #49917 |
| `ui-tui/src/__tests__/appChromeStatusRule.test.tsx` | PR | #50068 | — |
| `ui-tui/src/app/useInputHandlers.ts` | PR | #49915 | — |
| `ui-tui/src/components/appChrome.tsx` | PR | #50068 | — |
| `ui-tui/src/components/appLayout.tsx` | PR | #50068 | — |
| `ui-tui/src/types.ts` | PR | #50068 | — |

Totals (sum = 160 = delta): in-PR 139 + DISCARD 21 + ORPHAN 0.

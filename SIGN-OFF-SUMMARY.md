# Sign-off summary — file→PR map + DISCARD justifications

Independently corroborated live (see independent_corroboration.sh): 0 leaks / 41 PRs;
delta 165 = 138 in-PR + 25 DISCARD + 2 upstream + 0 orphans; 40 code PRs merge onto
v0.17.0 (40/40) AND origin/main (40/40) with 0 conflicts.

## In-PR files, grouped by owning PR (138 files across 40 PRs)

### #48024 — feat(reasoning): expose thinking/reasoning across API + cont (5 files)
  `agent/anthropic_adapter.py`, `agent/usage_pricing.py`, `gateway/platforms/api_server.py`, `tests/gateway/test_api_server.py`, `tests/test_anthropic_thinking_display.py`

### #48057 — fix(tools): drop empty-name tools before they reach the prov (1 files)
  `tools/schema_sanitizer.py`

### #48065 — fix(agent): unwrap pre-wrapped context-engine tool schemas (1 files)
  `agent/agent_init.py`

### #48069 — fix(tools): skip MCP keepalive during in-flight calls + fail (18 files)
  `agent/auxiliary_client.py`, `agent/chat_completion_helpers.py`, `cli.py`, `gateway/run.py`, `hermes_cli/auth.py`, `hermes_cli/config.py`, `hermes_cli/main.py`, `hermes_cli/web_server.py`, `hermes_constants.py`, `plugins/platforms/discord/adapter.py`, `run_agent.py`, `tests/hermes_cli/test_web_server.py`, `tests/test_hermes_constants.py`, `tests/test_tui_gateway_server.py`, `tools/delegate_tool.py`, `tools/mcp_tool.py`, `tools/send_message_tool.py`, `tui_gateway/server.py`

### #48101 — feat(agent): per-model system-prompt prelude (operation mode (3 files)
  `agent/system_prompt.py`, `agent/system_prompt_prelude.py`, `hermes_cli/prompt_size.py`

### #49184 — fix(copilot): route Claude on Copilot to /v1/messages instea (2 files)
  `agent/agent_runtime_helpers.py`, `agent/conversation_loop.py`

### #49449 — feat(models): correct under-reported Copilot/Codex per-model (1 files)
  `agent/models_dev.py`

### #49644 — feat(reasoning): accept "max" reasoning effort end-to-end wi (24 files)
  `agent/background_review.py`, `agent/gemini_cloudcode_adapter.py`, `agent/prompt_builder.py`, `agent/tool_executor.py`, `agent/transports/chat_completions.py`, `batch_runner.py`, `gateway/status.py`, `hermes_cli/banner.py`, `hermes_cli/commands.py`, `hermes_cli/doctor.py`, `hermes_cli/gateway.py`, `hermes_cli/kanban_db.py`, `hermes_cli/models.py`, `hermes_cli/plugins.py`, `hermes_cli/providers.py`, `hermes_cli/runtime_provider.py`, `hermes_state.py`, `tests/hermes_cli/test_doctor.py`, `tests/hermes_cli/test_kanban_db.py`, `tests/run_agent/test_provider_attribution_headers.py`, `tests/run_agent/test_run_agent.py`, `tests/tools/test_file_read_guards.py`, `tests/tools/test_file_tools.py`, `tools/file_tools.py`

### #49915 — fix(tui): Ctrl+C always interrupts a running turn, even in V (1 files)
  `ui-tui/src/app/useInputHandlers.ts`

### #49917 — feat(autopilot): engine-enforced goal-chasing mode (/autopil (10 files)
  `agent/autopilot/__init__.py`, `agent/autopilot/council_gate.py`, `agent/autopilot/driver.py`, `hermes_cli/_parser.py`, `tests/agent/test_autopilot_council_gate.py`, `tests/agent/test_autopilot_driver.py`, `tests/agent/test_autopilot_e2e_loop.py`, `tests/cli/test_autopilot_kick.py`, `tests/tui_gateway/test_autopilot_command.py`, `tests/tui_gateway/test_notify_autodispatch.py`

### #50021 — feat(tools): opt-in tool-timing sidecar (HERMES_TOOL_TRACE) (1 files)
  `agent/tool_trace_sidecar.py`

### #50022 — feat(tools): model_router proxy client tool (1 files)
  `tools/model_router.py`

### #50031 — feat(copilot): auto-mode router for the model:auto billing d (2 files)
  `agent/auto_router.py`, `tests/test_auto_router_live.py`

### #50032 — feat(tools): deterministic source accelerator (hermes_source (5 files)
  `hermes_cli/source.py`, `tests/tools/test_hermes_source_accelerator.py`, `tools/hermes_source.py`, `tools/project_source.py`, `toolsets.py`

### #50033 — feat(gemini): present authentic @google/gemini-cli identity  (2 files)
  `agent/gemini_native_adapter.py`, `agent/google_user_agent.py`

### #50038 — feat(codex): present authentic codex CLI version/identity (6 files)
  `agent/codex_version.py`, `agent/transports/codex.py`, `agent/transports/codex_app_server.py`, `agent/transports/codex_app_server_session.py`, `hermes_cli/codex_models.py`, `tests/agent/transports/test_codex_app_server_session.py`

### #50045 — feat(skills): hub path-repair, frontmatter-name resolution,  (7 files)
  `hermes_cli/skills_hub.py`, `tests/tools/test_skills_guard.py`, `tests/tools/test_skills_hub.py`, `tests/tools/test_skills_tool.py`, `tools/skills_guard.py`, `tools/skills_hub.py`, `tools/skills_tool.py`

### #50046 — feat(update): optional stable-release-tag update check (4 files)
  `hermes_cli/stable_update.py`, `tests/hermes_cli/test_banner_git_state.py`, `tests/hermes_cli/test_stable_update.py`, `tests/hermes_cli/test_update_check.py`

### #50048 — feat(send): [[plain]] directive + --plain flag for unformatt (2 files)
  `hermes_cli/send_cmd.py`, `tests/hermes_cli/test_send_cmd.py`

### #50053 — feat(context-engine): additive grounding hook points on the  (1 files)
  `agent/context_engine.py`

### #50056 — feat(state): HERMES_SQLITE_DRIVER selection + driver-agnosti (2 files)
  `tests/test_hermes_state.py`, `tests/test_hermes_state_wal_fallback.py`

### #50064 — feat(copilot): authentic @github/copilot CLI identity + Clau (18 files)
  `agent/copilot_acp_client.py`, `agent/model_metadata.py`, `hermes_cli/copilot_auth.py`, `hermes_cli/inventory.py`, `plugins/model-providers/copilot/__init__.py`, `tests/agent/test_anthropic_adapter.py`, `tests/agent/test_auxiliary_client.py`, `tests/agent/test_auxiliary_main_first.py`, `tests/agent/test_copilot_claude_anthropic_routing.py`, `tests/hermes_cli/test_copilot_auth.py`, `tests/hermes_cli/test_copilot_catalog_oauth_fallback.py`, `tests/hermes_cli/test_copilot_context.py`, `tests/hermes_cli/test_copilot_token_exchange.py`, `tests/hermes_cli/test_inventory.py`, `tests/hermes_cli/test_model_switch_copilot_api_mode.py`, `tests/hermes_cli/test_model_validation.py`, `tests/run_agent/test_copilot_native_vision_headers.py`, `tests/run_agent/test_run_agent_codex_responses.py`

### #50066 — test: bedrock EU-region fallback + sessions pagination total (1 files)
  `tests/hermes_cli/test_bedrock_model_picker.py`

### #50068 — feat(tui): autopilot + YOLO status-bar badges (frontend) (4 files)
  `ui-tui/src/__tests__/appChromeStatusRule.test.tsx`, `ui-tui/src/components/appChrome.tsx`, `ui-tui/src/components/appLayout.tsx`, `ui-tui/src/types.ts`

### #50073 — feat(compression): offload oversized single message to file- (1 files)
  `tests/agent/test_p2_p3_oversized_handling.py`

### #50078 — test: catch-up tests + discord 'max' effort description (5 files)
  `tests/agent/test_model_metadata.py`, `tests/cli/test_reasoning_command.py`, `tests/probe_prelude_e2e.py`, `tests/run_agent/test_agent_guardrails.py`, `tests/test_system_prompt_prelude.py`

### #50080 — test: context-engine unwrap + compression main-runtime field (2 files)
  `tests/run_agent/test_compression_feasibility.py`, `tests/test_context_engine_tool_wrap.py`

### #50457 — feat(copilot): opus-context unique files (slimmed from 100-f (2 files)
  `tests/agent/conftest.py`, `tests/agent/test_copilot_opus_context_fix_2026_06_04.py`

### #50555 — feat(provider): isolated agy-cli provider (WIP, draft — not  (6 files)
  `agent/agy_cli_client.py`, `plugins/model-providers/agy-cli/__init__.py`, `plugins/model-providers/agy-cli/plugin.yaml`, `tests/agent/test_agy_cli_client_v2.py`, `tests/agent/test_agy_cli_client_v3.py`, `tests/plugins/test_agy_cli_plugin_v2.py`

## DISCARD (25 files — non-contributable, never PR'd)
| file | reason |
|---|---|
| `.project-intel/BENCHMARKS.md` | generated project-intelligence index |
| `.project-intel/DATAFLOW_MAP.md` | generated project-intelligence index |
| `.project-intel/FEATURE_MAP.md` | generated project-intelligence index |
| `.project-intel/FLOW_MAP.md` | generated project-intelligence index |
| `.project-intel/FUNNEL_MAP.md` | generated project-intelligence index |
| `.project-intel/HEALTH.md` | generated project-intelligence index |
| `.project-intel/INDEX_STATE.md` | generated project-intelligence index |
| `.project-intel/PROJECT_INTELLIGENCE.md` | generated project-intelligence index |
| `.project-intel/QUERY_ROUTING.md` | generated project-intelligence index |
| `.project-intel/RESOURCE_INVENTORY.md` | generated project-intelligence index |
| `.project-intel/UI_FLOW_MAP.md` | generated project-intelligence index |
| `.project-intel/indexes/project-index.sqlite` | generated project-intelligence index |
| `agent/anthropic_adapter.py.bak.20260603100620` | editor/build backup snapshot |
| `agent/codex_responses_adapter.py.bak.20260603100620` | editor/build backup snapshot |
| `agent/context_engine.py.bak.20260607_231325` | editor/build backup snapshot |
| `agent/conversation_loop.py.bak.20260607_231325` | editor/build backup snapshot |
| `agent/model_metadata.py.bak.20260603100620` | editor/build backup snapshot |
| `agent/system_prompt.py.bak.20260614_131421` | editor/build backup snapshot |
| `hermes_cli/models.py.bak.20260603100620` | editor/build backup snapshot |
| `tools/mcp_tool.py.bak.keepalivefix.20260610122428` | editor/build backup snapshot |
| `tools/schema_sanitizer.py.bak.20260614_165117` | editor/build backup snapshot |
| `transcripts/C_opus_baseline.txt` | Fable-5 prelude eval-capture |
| `transcripts/C_opus_contradiction.txt` | Fable-5 prelude eval-capture |
| `transcripts/C_sonnet_baseline.txt` | Fable-5 prelude eval-capture |
| `transcripts/C_sonnet_contradiction.txt` | Fable-5 prelude eval-capture |

## Upstream-covered (2 files — by udatny's superset #29433)
- `agent/subdirectory_hints.py`
- `tests/agent/test_subdirectory_hints.py`
# Phase 5.8 Runtime Dependency Manifest

Base: `15e57836d2`.

This is the exhaustive per-consumer manifest captured by Phase 5.8.1. Responsibility
classes and migration rules are defined in
`PHASE5_8_RUNTIME_CONSUMER_BASELINE.md`. The destination column names the final
owner rather than an interim forwarding shim.

## Runtime dependency manifest

### Agent runtime

| Consumer | Upward dependency | Class | Phase 5.8 destination |
| --- | --- | --- | --- |
| `agent/agent_init.py` | `models_catalog_static.static_provider_model_ids` | Model identity/catalogue | Lower-domain model catalogue query exposed under `models/`; same owner used by CLI |
| `agent/agent_init.py` | `model_switch._check_hermes_model_warning` | Application/presentation | Agent/application warning helper; no domain authority and no CLI import |
| `agent/agent_init.py` | dynamic `models.copilot_default_headers` | Routing | Provider-specific transport/profile data below CLI, consumed by runtime routing/client construction |
| `agent/agent_runtime_helpers.py` | `models.opencode_provider_family` | Provider identity/registry | `providers.identity` / `providers.registry` provider-family query |
| `agent/agent_runtime_helpers.py` | `models.copilot_default_headers` | Routing | Provider-specific transport/profile data below CLI |
| `agent/auxiliary_client.py` | `model_selection_auxiliary.is_declared_vision_default`, `provider_vision_default`, `select_provider_auxiliary_fallback`, `select_provider_auxiliary_model`, `select_provider_vision_model` | Selection | `models.selection*`; configuration/availability facts supplied by the runtime caller |
| `agent/auxiliary_client.py` | `model_selection_auxiliary.provider_rejects_vision_input` | Metadata/capability | `models.metadata` capability query |
| `agent/auxiliary_client.py` | `models.get_nous_recommended_aux_model` | Selection | `models.selection*`; Nous account discovery remains outside pure selection |
| `agent/auxiliary_client.py` | `models_catalog_static.static_provider_model_ids` | Model identity/catalogue | Lower-domain model catalogue query |
| `agent/auxiliary_client.py` | `models.copilot_default_headers` | Routing | Provider-specific transport/profile data below CLI |
| `agent/chat_completion_helpers.py` | `models_catalog_static.static_provider_model_ids` | Model identity/catalogue | Lower-domain model catalogue query |
| `agent/client_lifecycle.py` | dynamic `models.copilot_default_headers` | Routing | Provider-specific transport/profile data below CLI |
| `agent/credits_tracker.py` | `models._is_model_free`; `models_pricing.peek_cached_pricing`, `get_pricing_for_provider`, `pricing_fetch_suppressed` | Metadata/capability | Lower-domain model pricing/metadata service; one cache owner, not an agent-local mirror |
| `agent/error_classifier.py` | `models_catalog_static.static_provider_model_ids` | Model identity/catalogue | Lower-domain model catalogue query |
| `agent/error_surface.py` | `models.provider_label` | Provider identity/registry | Provider registry/profile display metadata; presentation remains local |
| `agent/fallback_cooldown.py` | `models_catalog_static.static_provider_model_ids` | Model identity/catalogue | Lower-domain model catalogue query |
| `agent/fast_mode.py` | `models.resolve_fast_mode_overrides` | Metadata/capability | `models.metadata` fast-mode capability/override query |
| `agent/models_dev.py` | `models.opencode_provider_family` | Provider identity/registry | `providers.identity` / `providers.registry` |
| `agent/model_metadata.py` | `models.get_copilot_model_context` | Metadata/capability | `models.metadata.context` |
| `agent/opencode_affinity.py` | `models.opencode_provider_family` | Provider identity/registry | `providers.identity` / `providers.registry` |
| `agent/opencode_affinity.py` | `models.normalize_opencode_base_url`, `normalize_opencode_model_id` | Routing | `providers.routing` / `providers.model_normalizers` |
| `agent/reasoning_params.py` | `models.github_model_reasoning_efforts`, `clamp_github_reasoning_effort`, `models_local.lmstudio_model_reasoning_options`, `ollama_model_supports_thinking` | Metadata/capability | `models.metadata.reasoning` plus provider-specific capability sources |
| `agent/turn_failure_copy.py` | `models.provider_label` | Provider identity/registry | Provider registry/profile display metadata |
| `agent/turn_recovery.py` | `models_catalog_static.static_provider_model_ids` | Model identity/catalogue | Lower-domain model catalogue query |
| `agent/transports/codex.py` | `models.opencode_provider_family` | Provider identity/registry | `providers.identity` / `providers.registry` |

### Gateway runtime

| Consumer | Upward dependency | Class | Phase 5.8 destination |
| --- | --- | --- | --- |
| `gateway/platforms/api_server.py` | `model_switch.resolve_effective_model` | Selection | Gateway supplies config/session facts to `models.selection*`; application precedence remains explicit at the caller |
| `gateway/platforms/api_server.py` | `model_selection_defaults.select_provider_default`, `selected_model_id` | Selection | `models.selection_defaults` / canonical selection result |
| `gateway/run_agent_cache.py` | `models.normalize_opencode_base_url`, `opencode_provider_family` | Routing | `providers.routing` / `providers.model_normalizers`; provider-family identity comes from `providers.identity` |
| `gateway/run_config_loaders.py` | `model_switch.resolve_effective_model` | Selection | Gateway-owned fact precedence feeding `models.selection*`; no CLI coordinator |
| `gateway/run_turn.py` | `models_catalog_static.static_provider_model_ids` | Model identity/catalogue | Lower-domain model catalogue query |
| `gateway/run_turn_prepare.py` | `model_selection_defaults.select_provider_default`, `selected_model_id` | Selection | `models.selection_defaults` |
| `gateway/run_turn_prepare.py` | `models.resolve_fast_mode_overrides` | Metadata/capability | `models.metadata` fast-mode capability query |
| `gateway/run_watchers.py` | `model_catalog.refresh_catalogs`, `refresh_interval_seconds` | Model identity/catalogue | Lower-domain catalogue lifecycle service; gateway remains scheduler only |
| `gateway/session_local_route.py` | `model_switch.resolve_startup_model_route` | Selection | Split into `models.selection*` followed by `providers.routing`; gateway owns session application |
| `gateway/session_mutation_model.py` | `model_switch.switch_model` | Application/presentation | Gateway-owned model mutation coordinator consuming identity → selection → route → runtime apply |
| `gateway/slash_commands_model.py` | `model_switch.persist_model_selection`, `resolve_persist_behavior` | Credential/persistence | Gateway/application config persistence; no provider/model truth |
| `gateway/slash_commands_model.py` | `model_switch.switch_model` | Application/presentation | Gateway-owned mutation coordinator consuming canonical domains |
| `gateway/slash_commands_model.py` | `model_switch.format_model_for_display` | Application/presentation | Gateway presentation helper |
| `gateway/slash_commands_model.py` | `model_switch.resolve_display_context_length_async` | Metadata/capability | `models.metadata.context` plus runtime discovery facts |
| `gateway/slash_commands_model.py` | `model_switch_providers.list_picker_providers` | Application/presentation | Gateway picker projection over provider registry/catalogue |
| `gateway/slash_commands_model.py` | `model_switch.list_authenticated_providers` | Credential/persistence | Auth/credential availability query; provider identity remains in registry |
| `gateway/slash_commands_model.py` | `model_switch.parse_model_switch_args` | Application/presentation | Gateway command/request parser producing canonical selection input |
| `gateway/slash_commands_model.py` | `model_selection_guards.combined_selection_warning`, `selection_context_for_agent` | Application/presentation | Gateway warning/presentation layer after canonical selection |
| `gateway/slash_commands_model.py` | `models.clear_provider_models_cache` | Model identity/catalogue | Lower-domain catalogue cache owner |
| `gateway/slash_commands_model.py` | `models.model_supports_fast_mode` | Metadata/capability | `models.metadata` |

### TUI runtime

| Consumer | Upward dependency | Class | Phase 5.8 destination |
| --- | --- | --- | --- |
| `tui_gateway/agent_factory.py` | `model_selection_defaults.select_provider_default`, `selected_model_id` | Selection | `models.selection_defaults` |
| `tui_gateway/agent_factory.py` | `model_switch.resolve_startup_model_route` | Selection | `models.selection*` followed by `providers.routing`; TUI owns application |
| `tui_gateway/agent_factory.py` | `models.detect_static_provider_for_model` | Provider identity/registry | Provider/model identity query below CLI; catalogue facts supplied explicitly |
| `tui_gateway/agent_factory.py` | `model_switch.model_derived_api_mode` | Routing | `providers.routing` API-mode interpretation |
| `tui_gateway/agent_factory.py` | `models.normalize_opencode_base_url` | Routing | `providers.routing` / `providers.model_normalizers` |
| `tui_gateway/entry.py` | `model_switch_providers.prewarm_picker_cache_async` | Application/presentation | TUI/app-owned prewarm over lower-domain catalogue service |
| `tui_gateway/methods_config.py` | `models.list_available_providers` | Provider identity/registry | `providers.registry` projection |
| `tui_gateway/methods_config_set.py` | `model_switch.parse_model_switch_args` | Application/presentation | TUI request parser producing canonical selection input |
| `tui_gateway/methods_config_set.py` | `models.resolve_fast_mode_overrides` | Metadata/capability | `models.metadata` |
| `tui_gateway/methods_profiles.py` | dynamic `model_selection_guards.combined_selection_warning` | Application/presentation | TUI warning/presentation layer |
| `tui_gateway/methods_session_model_guard.py` | `models_validate.static_model_provider_conflict` | Model identity/catalogue | Canonical provider/model membership validation over lower-domain catalogue |
| `tui_gateway/model_switch.py` | `model_switch.MODEL_SWITCH_ERR_ONCE_WITH_GLOBAL`, `MODEL_SWITCH_ERROR_TEXT`, `parse_model_switch_args` | Application/presentation | TUI-local command/error presentation; shared parsing may be a non-CLI application helper |
| `tui_gateway/model_switch.py` | `model_switch.resolve_persist_behavior`, `persist_model_selection` | Credential/persistence | TUI/application persistence |
| `tui_gateway/model_switch.py` | `model_switch.switch_model` | Application/presentation | TUI-owned apply coordinator consuming canonical identity/selection/routing |
| `tui_gateway/model_switch.py` | `model_selection_guards.combined_selection_warning`, `selection_context_for_agent` | Application/presentation | TUI warning/presentation layer |
| `tui_gateway/server.py` | `models.resolve_fast_mode_overrides` | Metadata/capability | `models.metadata` |

### ACP runtime

| Consumer | Upward dependency | Class | Phase 5.8 destination |
| --- | --- | --- | --- |
| `acp_adapter/model_catalog.py` | `model_switch._declared_model_ids`, `_entry_models_discovered`, `_models_config_is_allowlist` | Model identity/catalogue | Public lower-domain catalogue/config projection; private CLI helpers are deleted from the ACP path |
| `acp_adapter/model_catalog.py` | `model_switch_providers._NativePickerModelList`, `_fetch_picker_live_models`, `_discover_flag` | Model identity/catalogue | Public lower-domain live catalogue/discovery query |
| `acp_adapter/model_catalog.py` | `models_local.should_use_ollama_native_catalog` | Model identity/catalogue | Lower-domain local-provider catalogue policy |
| `acp_adapter/model_catalog.py` | `models.provider_label` | Provider identity/registry | Provider registry/profile display metadata |
| `acp_adapter/server.py` | `model_switch.switch_model` | Application/presentation | ACP-owned session mutation consuming canonical identity → selection → route |

### Provider and platform plugins

| Consumer | Upward dependency | Class | Phase 5.8 destination |
| --- | --- | --- | --- |
| `plugins/model-providers/anthropic/__init__.py` | `models._ANTHROPIC_MODELS_MAX_PAGES`, `_anthropic_models_url`, `_anthropic_next_cursor` | Model identity/catalogue | Anthropic provider plugin owns transport/pagination mechanics; normalized catalogue facts feed the lower-domain catalogue |
| `plugins/model-providers/copilot/__init__.py` | `models.clamp_github_reasoning_effort`, `github_model_reasoning_efforts` | Metadata/capability | `models.metadata.reasoning` with provider plugin as capability source |
| `plugins/model-providers/deepinfra/__init__.py` | `models._fetch_deepinfra_models_by_tag` | Model identity/catalogue | DeepInfra provider plugin owns provider-specific fetch; normalized results feed the one catalogue owner |
| `plugins/model-providers/nous/__init__.py` | `models.get_nous_recommended_aux_model` | Selection | `models.selection*`; account/Portal discovery remains provider/runtime input |
| `plugins/model-providers/openrouter/__init__.py` | `models.clamp_reasoning_effort_to_supported` | Metadata/capability | `models.metadata.reasoning` |
| `plugins/image_gen/deepinfra/__init__.py` | `models._fetch_deepinfra_models_by_tag` | Model identity/catalogue | Shared DeepInfra provider catalogue service, not a second media catalogue |
| `plugins/image_gen/deepinfra/__init__.py` | `models.deepinfra_base_url` | Routing | DeepInfra provider/profile endpoint metadata below CLI |
| `plugins/video_gen/deepinfra/__init__.py` | `models._fetch_deepinfra_models_by_tag` | Model identity/catalogue | Shared DeepInfra provider catalogue service |
| `plugins/platforms/discord/adapter.py` | `model_selection_guards.combined_selection_warning` | Application/presentation | Discord presentation layer; consumes selection result/facts without CLI dependency |
| `plugins/platforms/feishu/feishu_comment.py` | `model_selection_defaults.select_provider_default`, `selected_model_id` | Selection | `models.selection_defaults` |
| `plugins/platforms/telegram/adapter.py` | `models_catalog_static.group_providers`, `PROVIDER_GROUPS` | Application/presentation | Telegram/picker presentation projection fed by canonical provider/catalogue data |
| `plugins/platforms/telegram/adapter.py` | `model_selection_guards.combined_selection_warning` | Application/presentation | Telegram presentation layer |

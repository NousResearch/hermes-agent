# Plugin compatibility manifest (temporary)

The September 2026 decomposition (PR #102117) split the large modules of Hermes Agent into focused
files. **Internal import paths are not a stable API**, and after that PR the names below are no
longer defined where they used to be. To give external plugin authors time to update, every name
is still importable from its OLD module through a `PLUGIN-COMPAT` block appended to that module.

**This layer is temporary and removed on 2026-09-14.** It was added as a single commit and is removed
by reverting that commit. Update your plugin to import from the `new location` column now.
Nothing inside this repository is allowed to use these pointers (`scripts/check_compat_pointers.py`
fails CI if it does).

**What happens to an affected plugin.**

| when | CLI banner / `hermes doctor` / `hermes update` | Desktop | the plugin |
|---|---|---|---|
| before 2026-09-14 | yellow notice naming the plugin, the date, and `hermes plugins compat` | one-time modal (per set of affected plugins) | loads; each old-path resolution emits `HermesPluginCompatWarning` once |
| from 2026-09-14 | red notice: plugin **DISABLED** | one-time modal | **not loaded**; `hermes plugins list` shows the reason |
| after the revert lands | same | same | not loaded (the old paths no longer exist) |

Escape hatch for users who cannot wait on an author: `plugins.allow_deprecated_imports: true` in
`config.yaml` keeps affected plugins loading after the date, until the revert actually removes the paths.

**For plugin authors:** run `hermes plugins compat <path-to-your-plugin>` — it prints every `file:line`,
old path → new path, and exits 1 while anything remains. Import from the `new location` column.

**You will see a warning.** The first time a process resolves a name through one of these blocks, Hermes emits a
`HermesPluginCompatWarning` (a `FutureWarning`) naming the old path, the new path, and the removal target — once per
name per process. Fix the import and it goes away. To silence during migration:
`python -W ignore::hermes_cli.plugin_compat.HermesPluginCompatWarning` or `warnings.filterwarnings("ignore", category=HermesPluginCompatWarning)`.

**Scope.** Only PUBLIC names (no leading underscore) that were defined or imported at module top level
before the decomposition are covered. Private names (`_foo`, `_TG_NAME_LIMIT`, `_clamp_telegram_names`,
...) were never part of any surface and are NOT restored; a plugin that patched or imported one must move
to the public equivalent or the new module. Test monkeypatch seams are likewise not preserved.

| kind | count | meaning |
|---|---|---|
| moved | 0 | name now defined in `new location`; re-exported from the old module |
| moved-lazy | 1148 | same, resolved lazily via `__getattr__` to avoid an import cycle |
| import | 592 | a third-party/stdlib name the old module used to expose; original import restored |
| restored-def | 290 | public name that was deleted as unused; its pre-decomposition definition is restored verbatim |
| restored-helper | 41 | private helper restored only because a restored-def above depends on it |
| restored-import | 17 | import re-added only because a restored-def above depends on it |
| module-stub | 3 | whole module deleted; stub re-exports from its replacement |
| unrestorable | 34 | not restorable (e.g. leaked loop variables); listed for completeness |

## Names by old module


### `acp_adapter.auth`

| name | kind | new location |
|---|---|---|
| `has_provider` | restored-def | `(deleted; BASE body restored)` |

### `acp_adapter.edit_approval`

| name | kind | new location |
|---|---|---|
| `FutureTimeout` | import | `concurrent.futures` |
| `clear_edit_approval_requester` | restored-def | `(deleted; BASE body restored)` |
| `get_edit_approval_requester` | restored-def | `(deleted; BASE body restored)` |

### `acp_adapter.events`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |

### `acp_adapter.server`

| name | kind | new location |
|---|---|---|
| `ACP_MAX_MODELS_PER_PROVIDER` | moved-lazy | `acp_adapter.model_catalog` |
| `AgentThoughtChunk` | import | `acp.schema` |
| `AudioContentBlock` | import | `acp.schema` |
| `AvailableCommand` | import | `acp.schema` |
| `AvailableCommandsUpdate` | import | `acp.schema` |
| `BlobResourceContents` | import | `acp.schema` |
| `EmbeddedResourceContentBlock` | import | `acp.schema` |
| `ImageContentBlock` | import | `acp.schema` |
| `Path` | import | `pathlib` |
| `ResourceContentBlock` | import | `acp.schema` |
| `TextResourceContents` | import | `acp.schema` |
| `UnstructuredCommandInput` | import | `acp.schema` |
| `base64` | import | `base64` |
| `json` | import | `json` |
| `unquote` | import | `urllib.parse` |
| `urlparse` | import | `urllib.parse` |

### `acp_adapter.session`

| name | kind | new location |
|---|---|---|
| `Lock` | import | `threading` |

### `agent`

| name | kind | new location |
|---|---|---|
| `message_sanitization` | unrestorable | `no top-level definition on BASE` |

### `agent.agent_init`

| name | kind | new location |
|---|---|---|
| `ToolGuardrailDecision` | moved-lazy | `agent.tool_guardrails` |

### `agent.agent_runtime_helpers`

| name | kind | new location |
|---|---|---|
| `agent_runtime_owns_post_tool_hook` | restored-def | `(deleted; BASE body restored)` |
| `intent_ack_continuation_enabled` | restored-def | `(deleted; BASE body restored)` |

### `agent.anthropic_adapter`

| name | kind | new location |
|---|---|---|
| `CredentialPersistError` | moved-lazy | `agent.anthropic_credentials` |
| `Path` | import | `pathlib` |
| `Tuple` | import | `typing` |
| `base_url_host_matches` | moved-lazy | `utils` |
| `base_url_hostname` | moved-lazy | `utils` |
| `claude_code_credentials_path` | moved-lazy | `agent.anthropic_credentials` |
| `copy` | import | `copy` |
| `get_hermes_home` | moved-lazy | `hermes_constants` |
| `is_claude_code_token_valid` | moved-lazy | `agent.anthropic_credentials` |
| `is_rotation_consumed_uncommitted` | moved-lazy | `agent.anthropic_credentials` |
| `json` | import | `json` |
| `mark_rotation_consumed_uncommitted` | moved-lazy | `agent.anthropic_credentials` |
| `os` | import | `os` |
| `platform` | import | `platform` |
| `read_claude_code_credentials` | moved-lazy | `agent.anthropic_credentials` |
| `read_hermes_oauth_credentials` | moved-lazy | `agent.anthropic_credentials` |
| `refresh_anthropic_oauth_pure` | moved-lazy | `agent.anthropic_credentials` |
| `resolve_anthropic_token` | moved-lazy | `agent.anthropic_credentials` |
| `run_hermes_oauth_login_pure` | moved-lazy | `agent.anthropic_credentials` |
| `run_oauth_setup_token` | moved-lazy | `agent.anthropic_credentials` |
| `secrets` | import | `secrets` |
| `stat` | import | `stat` |
| `urlparse` | import | `urllib.parse` |

### `agent.aux_accounting`

| name | kind | new location |
|---|---|---|
| `get_accounting_context` | restored-def | `(deleted; BASE body restored)` |

### `agent.auxiliary_client`

| name | kind | new location |
|---|---|---|
| `NOUS_EXTRA_BODY` | restored-def | `(deleted; BASE body restored)` |
| `Path` | import | `pathlib` |
| `copy` | import | `copy` |
| `get_async_text_auxiliary_client` | restored-def | `(deleted; BASE body restored)` |

### `agent.backend_identity`

| name | kind | new location |
|---|---|---|
| `_REASON_SCOPES` | restored-helper | `(deleted; restored as a dependency of classify_failure_scope)` |
| `classify_failure_scope` | restored-def | `(deleted; BASE body restored)` |

### `agent.background_review`

| name | kind | new location |
|---|---|---|
| `Path` | import | `pathlib` |
| `is_background_review_enabled` | restored-def | `(deleted; BASE body restored)` |

### `agent.bedrock_adapter`

| name | kind | new location |
|---|---|---|
| `CONTEXT_OVERFLOW_PATTERNS` | restored-def | `(deleted; BASE body restored)` |
| `OVERLOAD_PATTERNS` | restored-def | `(deleted; BASE body restored)` |
| `THROTTLE_PATTERNS` | restored-def | `(deleted; BASE body restored)` |
| `call_converse_stream` | restored-def | `(deleted; BASE body restored)` |
| `classify_bedrock_error` | restored-def | `(deleted; BASE body restored)` |
| `is_context_overflow_error` | restored-helper | `(deleted; restored as a dependency of classify_bedrock_error)` |
| `is_context_overflow_error` | restored-def | `(deleted; BASE body restored)` |

### `agent.bounded_response`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |
| `read_error_body_or_default` | restored-def | `(deleted; BASE body restored)` |

### `agent.browser_provider`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |
| `Optional` | import | `typing` |

### `agent.browser_registry`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `List` | import | `typing` |
| `hermes_home_key` | moved-lazy | `hermes_constants` |
| `threading` | import | `threading` |

### `agent.codex_runtime`

| name | kind | new location |
|---|---|---|
| `run_codex_create_stream_fallback` | restored-def | `(deleted; BASE body restored)` |

### `agent.coding_context`

| name | kind | new location |
|---|---|---|
| `_PROFILES` | restored-helper | `(deleted; restored as a dependency of get_profile)` |
| `coding_system_blocks` | restored-def | `(deleted; BASE body restored)` |
| `get_profile` | restored-def | `(deleted; BASE body restored)` |

### `agent.context_compressor`

| name | kind | new location |
|---|---|---|
| `tool_result_id_variants` | moved-lazy | `agent.message_sanitization` |

### `agent.conversation_compression`

| name | kind | new location |
|---|---|---|
| `CompressionExecutorSaturatedError` | restored-def | `(deleted; BASE body restored)` |

### `agent.conversation_loop`

| name | kind | new location |
|---|---|---|
| `COMPRESSION_RETRY_CONTEXT_REDUCED_STATUS_TEMPLATE` | moved-lazy | `agent.conversation_compression` |
| `COMPRESSION_RETRY_MESSAGES_STATUS_TEMPLATE` | moved-lazy | `agent.conversation_compression` |
| `COMPRESSION_RETRY_TOKENS_STATUS_TEMPLATE` | moved-lazy | `agent.conversation_compression` |
| `COMPRESSION_RETRY_TOO_LARGE_STATUS_TEMPLATE` | moved-lazy | `agent.conversation_compression` |
| `FailoverReason` | moved-lazy | `agent.error_classifier` |
| `KawaiiSpinner` | moved-lazy | `agent.display` |
| `PARTIAL_STREAM_STUB_ID` | moved-lazy | `hermes_constants` |
| `PRE_API_COMPRESSION_STATUS_TEMPLATE` | moved-lazy | `agent.conversation_compression` |
| `adaptive_rate_limit_backoff` | moved-lazy | `agent.retry_utils` |
| `anchored_context_tokens` | moved-lazy | `agent.model_metadata` |
| `automatic_compaction_status_message` | moved-lazy | `agent.context_engine` |
| `capture_usage_anchor` | moved-lazy | `agent.model_metadata` |
| `classify_api_error` | moved-lazy | `agent.error_classifier` |
| `close_interrupted_tool_sequence` | moved-lazy | `agent.message_sanitization` |
| `coalesce_tool_call_id` | moved-lazy | `agent.message_sanitization` |
| `compose_user_api_content` | moved-lazy | `agent.turn_context` |
| `compression_blocked_transiently` | moved-lazy | `agent.conversation_compression` |
| `compression_skipped_due_to_lock` | moved-lazy | `agent.conversation_compression` |
| `context_compression_timed_out` | moved-lazy | `agent.conversation_compression` |
| `conversation_history_after_compression` | moved-lazy | `agent.conversation_compression` |
| `env_var_enabled` | moved-lazy | `utils` |
| `estimate_messages_tokens_rough` | moved-lazy | `agent.model_metadata` |
| `estimate_request_tokens_rough` | moved-lazy | `agent.model_metadata` |
| `estimate_usage_cost` | moved-lazy | `agent.usage_pricing` |
| `get_context_length_from_provider_error` | moved-lazy | `agent.model_metadata` |
| `has_incomplete_scratchpad` | moved-lazy | `agent.trajectory` |
| `is_output_cap_error` | moved-lazy | `agent.model_metadata` |
| `is_repetition_dominated` | moved-lazy | `agent.repetition_guard` |
| `is_zai_coding_overload_error` | moved-lazy | `agent.retry_utils` |
| `jittered_backoff` | moved-lazy | `agent.retry_utils` |
| `normalize_usage` | moved-lazy | `agent.usage_pricing` |
| `os` | import | `os` |
| `parse_available_output_tokens_from_error` | moved-lazy | `agent.model_metadata` |
| `random` | import | `random` |
| `reanchor_current_turn_user_idx` | moved-lazy | `agent.turn_context` |
| `save_context_length` | moved-lazy | `agent.model_metadata` |
| `serialized_messages_bytes` | moved-lazy | `agent.message_sanitization` |
| `splice_provider_projection` | moved-lazy | `agent.provider_projection` |
| `ssl` | import | `ssl` |
| `sys` | import | `sys` |
| `zai_coding_overload_retry_ceiling` | moved-lazy | `agent.retry_utils` |

### `agent.credential_sources`

| name | kind | new location |
|---|---|---|
| `register` | restored-def | `(deleted; BASE body restored)` |

### `agent.display`

| name | kind | new location |
|---|---|---|
| `get_friendly_tool_labels` | restored-def | `(deleted; BASE body restored)` |

### `agent.error_surface`

| name | kind | new location |
|---|---|---|
| `LAYER_RUNTIME` | restored-def | `(deleted; BASE body restored)` |

### `agent.estop`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `agent.file_safety`

| name | kind | new location |
|---|---|---|
| `PROFILE_SCOPED_AREAS` | restored-def | `(deleted; BASE body restored)` |
| `classify_cross_profile_target` | restored-def | `(deleted; BASE body restored)` |
| `get_cross_profile_warning` | restored-def | `(deleted; BASE body restored)` |

### `agent.image_gen_provider`

| name | kind | new location |
|---|---|---|
| `base64` | import | `base64` |
| `datetime` | import | `datetime` |
| `uuid` | import | `uuid` |

### `agent.image_gen_registry`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `List` | import | `typing` |
| `hermes_home_key` | moved-lazy | `hermes_constants` |
| `threading` | import | `threading` |

### `agent.learning_graph_render`

| name | kind | new location |
|---|---|---|
| `Grid` | restored-def | `(deleted; BASE body restored)` |
| `Run` | restored-def | `(deleted; BASE body restored)` |

### `agent.lsp.eventlog`

| name | kind | new location |
|---|---|---|
| `_announce_once` | restored-helper | `(deleted; restored as a dependency of log_no_server_configured)` |
| `_announced_no_server` | restored-helper | `(deleted; restored as a dependency of log_no_server_configured)` |
| `log_no_server_configured` | restored-def | `(deleted; BASE body restored)` |

### `agent.lsp.protocol`

| name | kind | new location |
|---|---|---|
| `ERROR_REQUEST_CANCELLED` | restored-def | `(deleted; BASE body restored)` |

### `agent.monitoring.cron_health`

| name | kind | new location |
|---|---|---|
| `GatewayHealthSnapshot` | moved-lazy | `agent.monitoring.gateway_health` |
| `hashlib` | import | `hashlib` |

### `agent.monitoring.emitter`

| name | kind | new location |
|---|---|---|
| `TelemetryEmitter` | restored-def | `(deleted; BASE body restored)` |

### `agent.monitoring.gateway_health`

| name | kind | new location |
|---|---|---|
| `redact_gateway_message` | restored-def | `(deleted; BASE body restored)` |

### `agent.monitoring.gateway_health_export`

| name | kind | new location |
|---|---|---|
| `re` | import | `re` |

### `agent.nous_rate_guard`

| name | kind | new location |
|---|---|---|
| `atomic_replace` | moved-lazy | `utils` |
| `tempfile` | import | `tempfile` |

### `agent.outbound_webhooks`

| name | kind | new location |
|---|---|---|
| `Path` | import | `pathlib` |
| `datetime` | import | `datetime` |
| `timezone` | import | `datetime` |

### `agent.pet.generate.atlas`

| name | kind | new location |
|---|---|---|
| `COLUMNS` | restored-def | `(deleted; BASE body restored)` |
| `FRAME_COUNTS` | restored-def | `(deleted; BASE body restored)` |
| `ROWS` | restored-def | `(deleted; BASE body restored)` |
| `atlas_to_webp_bytes` | restored-def | `(deleted; BASE body restored)` |
| `io` | restored-import | `io` |
| `io` | import | `io` |

### `agent.prompt_builder`

| name | kind | new location |
|---|---|---|
| `List` | import | `typing` |
| `org_id_of_path` | moved-lazy | `agent.skill_utils` |

### `agent.prompt_cache_boundary`

| name | kind | new location |
|---|---|---|
| `clear_stable_prefixes` | restored-def | `(deleted; BASE body restored)` |

### `agent.proxy_sources.iron_proxy`

| name | kind | new location |
|---|---|---|
| `stat` | import | `stat` |

### `agent.reasoning_effort`

| name | kind | new location |
|---|---|---|
| `CODEX_RESPONSES_EFFORTS` | restored-def | `(deleted; BASE body restored)` |

### `agent.relay_llm`

| name | kind | new location |
|---|---|---|
| `dataclass` | import | `dataclasses` |

### `agent.relay_runtime`

| name | kind | new location |
|---|---|---|
| `auto` | import | `enum` |
| `emit_mark` | restored-def | `(deleted; BASE body restored)` |
| `ensure_session` | restored-def | `(deleted; BASE body restored)` |
| `get_host` | restored-def | `(deleted; BASE body restored)` |
| `get_session_handle` | restored-def | `(deleted; BASE body restored)` |
| `run_in_session` | restored-def | `(deleted; BASE body restored)` |
| `run_in_session_async` | restored-def | `(deleted; BASE body restored)` |

### `agent.relay_tools`

| name | kind | new location |
|---|---|---|
| `asyncio` | import | `asyncio` |
| `inspect` | import | `inspect` |

### `agent.review_idle_queue`

| name | kind | new location |
|---|---|---|
| `List` | import | `typing` |

### `agent.secret_sources._cache`

| name | kind | new location |
|---|---|---|
| `FetchResult` | moved-lazy | `agent.secret_sources.base` |
| `is_valid_env_name` | moved-lazy | `agent.secret_sources.base` |

### `agent.secret_sources.bitwarden`

| name | kind | new location |
|---|---|---|
| `DiskCache` | moved-lazy | `agent.secret_sources._cache` |
| `apply_bitwarden_secrets` | restored-def | `(deleted; BASE body restored)` |
| `stat` | import | `stat` |

### `agent.secret_sources.command`

| name | kind | new location |
|---|---|---|
| `apply_command_secrets` | restored-def | `(deleted; BASE body restored)` |
| `get_command_secret` | restored-def | `(deleted; BASE body restored)` |
| `get_source_environment` | moved-lazy | `agent.secret_sources.base` |
| `list_command_secrets` | restored-def | `(deleted; BASE body restored)` |
| `parse_secret_output` | restored-helper | `(deleted; restored as a dependency of get_command_secret)` |
| `parse_secret_output` | restored-def | `(deleted; BASE body restored)` |

### `agent.secret_sources.onepassword`

| name | kind | new location |
|---|---|---|
| `DiskCache` | moved-lazy | `agent.secret_sources._cache` |
| `hashlib` | import | `hashlib` |

### `agent.shell_hooks`

| name | kind | new location |
|---|---|---|
| `shlex` | import | `shlex` |

### `agent.skill_bundles`

| name | kind | new location |
|---|---|---|
| `re` | import | `re` |

### `agent.skill_utils`

| name | kind | new location |
|---|---|---|
| `get_scan_ordered_skills_dirs` | restored-def | `(deleted; BASE body restored)` |

### `agent.ssl_guard`

| name | kind | new location |
|---|---|---|
| `verify_ca_bundle_with_fallback` | restored-def | `(deleted; BASE body restored)` |

### `agent.system_prompt`

| name | kind | new location |
|---|---|---|
| `OPENAI_MODEL_EXECUTION_GUIDANCE` | moved-lazy | `agent.prompt_builder` |

### `agent.terminal_env_registry`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `hermes_home_key` | moved-lazy | `hermes_constants` |
| `threading` | import | `threading` |

### `agent.transcript_repair`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |

### `agent.transcription_provider`

| name | kind | new location |
|---|---|---|
| `List` | import | `typing` |
| `logger` | moved-lazy | `agent.i18n` |
| `logging` | import | `logging` |

### `agent.transcription_registry`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `List` | import | `typing` |
| `Optional` | import | `typing` |
| `hermes_home_key` | moved-lazy | `hermes_constants` |
| `threading` | import | `threading` |

### `agent.transports.chat_completions`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |

### `agent.transports.codex`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `List` | import | `typing` |
| `Tuple` | import | `typing` |

### `agent.transports.codex_app_server`

| name | kind | new location |
|---|---|---|
| `field` | import | `dataclasses` |
| `time` | import | `time` |

### `agent.tts_registry`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `List` | import | `typing` |
| `Optional` | import | `typing` |
| `hermes_home_key` | moved-lazy | `hermes_constants` |
| `threading` | import | `threading` |

### `agent.turn_context`

| name | kind | new location |
|---|---|---|
| `IDLE_COMPACTION_STATUS_TEMPLATE` | moved-lazy | `agent.conversation_compression` |
| `PREFLIGHT_COMPRESSION_STATUS_TEMPLATE` | moved-lazy | `agent.conversation_compression` |
| `automatic_compaction_status_message` | moved-lazy | `agent.context_engine` |
| `compression_skipped_due_to_lock` | moved-lazy | `agent.conversation_compression` |
| `conversation_history_after_compression` | moved-lazy | `agent.conversation_compression` |

### `agent.turn_retry_state`

| name | kind | new location |
|---|---|---|
| `fields` | import | `dataclasses` |

### `agent.usage_pricing`

| name | kind | new location |
|---|---|---|
| `DEFAULT_PRICING` | restored-def | `(deleted; BASE body restored)` |

### `agent.video_gen_provider`

| name | kind | new location |
|---|---|---|
| `base64` | import | `base64` |
| `datetime` | import | `datetime` |
| `uuid` | import | `uuid` |

### `agent.video_gen_registry`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `List` | import | `typing` |
| `hermes_home_key` | moved-lazy | `hermes_constants` |
| `threading` | import | `threading` |

### `agent.web_search_provider`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |

### `agent.web_search_registry`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `List` | import | `typing` |
| `hermes_home_key` | moved-lazy | `hermes_constants` |
| `threading` | import | `threading` |

### `cli`

| name | kind | new location |
|---|---|---|
| `AIAgent` | restored-def | `(deleted; BASE body restored)` |
| `CanonicalUsage` | restored-def | `(deleted; BASE body restored)` |
| `CompletionsMenu` | import | `prompt_toolkit.layout.menus` |
| `Condition` | import | `prompt_toolkit.filters` |
| `ConditionalContainer` | import | `prompt_toolkit.layout` |
| `ConditionalProcessor` | import | `prompt_toolkit.layout.processors` |
| `DEFAULT_BROWSER_CDP_URL` | moved-lazy | `hermes_cli.browser_connect` |
| `Dimension` | import | `prompt_toolkit.layout.dimension` |
| `FileHistory` | import | `prompt_toolkit.history` |
| `FormattedTextControl` | import | `prompt_toolkit.layout` |
| `HERMES_AGENT_LOGO` | moved-lazy | `hermes_cli.banner` |
| `HERMES_CADUCEUS` | moved-lazy | `hermes_cli.banner` |
| `HSplit` | import | `prompt_toolkit.layout` |
| `KeyBindings` | import | `prompt_toolkit.key_binding` |
| `Layout` | import | `prompt_toolkit.layout` |
| `PTStyle` | import | `prompt_toolkit.styles` |
| `Panel` | import | `rich.panel` |
| `PasswordProcessor` | import | `prompt_toolkit.layout.processors` |
| `Processor` | import | `prompt_toolkit.layout.processors` |
| `SlashCommandAutoSuggest` | moved-lazy | `hermes_cli.commands_completion` |
| `SlashCommandCompleter` | moved-lazy | `hermes_cli.commands_completion` |
| `TextArea` | import | `prompt_toolkit.widgets` |
| `Transformation` | import | `prompt_toolkit.layout.processors` |
| `Window` | import | `prompt_toolkit.layout` |
| `WindowAlign` | import | `prompt_toolkit.layout` |
| `base64` | import | `base64` |
| `build_welcome_banner` | moved-lazy | `hermes_cli.banner` |
| `concurrent` | import | `concurrent.futures` |
| `copy` | import | `copy` |
| `display_hermes_home` | moved-lazy | `hermes_constants` |
| `estimate_usage_cost` | moved-lazy | `agent.usage_pricing` |
| `get_all_toolsets` | moved-lazy | `toolsets` |
| `get_job` | moved-lazy | `cron.jobs` |
| `get_toolset_for_tool` | moved-lazy | `model_tools` |
| `get_toolset_info` | moved-lazy | `toolsets` |
| `init_skin_from_config` | moved-lazy | `hermes_cli.skin_engine` |
| `is_browser_debug_ready` | moved-lazy | `hermes_cli.browser_connect` |
| `is_table_divider` | moved-lazy | `agent.markdown_tables` |
| `looks_like_table_row` | moved-lazy | `agent.markdown_tables` |
| `manual_chrome_debug_command` | moved-lazy | `hermes_cli.browser_connect` |
| `print_config_warnings` | moved-lazy | `hermes_cli.config` |
| `prompt_for_secret` | moved-lazy | `hermes_cli.callbacks` |
| `rich_box` | import | `rich` |
| `set_friendly_tool_labels` | moved-lazy | `agent.display` |
| `set_tool_preview_max_len` | moved-lazy | `agent.display` |
| `setup_logging` | moved-lazy | `hermes_logging` |
| `tempfile` | import | `tempfile` |
| `try_launch_chrome_debug` | unrestorable | `no top-level definition on BASE` |

### `cron.jobs`

| name | kind | new location |
|---|---|---|
| `clear_drift_alerted` | restored-def | `(deleted; BASE body restored)` |

### `cron.scheduler`

| name | kind | new location |
|---|---|---|
| `BOT_CHAT_PLATFORM` | moved-lazy | `cron.scheduler_delivery` |
| `SharedRouteAdapters` | moved-lazy | `cron.scheduler_preflight` |
| `asyncio` | import | `asyncio` |
| `cron_delivery_targets` | moved-lazy | `cron.scheduler_delivery` |
| `parse_bot_chat_deliver_token` | moved-lazy | `cron.scheduler_delivery` |
| `shutil` | import | `shutil` |
| `signal` | import | `signal` |

### `cron.scheduler_provider`

| name | kind | new location |
|---|---|---|
| `provider_supports_fire_cancel` | restored-def | `(deleted; BASE body restored)` |

### `gateway.browser_control_artifacts`

| name | kind | new location |
|---|---|---|
| `ArtifactOverwrite` | restored-def | `(deleted; BASE body restored)` |

### `gateway.browser_control_broker`

| name | kind | new location |
|---|---|---|
| `BROWSER_CONTROL_ALL_CAPABILITIES` | restored-def | `(deleted; BASE body restored)` |

### `gateway.config`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |

### `gateway.delivery_ledger`

| name | kind | new location |
|---|---|---|
| `debug_rows` | restored-def | `(deleted; BASE body restored)` |
| `json` | restored-import | `json` |
| `json` | import | `json` |

### `gateway.disk_status`

| name | kind | new location |
|---|---|---|
| `logger` | moved-lazy | `gateway.run` |
| `logging` | import | `logging` |

### `gateway.hosted_room_discussion`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |

### `gateway.hosted_room_driver`

| name | kind | new location |
|---|---|---|
| `Iterator` | import | `typing` |
| `Path` | import | `pathlib` |
| `contextmanager` | import | `contextlib` |
| `re` | import | `re` |

### `gateway.hosted_room_execution_policy`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |
| `re` | import | `re` |

### `gateway.hosted_room_peer`

| name | kind | new location |
|---|---|---|
| `RoomLinkProbe` | restored-def | `(deleted; BASE body restored)` |
| `RoomLinkProbe` | restored-helper | `(deleted; restored as a dependency of select_room_link)` |
| `_LINK_PRIORITY` | restored-helper | `(deleted; restored as a dependency of select_room_link)` |
| `select_room_link` | restored-def | `(deleted; BASE body restored)` |
| `time` | import | `time` |

### `gateway.hosted_room_replicas`

| name | kind | new location |
|---|---|---|
| `MAX_EVENT_JSON_BYTES` | moved-lazy | `gateway.hosted_rooms` |
| `MAX_ROOM_ID_CHARS` | moved-lazy | `gateway.hosted_rooms` |
| `Path` | import | `pathlib` |
| `time` | import | `time` |

### `gateway.hosted_rooms`

| name | kind | new location |
|---|---|---|
| `Iterator` | import | `typing` |
| `NoReturn` | import | `typing` |
| `contextmanager` | import | `contextlib` |
| `time` | import | `time` |

### `gateway.kanban_watchers`

| name | kind | new location |
|---|---|---|
| `Callable` | import | `typing` |
| `Context` | import | `contextvars` |
| `logging` | import | `logging` |
| `re` | import | `re` |
| `sqlite3` | import | `sqlite3` |
| `t` | moved-lazy | `agent.i18n` |

### `gateway.memory_status`

| name | kind | new location |
|---|---|---|
| `logger` | moved-lazy | `gateway.run` |
| `logging` | import | `logging` |

### `gateway.platforms.helpers`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |
| `TYPE_CHECKING` | import | `typing` |
| `TextBatchAggregator` | restored-def | `(deleted; BASE body restored)` |
| `asyncio` | restored-import | `asyncio` |
| `asyncio` | import | `asyncio` |

### `gateway.platforms.qqbot`

| name | kind | new location |
|---|---|---|
| `ApprovalSender` | unrestorable | `no top-level definition on BASE` |

### `gateway.platforms.qqbot.adapter`

| name | kind | new location |
|---|---|---|
| `base64` | import | `base64` |
| `mimetypes` | import | `mimetypes` |

### `gateway.platforms.qqbot.chunked_upload`

| name | kind | new location |
|---|---|---|
| `ApiRequestFn` | restored-def | `(deleted; BASE body restored)` |
| `Optional` | import | `typing` |
| `functools` | import | `functools` |

### `gateway.platforms.qqbot.keyboards`

| name | kind | new location |
|---|---|---|
| `ApprovalSender` | restored-def | `(deleted; BASE body restored)` |
| `Awaitable` | restored-import | `typing` |
| `Awaitable` | import | `typing` |
| `Callable` | restored-import | `typing` |
| `Callable` | import | `typing` |
| `PostMessageFn` | restored-helper | `(deleted; restored as a dependency of ApprovalSender)` |
| `PostMessageFn` | restored-def | `(deleted; BASE body restored)` |
| `logger` | restored-helper | `(deleted; restored as a dependency of ApprovalSender)` |
| `logger` | moved-lazy | `gateway.platforms.qqbot.adapter` |
| `logging` | restored-import | `logging` |
| `logging` | import | `logging` |

### `gateway.platforms.signal`

| name | kind | new location |
|---|---|---|
| `DEFAULT_EXT_TO_MIME` | moved-lazy | `gateway.platforms.media_cache` |
| `TYPING_INTERVAL` | restored-def | `(deleted; BASE body restored)` |

### `gateway.platforms.weixin`

| name | kind | new location |
|---|---|---|
| `MSG_TYPE_USER` | restored-def | `(deleted; BASE body restored)` |
| `struct` | import | `struct` |

### `gateway.platforms.yuanbao`

| name | kind | new location |
|---|---|---|
| `AUTH_FAILED_CODES` | restored-def | `(deleted; BASE body restored)` |
| `AUTH_RETRYABLE_CODES` | restored-def | `(deleted; BASE body restored)` |
| `FileUrlHandler` | restored-def | `(deleted; BASE body restored)` |
| `GroupQueryService` | restored-def | `(deleted; BASE body restored)` |
| `REPLY_REF_TTL_S` | restored-def | `(deleted; BASE body restored)` |
| `get_active_adapter` | restored-def | `(deleted; BASE body restored)` |
| `send_yuanbao_direct` | restored-def | `(deleted; BASE body restored)` |

### `gateway.platforms.yuanbao_media`

| name | kind | new location |
|---|---|---|
| `COS_USE_ACCELERATE` | restored-def | `(deleted; BASE body restored)` |

### `gateway.platforms.yuanbao_proto`

| name | kind | new location |
|---|---|---|
| `DEBUG_MODE` | restored-def | `(deleted; BASE body restored)` |
| `_encode_forward_msg` | restored-helper | `(deleted; restored as a dependency of encode_forward_msg_data)` |
| `_encode_forward_msg_content` | restored-helper | `(deleted; restored as a dependency of encode_forward_msg_data)` |
| `_encode_forward_multimedia` | restored-helper | `(deleted; restored as a dependency of encode_forward_msg_data)` |
| `encode_forward_msg_data` | restored-def | `(deleted; BASE body restored)` |
| `logger` | moved-lazy | `gateway.platforms.base` |
| `logging` | import | `logging` |

### `gateway.relay`

| name | kind | new location |
|---|---|---|
| `relay_bot_username` | restored-def | `(deleted; BASE body restored)` |

### `gateway.relay.adapter`

| name | kind | new location |
|---|---|---|
| `cast` | import | `typing` |

### `gateway.relay.auth`

| name | kind | new location |
|---|---|---|
| `DELIVERY_SIG_HEADER` | restored-def | `(deleted; BASE body restored)` |
| `DELIVERY_TS_HEADER` | restored-def | `(deleted; BASE body restored)` |
| `Optional` | import | `typing` |
| `Sequence` | import | `typing` |
| `_DEFAULT_MAX_SKEW_SECONDS` | restored-helper | `(deleted; restored as a dependency of verify_delivery_signature)` |
| `_delivery_payload` | restored-helper | `(deleted; restored as a dependency of verify_delivery_signature)` |
| `_hmac_hex` | restored-helper | `(deleted; restored as a dependency of verify_delivery_signature)` |
| `verify_delivery_signature` | restored-def | `(deleted; BASE body restored)` |
| `verify_signature` | restored-helper | `(deleted; restored as a dependency of verify_delivery_signature)` |
| `verify_signature` | restored-def | `(deleted; BASE body restored)` |
| `verify_token` | restored-def | `(deleted; BASE body restored)` |

### `gateway.response_filters`

| name | kind | new location |
|---|---|---|
| `SILENT_REPLY_TOKEN` | restored-def | `(deleted; BASE body restored)` |

### `gateway.restart_loop_guard`

| name | kind | new location |
|---|---|---|
| `is_restart_loop_tripped` | restored-def | `(deleted; BASE body restored)` |

### `gateway.run`

| name | kind | new location |
|---|---|---|
| `Awaitable` | import | `typing` |
| `Context` | import | `contextvars` |
| `DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT` | moved-lazy | `gateway.restart` |
| `DEFAULT_HEARTBEAT_INTERVAL_S` | moved-lazy | `gateway.shutdown_watchdog` |
| `DEFAULT_LEASE_WAIT` | moved-lazy | `gateway.turn_lease` |
| `DEFAULT_LOOP_WATCHDOG_INTERVAL_S` | moved-lazy | `gateway.shutdown_watchdog` |
| `DEFAULT_LOOP_WATCHDOG_MAX_STRIKES` | moved-lazy | `gateway.shutdown_watchdog` |
| `DEFAULT_LOOP_WATCHDOG_TIMEOUT_S` | moved-lazy | `gateway.shutdown_watchdog` |
| `EphemeralReply` | moved-lazy | `gateway.platforms.base` |
| `GATEWAY_FATAL_CONFIG_EXIT_CODE` | moved-lazy | `gateway.restart` |
| `GATEWAY_SERVICE_RESTART_EXIT_CODE` | moved-lazy | `gateway.restart` |
| `SessionEntry` | moved-lazy | `gateway.session` |
| `TranscriptReadError` | moved-lazy | `gateway.session_transcript` |
| `TurnContext` | moved-lazy | `gateway.turn_context` |
| `TurnLeaseTimeoutError` | moved-lazy | `gateway.turn_lease` |
| `TurnRunner` | moved-lazy | `gateway.run_turn_runner` |
| `Union` | import | `typing` |
| `arm_shutdown_watchdog` | moved-lazy | `gateway.shutdown_watchdog` |
| `atomic_json_write` | moved-lazy | `utils` |
| `base_url_hostname` | moved-lazy | `utils` |
| `build_auto_tts_output_path` | moved-lazy | `gateway.platforms.base` |
| `build_channel_continuity_note` | moved-lazy | `gateway.session` |
| `build_session_context` | moved-lazy | `gateway.session` |
| `build_session_context_prompt` | moved-lazy | `gateway.session` |
| `consume_detached_task_result` | moved-lazy | `agent.async_utils` |
| `faulthandler` | import | `faulthandler` |
| `functools` | import | `functools` |
| `inspect` | import | `inspect` |
| `is_global_startup_conflict` | moved-lazy | `gateway.restart` |
| `is_shared_multi_user_session` | moved-lazy | `gateway.session` |
| `is_truthy_value` | moved-lazy | `utils` |
| `load_dotenv` | import | `dotenv` |
| `looks_like_telegram_private_chat_id` | moved-lazy | `gateway.delivery` |
| `loop_heartbeat_forever` | moved-lazy | `gateway.shutdown_watchdog` |
| `merge_pending_message_event` | moved-lazy | `gateway.platforms.base` |
| `neutralize_untrusted_inline_text` | moved-lazy | `gateway.session` |
| `parse_cron_drain_timeout` | moved-lazy | `gateway.restart` |
| `parse_restart_after_turn_timeout` | moved-lazy | `gateway.restart` |
| `parse_restart_drain_timeout` | moved-lazy | `gateway.restart` |
| `parse_signal_interrupt_grace_timeout` | moved-lazy | `gateway.restart` |
| `project_compaction_message_for_display` | moved-lazy | `agent.compaction_display` |
| `queue` | import | `queue` |
| `repair_explicit_computer_use_media_paths` | moved-lazy | `gateway.media_repair` |
| `resolve_cron_drain_budget` | moved-lazy | `gateway.restart` |
| `resolve_delivery_transport` | moved-lazy | `gateway.delivery` |
| `resolve_shutdown_watchdog_delay` | moved-lazy | `gateway.shutdown_watchdog` |
| `start_loop_liveness_watchdog` | moved-lazy | `gateway.shutdown_watchdog` |
| `t` | moved-lazy | `agent.i18n` |
| `timedelta` | import | `datetime` |
| `timezone` | import | `datetime` |
| `utf16_len` | moved-lazy | `gateway.platforms.base` |

### `gateway.session`

| name | kind | new location |
|---|---|---|
| `SessionResetPolicy` | moved-lazy | `gateway.config` |
| `TranscriptReadError` | moved-lazy | `gateway.session_transcript` |
| `atomic_replace` | moved-lazy | `utils` |
| `auto_continue_freshness_window` | moved-lazy | `gateway.session_lifecycle` |
| `extract_api_content_sidecar` | moved-lazy | `agent.turn_context` |
| `normalize_whatsapp_identifier` | moved-lazy | `gateway.whatsapp_identity` |
| `replace` | import | `dataclasses` |
| `uuid` | import | `uuid` |

### `gateway.slash_access`

| name | kind | new location |
|---|---|---|
| `Tuple` | import | `typing` |

### `gateway.slash_commands`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |
| `HISTORY_UNREADABLE` | moved-lazy | `gateway.slash_commands_status` |
| `MessageType` | moved-lazy | `gateway.platforms.base` |
| `SessionSource` | moved-lazy | `gateway.session` |
| `base_url_host_matches` | moved-lazy | `utils` |
| `build_session_key` | moved-lazy | `gateway.session` |
| `clear_model_endpoint_credentials` | moved-lazy | `hermes_cli.config` |
| `extract_api_content_sidecar` | moved-lazy | `agent.turn_context` |
| `fetch_account_usage` | moved-lazy | `agent.account_usage` |
| `hashlib` | import | `hashlib` |
| `is_shared_multi_user_session` | moved-lazy | `gateway.session` |
| `render_account_usage_lines` | moved-lazy | `agent.account_usage` |

### `gateway.startup_watchdog`

| name | kind | new location |
|---|---|---|
| `*` | module-stub | `gateway.shutdown_watchdog` |

### `gateway.status`

| name | kind | new location |
|---|---|---|
| `clear_planned_stop_marker` | restored-def | `(deleted; BASE body restored)` |
| `is_gateway_running` | restored-def | `(deleted; BASE body restored)` |

### `gateway.sticker_cache`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |
| `tempfile` | import | `tempfile` |

### `gateway.stream_consumer`

| name | kind | new location |
|---|---|---|
| `MEDIA_TAG_CLEANUP_RE` | moved-lazy | `gateway.platforms.base` |
| `escape_code_fences_for_display` | moved-lazy | `gateway.stream_consumer_fences` |

### `gateway.stream_dispatch`

| name | kind | new location |
|---|---|---|
| `ToolCallFinished` | moved-lazy | `gateway.stream_events` |

### `gateway.systemd_notify`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |

### `hermes_cli.agent_import`

| name | kind | new location |
|---|---|---|
| `backup_memory_file` | restored-def | `(deleted; BASE body restored)` |
| `default_source_dir` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.auth`

| name | kind | new location |
|---|---|---|
| `BaseHTTPRequestHandler` | import | `http.server` |
| `CODEX_OAUTH_USER_AGENT` | moved-lazy | `hermes_cli.auth_constants` |
| `CODEX_QUOTA_PROBE_MIN_INTERVAL_SECONDS` | moved-lazy | `hermes_cli.auth_codex` |
| `DEFAULT_SPOTIFY_REDIRECT_URI` | moved-lazy | `hermes_cli.auth_constants` |
| `DEVICE_AUTH_POLL_INTERVAL_CAP_SECONDS` | moved-lazy | `hermes_cli.auth_constants` |
| `HTTPServer` | import | `http.server` |
| `MINIMAX_OAUTH_GRANT_TYPE` | moved-lazy | `hermes_cli.auth_constants` |
| `NOUS_INFERENCE_INVOKE_SCOPE` | moved-lazy | `hermes_cli.auth_constants` |
| `NOUS_SHARED_STORE_FILENAME` | moved-lazy | `hermes_cli.auth_nous` |
| `OAUTH_OVER_SSH_DOCS_URL` | moved-lazy | `hermes_cli.auth_constants` |
| `QWEN_OAUTH_CLIENT_ID` | moved-lazy | `hermes_cli.auth_constants` |
| `QWEN_OAUTH_TOKEN_URL` | moved-lazy | `hermes_cli.auth_constants` |
| `SINGLE_USE_OAUTH_SINGLETON_FILES` | moved-lazy | `hermes_cli.auth_oauth_grants` |
| `SPOTIFY_ACCESS_TOKEN_REFRESH_SKEW_SECONDS` | moved-lazy | `hermes_cli.auth_constants` |
| `SPOTIFY_DASHBOARD_URL` | moved-lazy | `hermes_cli.auth_constants` |
| `TYPE_CHECKING` | import | `typing` |
| `XAI_OAUTH_DEVICE_CODE_URL` | moved-lazy | `hermes_cli.auth_constants` |
| `XAI_OAUTH_DISCOVERY_URL` | moved-lazy | `hermes_cli.auth_constants` |
| `XAI_OAUTH_ISSUER` | moved-lazy | `hermes_cli.auth_constants` |
| `base64` | import | `base64` |
| `hashlib` | import | `hashlib` |
| `parse_qs` | import | `urllib.parse` |
| `refresh_nous_oauth_pure` | moved-lazy | `hermes_cli.auth_nous` |
| `ssl` | import | `ssl` |
| `subprocess` | import | `subprocess` |
| `sys` | import | `sys` |
| `urlencode` | import | `urllib.parse` |

### `hermes_cli.backup`

| name | kind | new location |
|---|---|---|
| `copy_db_and_verify` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.browser_connect`

| name | kind | new location |
|---|---|---|
| `try_launch_chrome_debug` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.callbacks`

| name | kind | new location |
|---|---|---|
| `approval_callback` | restored-def | `(deleted; BASE body restored)` |
| `clarify_callback` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.checkpoints`

| name | kind | new location |
|---|---|---|
| `cmd_list` | moved-lazy | `hermes_cli.plugins_cmd` |
| `datetime` | import | `datetime` |

### `hermes_cli.commands`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |
| `AutoSuggest` | unrestorable | `no top-level definition on BASE` |
| `Callable` | import | `collections.abc` |
| `Completer` | unrestorable | `no top-level definition on BASE` |
| `Completion` | unrestorable | `no top-level definition on BASE` |
| `Dict` | import | `typing` |
| `Mapping` | import | `collections.abc` |
| `Optional` | import | `typing` |
| `Sequence` | import | `collections.abc` |
| `SlashCommandAutoSuggest` | moved-lazy | `hermes_cli.commands_completion` |
| `SlashCommandCompleter` | moved-lazy | `hermes_cli.commands_completion` |
| `Suggestion` | unrestorable | `no top-level definition on BASE` |
| `Tuple` | import | `typing` |
| `_CMD_NAME_LIMIT` | restored-helper | `(deleted; restored as a dependency of discord_skill_commands)` |
| `_clamp_command_names` | restored-helper | `(deleted; restored as a dependency of discord_skill_commands)` |
| `_collect_gateway_skill_entries` | restored-helper | `(deleted; restored as a dependency of discord_skill_commands)` |
| `_requires_argument` | restored-helper | `(deleted; restored as a dependency of discord_skill_commands)` |
| `discord_skill_commands` | restored-def | `(deleted; BASE body restored)` |
| `discord_skill_commands_by_category` | moved-lazy | `hermes_cli.commands_platforms` |
| `field` | import | `dataclasses` |
| `key` | unrestorable | `no top-level definition on BASE` |
| `m` | unrestorable | `no top-level definition on BASE` |
| `os` | import | `os` |
| `shutil` | import | `shutil` |
| `slack_app_manifest` | moved-lazy | `hermes_cli.commands_platforms` |
| `slack_native_slashes` | moved-lazy | `hermes_cli.commands_platforms` |
| `slack_subcommand_map` | moved-lazy | `hermes_cli.commands_platforms` |
| `subprocess` | import | `subprocess` |
| `telegram_bot_commands` | moved-lazy | `hermes_cli.commands_platforms` |
| `telegram_menu_commands` | moved-lazy | `hermes_cli.commands_platforms` |
| `telegram_menu_max_commands` | moved-lazy | `hermes_cli.commands_platforms` |
| `time` | import | `time` |

### `hermes_cli.config`

| name | kind | new location |
|---|---|---|
| `_install_method_project_root` | restored-helper | `(deleted; restored as a dependency of stamp_install_method)` |
| `normalize_route_base_url` | moved-lazy | `hermes_cli.route_identity` |
| `stamp_install_method` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.console_engine`

| name | kind | new location |
|---|---|---|
| `shlex` | import | `shlex` |

### `hermes_cli.curses_ui`

| name | kind | new location |
|---|---|---|
| `Protocol` | import | `typing` |

### `hermes_cli.dashboard_auth.audit`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `hermes_cli.dashboard_auth.middleware`

| name | kind | new location |
|---|---|---|
| `DashboardAuthProvider` | moved-lazy | `hermes_cli.dashboard_auth.base` |

### `hermes_cli.dingtalk_auth`

| name | kind | new location |
|---|---|---|
| `logger` | moved-lazy | `hermes_cli.auth` |
| `logging` | import | `logging` |

### `hermes_cli.doctor`

| name | kind | new location |
|---|---|---|
| `FTS_STORAGE_VERSION` | moved-lazy | `hermes_state_common` |
| `OPENROUTER_MODELS_URL` | moved-lazy | `hermes_constants` |
| `Path` | import | `pathlib` |
| `STATE_DB_SIZE_WARN_BYTES` | moved-lazy | `hermes_cli.doctor_state` |
| `agent_browser_runnable` | moved-lazy | `hermes_constants` |
| `base_url_host_matches` | moved-lazy | `utils` |
| `check_certificates` | moved-lazy | `hermes_cli.doctor_platform` |
| `check_fail` | restored-def | `(deleted; BASE body restored)` |
| `check_macos_full_disk_access` | moved-lazy | `hermes_cli.doctor_platform` |
| `check_macos_tcc_anchor` | moved-lazy | `hermes_cli.doctor_platform` |
| `check_macos_tcc_grants` | moved-lazy | `hermes_cli.doctor_platform` |
| `check_ok` | restored-def | `(deleted; BASE body restored)` |
| `check_warn` | restored-def | `(deleted; BASE body restored)` |
| `collect_deprecated_config_keys` | moved-lazy | `hermes_cli.doctor_config` |
| `collect_deprecated_env_vars` | moved-lazy | `hermes_cli.doctor_config` |
| `collect_relay_plugin_cutover_findings` | moved-lazy | `hermes_cli.doctor_config` |
| `describe_vercel_auth` | moved-lazy | `hermes_cli.vercel_auth` |
| `detect_install_method` | moved-lazy | `hermes_cli.config` |
| `importlib` | import | `importlib.util` |
| `is_nix_install_method` | moved-lazy | `hermes_cli.config` |
| `managed_scope_check` | moved-lazy | `hermes_cli.doctor_config` |
| `recommended_update_command_for_method` | moved-lazy | `hermes_cli.config` |
| `report_deprecated_config_and_env` | moved-lazy | `hermes_cli.doctor_config` |
| `shutil` | import | `shutil` |
| `subprocess` | import | `subprocess` |

### `hermes_cli.doctor_live`

| name | kind | new location |
|---|---|---|
| `ELEVENLABS_VOICES_URL` | restored-def | `(deleted; BASE body restored)` |
| `FAL_MODELS_URL` | restored-def | `(deleted; BASE body restored)` |
| `FIRECRAWL_HEALTH_URL` | restored-def | `(deleted; BASE body restored)` |
| `GROQ_MODELS_URL` | restored-def | `(deleted; BASE body restored)` |
| `OPENAI_MODELS_URL` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.focus_view`

| name | kind | new location |
|---|---|---|
| `FOCUS_USAGE` | restored-def | `(deleted; BASE body restored)` |
| `effective_tool_progress_mode` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.foreign_sessions`

| name | kind | new location |
|---|---|---|
| `list_claude_sessions` | restored-def | `(deleted; BASE body restored)` |
| `list_codex_sessions` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.gateway`

| name | kind | new location |
|---|---|---|
| `DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT` | moved-lazy | `gateway.restart` |
| `print_systemd_linger_guidance` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.gitlock`

| name | kind | new location |
|---|---|---|
| `is_ancestor_of_head` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.heartbeat`

| name | kind | new location |
|---|---|---|
| `Dict` | import | `typing` |

### `hermes_cli.journey`

| name | kind | new location |
|---|---|---|
| `cmd_journey` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.kanban`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |

### `hermes_cli.kanban_db`

| name | kind | new location |
|---|---|---|
| `DEFAULT_BUSY_TIMEOUT_MS` | moved-lazy | `hermes_cli.kanban_db_connect` |
| `DEFAULT_LOG_BACKUP_COUNT` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `DEFAULT_LOG_ROTATE_BYTES` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `DEFAULT_SPAWN_FAILURE_LIMIT` | restored-def | `(deleted; BASE body restored)` |
| `DERIVED_MAX_IN_PROGRESS_CEILING` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `DERIVED_MAX_IN_PROGRESS_FLOOR` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `KanbanDbCorruptError` | moved-lazy | `hermes_cli.kanban_db_connect` |
| `MEMORY_GUARD_MB_PER_WORKER` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `Mapping` | import | `typing` |
| `RepairResult` | moved-lazy | `hermes_cli.kanban_db_connect` |
| `add_notify_sub` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `advance_notify_cursor` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `check_respawn_guard` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `claim_unseen_events_for_sub` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `configured_max_in_progress` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `connect` | moved-lazy | `hermes_cli.kanban_db_connect` |
| `connect_closing` | moved-lazy | `hermes_cli.kanban_db_connect` |
| `count_notify_subs` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `count_running_tasks` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `count_running_tasks_other_boards` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `derive_default_max_in_progress` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `detect_crashed_workers` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `detect_stale_running` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `dispatch_once` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `enforce_max_runtime` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `field` | import | `dataclasses` |
| `has_spawnable_ready` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `has_spawnable_review` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `hashlib` | import | `hashlib` |
| `heartbeat_worker` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `list_notify_subs` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `parent_results` | restored-def | `(deleted; BASE body restored)` |
| `purge_stale_done_notify_subs` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `random` | import | `random` |
| `reap_worker_zombies` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `reconcile_orphaned_running` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `remove_notify_sub` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `repair_db` | moved-lazy | `hermes_cli.kanban_db_connect` |
| `resolve_max_in_progress` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `resolve_workspace` | moved-lazy | `hermes_cli.kanban_db_workspace` |
| `review_dispatch_enabled` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `rewind_notify_cursor` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `run_daemon` | moved-lazy | `hermes_cli.kanban_db_dispatch` |
| `set_branch_name` | moved-lazy | `hermes_cli.kanban_db_workspace` |
| `set_workspace_path` | moved-lazy | `hermes_cli.kanban_db_workspace` |
| `shutil` | import | `shutil` |
| `threading` | import | `threading` |
| `unseen_events_for_sub` | moved-lazy | `hermes_cli.kanban_db_notify` |
| `worker_log_rotation_config` | moved-lazy | `hermes_cli.kanban_db_dispatch` |

### `hermes_cli.kanban_decompose`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |
| `os` | import | `os` |

### `hermes_cli.kanban_diagnostics`

| name | kind | new location |
|---|---|---|
| `DIAGNOSTIC_KINDS` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.local_runtime.binaries`

| name | kind | new location |
|---|---|---|
| `get_hermes_home` | moved-lazy | `hermes_constants` |

### `hermes_cli.local_runtime.bootstrap`

| name | kind | new location |
|---|---|---|
| `get_hermes_home` | moved-lazy | `hermes_constants` |

### `hermes_cli.local_runtime.capabilities`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |
| `urllib` | import | `urllib.request` |

### `hermes_cli.local_runtime.catalog`

| name | kind | new location |
|---|---|---|
| `find_variant` | restored-def | `(deleted; BASE body restored)` |
| `re` | import | `re` |
| `recommended_id` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.local_runtime.hf_browse`

| name | kind | new location |
|---|---|---|
| `field` | import | `dataclasses` |

### `hermes_cli.main`

| name | kind | new location |
|---|---|---|
| `hashlib` | import | `hashlib` |
| `line_input` | moved-lazy | `hermes_cli.cli_output` |
| `shlex` | import | `shlex` |
| `stat` | import | `stat` |
| `tempfile` | import | `tempfile` |

### `hermes_cli.mcp_picker`

| name | kind | new location |
|---|---|---|
| `color` | moved-lazy | `hermes_cli.colors` |

### `hermes_cli.mcp_security`

| name | kind | new location |
|---|---|---|
| `is_mcp_server_entry_suspicious` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.middleware`

| name | kind | new location |
|---|---|---|
| `API_EXECUTION_MIDDLEWARE` | restored-def | `(deleted; BASE body restored)` |
| `API_REQUEST_MIDDLEWARE` | restored-def | `(deleted; BASE body restored)` |
| `apply_api_request_middleware` | restored-def | `(deleted; BASE body restored)` |
| `run_api_execution_middleware` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.moa_config`

| name | kind | new location |
|---|---|---|
| `build_moa_turn_prompt` | restored-def | `(deleted; BASE body restored)` |
| `encode_moa_turn` | restored-helper | `(deleted; restored as a dependency of build_moa_turn_prompt)` |
| `encode_moa_turn` | restored-def | `(deleted; BASE body restored)` |
| `list_moa_presets` | restored-def | `(deleted; BASE body restored)` |
| `set_active_moa_preset` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.model_setup_flows`

| name | kind | new location |
|---|---|---|
| `BEDROCK_GEO_PREFIXES` | moved-lazy | `hermes_cli.model_setup_flows_bedrock` |
| `bedrock_model_routable_from_region` | moved-lazy | `hermes_cli.model_setup_flows_bedrock` |
| `bedrock_region_geo_prefix` | moved-lazy | `hermes_cli.model_setup_flows_bedrock` |
| `custom_provider_slug` | moved-lazy | `hermes_cli.providers` |
| `line_input` | moved-lazy | `hermes_cli.cli_output` |
| `subprocess` | import | `subprocess` |
| `urllib` | import | `urllib.parse` |

### `hermes_cli.model_switch`

| name | kind | new location |
|---|---|---|
| `List` | import | `typing` |
| `base_url_host_matches` | moved-lazy | `utils` |
| `custom_provider_slug` | moved-lazy | `hermes_cli.providers` |
| `http` | import | `http.client` |
| `list_picker_providers` | moved-lazy | `hermes_cli.model_switch_providers` |
| `prewarm_picker_cache_async` | moved-lazy | `hermes_cli.model_switch_providers` |
| `time` | import | `time` |

### `hermes_cli.models`

| name | kind | new location |
|---|---|---|
| `LMStudioLoadResult` | moved-lazy | `hermes_cli.models_local` |
| `NamedTuple` | import | `typing` |
| `PROVIDER_GROUPS` | moved-lazy | `hermes_cli.models_catalog_static` |
| `ProviderEntry` | moved-lazy | `hermes_cli.models_catalog_static` |
| `_OPENCODE_KEYLESS_EXTRA_SLUGS` | restored-helper | `(deleted; restored as a dependency of is_opencode_zen_free_model)` |
| `atomic_json_write` | moved-lazy | `utils` |
| `base_url_host_matches` | moved-lazy | `utils` |
| `compute_sale_discount` | moved-lazy | `hermes_cli.models_pricing` |
| `ensure_lmstudio_model_loaded` | moved-lazy | `hermes_cli.models_local` |
| `fetch_ai_gateway_pricing` | moved-lazy | `hermes_cli.models_pricing` |
| `fetch_lmstudio_models` | moved-lazy | `hermes_cli.models_local` |
| `fetch_models_with_pricing` | moved-lazy | `hermes_cli.models_pricing` |
| `fetch_ollama_local_models` | moved-lazy | `hermes_cli.models_local` |
| `get_cached_nous_inference_base_url` | moved-lazy | `hermes_cli.models_pricing` |
| `get_close_matches` | import | `difflib` |
| `get_pricing_for_provider` | moved-lazy | `hermes_cli.models_pricing` |
| `group_providers` | moved-lazy | `hermes_cli.models_catalog_static` |
| `http` | import | `http.client` |
| `is_nous_free_tier` | restored-def | `(deleted; BASE body restored)` |
| `is_opencode_zen_free_model` | restored-def | `(deleted; BASE body restored)` |
| `lmstudio_model_reasoning_options` | moved-lazy | `hermes_cli.models_local` |
| `nous_catalog_url` | moved-lazy | `hermes_cli.models_reasoning_caps` |
| `nous_model_reasoning_capabilities` | moved-lazy | `hermes_cli.models_reasoning_caps` |
| `nous_policy_allowed_ids` | moved-lazy | `hermes_cli.models_pricing` |
| `ollama_model_supports_thinking` | moved-lazy | `hermes_cli.models_local` |
| `openrouter_model_reasoning_capabilities` | moved-lazy | `hermes_cli.models_reasoning_caps` |
| `parse_openrouter_reasoning_capabilities` | moved-lazy | `hermes_cli.models_reasoning_caps` |
| `peek_cached_pricing` | moved-lazy | `hermes_cli.models_pricing` |
| `pricing_cache_scope` | moved-lazy | `hermes_cli.models_pricing` |
| `probe_lmstudio_models` | moved-lazy | `hermes_cli.models_local` |
| `probe_ollama_local_models` | moved-lazy | `hermes_cli.models_local` |
| `provider_group_for_slug` | moved-lazy | `hermes_cli.models_catalog_static` |
| `refresh_reasoning_caps_async` | moved-lazy | `hermes_cli.models_reasoning_caps` |
| `restrict_to_nous_policy` | moved-lazy | `hermes_cli.models_pricing` |
| `should_use_ollama_native_catalog` | moved-lazy | `hermes_cli.models_local` |
| `url_origin` | moved-lazy | `hermes_cli.urllib_security` |
| `validate_requested_model` | moved-lazy | `hermes_cli.models_validate` |
| `warm_nous_reasoning_caps_async` | moved-lazy | `hermes_cli.models_reasoning_caps` |
| `warm_openrouter_reasoning_caps_async` | moved-lazy | `hermes_cli.models_reasoning_caps` |

### `hermes_cli.nous_billing`

| name | kind | new location |
|---|---|---|
| `BILLING_MANAGE_SCOPE` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.nous_subscription`

| name | kind | new location |
|---|---|---|
| `managed_nous_tools_enabled` | moved-lazy | `tools.tool_backend_helpers` |

### `hermes_cli.observability.relay_runtime`

| name | kind | new location |
|---|---|---|
| `*` | module-stub | `agent.relay_runtime` |

### `hermes_cli.observability.relay_shared_metrics`

| name | kind | new location |
|---|---|---|
| `CLIENT_ACTIVE_MARK` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `MODEL_CALL_PROFILE_MODEL` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `SCHEMA_KEY` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `SCHEMA_VERSION` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `SKILL_LIFECYCLE_MARK` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `SKILL_LOAD_MARK` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `TOOL_APPROVAL_MARK` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `TOOL_CALL_SCOPE` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `model_call_fields` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `prepare_session_start` | restored-def | `(deleted; BASE body restored)` |
| `skill_lifecycle_fields` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `skill_load_fields` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `task_start_fields` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `task_terminal_fields` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `task_terminal_state` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `tool_approval_outcome` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `tool_category` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |
| `tool_terminal_fields` | moved-lazy | `hermes_cli.observability.shared_metrics_contract` |

### `hermes_cli.onepassword_secrets_cli`

| name | kind | new location |
|---|---|---|
| `Table` | import | `rich.table` |
| `masked_secret_prompt` | moved-lazy | `hermes_cli.secret_prompt` |
| `sys` | import | `sys` |

### `hermes_cli.platform_actions`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |

### `hermes_cli.plugins`

| name | kind | new location |
|---|---|---|
| `CAPABILITY_REGISTRY` | moved-lazy | `hermes_cli.plugin_capabilities` |
| `ENTRY_POINT_CAPABILITIES_GROUP` | moved-lazy | `hermes_cli.plugins_discovery` |
| `Iterable` | import | `typing` |
| `LEGACY_RELAY_PLUGIN_KEYS` | moved-lazy | `hermes_cli.relay_plugin_cutover` |
| `MAX_SYSTEM_PROMPT_SECTIONS` | moved-lazy | `hermes_cli.plugins_dispatch` |
| `OBSERVER_SCHEMA_VERSION` | moved-lazy | `hermes_cli.middleware` |
| `Type` | import | `typing` |
| `VALID_CAPABILITY_IDS` | moved-lazy | `hermes_cli.plugin_capabilities` |
| `cfg_get` | moved-lazy | `hermes_cli.config` |
| `contextmanager` | import | `contextlib` |
| `contextvars` | import | `contextvars` |
| `copy` | import | `copy` |
| `fast_safe_load` | moved-lazy | `utils` |
| `format_system_prompt_section` | moved-lazy | `hermes_cli.plugins_dispatch` |
| `get_plugin_subscriptions` | restored-def | `(deleted; BASE body restored)` |
| `hashlib` | import | `hashlib` |
| `reset_hermes_home_override` | moved-lazy | `hermes_constants` |
| `set_hermes_home_override` | moved-lazy | `hermes_constants` |
| `time` | import | `time` |
| `unload_plugins` | restored-def | `(deleted; BASE body restored)` |
| `wraps` | import | `functools` |
| `yaml` | unrestorable | `no top-level definition on BASE` |

### `hermes_cli.plugins_cmd`

| name | kind | new location |
|---|---|---|
| `importlib` | import | `importlib.metadata` |

### `hermes_cli.profile_describer`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |

### `hermes_cli.profile_distribution`

| name | kind | new location |
|---|---|---|
| `is_excluded_skill_path` | moved-lazy | `agent.skill_utils` |

### `hermes_cli.profiles`

| name | kind | new location |
|---|---|---|
| `has_bundled_skills_opt_out` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.runtime_provider`

| name | kind | new location |
|---|---|---|
| `custom_provider_aliases` | moved-lazy | `hermes_cli.providers` |
| `custom_provider_slug` | moved-lazy | `hermes_cli.providers` |
| `os` | import | `os` |

### `hermes_cli.security_advisories`

| name | kind | new location |
|---|---|---|
| `render_doctor_section` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.setup`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |
| `Dict` | import | `typing` |
| `Optional` | import | `typing` |
| `get_nous_subscription_features` | moved-lazy | `hermes_cli.nous_subscription` |
| `get_optional_skills_dir` | moved-lazy | `hermes_constants` |
| `json` | import | `json` |
| `managed_nous_tools_enabled` | moved-lazy | `tools.tool_backend_helpers` |
| `shutil` | import | `shutil` |

### `hermes_cli.slack_cli`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `hermes_cli.sqlite_safe_read`

| name | kind | new location |
|---|---|---|
| `SQLITE_HEADER_MAGIC` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.status`

| name | kind | new location |
|---|---|---|
| `format_nous_portal_entitlement_message` | moved-lazy | `hermes_cli.nous_account` |
| `get_nous_portal_account_info` | moved-lazy | `hermes_cli.nous_account` |
| `get_nous_subscription_features` | moved-lazy | `hermes_cli.nous_subscription` |
| `managed_nous_tools_enabled` | moved-lazy | `tools.tool_backend_helpers` |
| `redact_key` | moved-lazy | `hermes_cli.config` |
| `subprocess` | import | `subprocess` |

### `hermes_cli.telegram_managed_bot`

| name | kind | new location |
|---|---|---|
| `DEFAULT_MANAGER_BOT` | restored-def | `(deleted; BASE body restored)` |
| `_USERNAME_SLUG_ALPHABET` | restored-helper | `(deleted; restored as a dependency of generate_bot_username)` |
| `auto_setup_telegram_bot` | restored-def | `(deleted; BASE body restored)` |
| `generate_bot_username` | restored-def | `(deleted; BASE body restored)` |
| `generate_deep_link` | restored-def | `(deleted; BASE body restored)` |
| `generate_pairing_nonce` | restored-def | `(deleted; BASE body restored)` |
| `generate_username_slug` | restored-helper | `(deleted; restored as a dependency of generate_bot_username)` |
| `generate_username_slug` | restored-def | `(deleted; BASE body restored)` |
| `poll_for_token` | restored-def | `(deleted; BASE body restored)` |
| `poll_pairing_once` | restored-def | `(deleted; BASE body restored)` |
| `secrets` | restored-import | `secrets` |
| `secrets` | import | `secrets` |
| `urllib` | restored-import | `urllib.parse` |
| `urllib` | import | `urllib.parse` |

### `hermes_cli.tools_config`

| name | kind | new location |
|---|---|---|
| `MANAGED_FEATURE_COVERAGE_CATEGORY` | moved-lazy | `hermes_cli.nous_subscription` |
| `NOUS_MANAGED_PROVIDER` | moved-lazy | `tools.tool_backend_helpers` |
| `base_url_hostname` | moved-lazy | `utils` |
| `fal_key_is_configured` | moved-lazy | `tools.tool_backend_helpers` |
| `format_nous_portal_entitlement_message` | moved-lazy | `hermes_cli.nous_account` |
| `is_truthy_value` | moved-lazy | `utils` |
| `save_env_value` | moved-lazy | `hermes_cli.config` |
| `shutil` | import | `shutil` |
| `subprocess` | import | `subprocess` |
| `sys` | import | `sys` |

### `hermes_cli.uninstall`

| name | kind | new location |
|---|---|---|
| `find_shell_configs` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.update_cmd`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |
| `datetime` | import | `datetime` |
| `hashlib` | import | `hashlib` |
| `json` | import | `json` |

### `hermes_cli.update_inventory`

| name | kind | new location |
|---|---|---|
| `Path` | import | `pathlib` |
| `os` | import | `os` |

### `hermes_cli.web_deps`

| name | kind | new location |
|---|---|---|
| `get_dashboard_health` | restored-def | `(deleted; BASE body restored)` |
| `get_session_token` | restored-def | `(deleted; BASE body restored)` |
| `has_valid_session_token` | restored-def | `(deleted; BASE body restored)` |
| `late_attr` | restored-def | `(deleted; BASE body restored)` |

### `hermes_cli.web_routers.cron`

| name | kind | new location |
|---|---|---|
| `logging` | import | `logging` |

### `hermes_cli.web_routers.mcp`

| name | kind | new location |
|---|---|---|
| `logging` | import | `logging` |

### `hermes_cli.web_routers.sessions`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |
| `Dict` | import | `typing` |
| `logging` | import | `logging` |

### `hermes_cli.web_routers.skills`

| name | kind | new location |
|---|---|---|
| `LateState` | moved-lazy | `hermes_cli.web_deps` |
| `logging` | import | `logging` |

### `hermes_cli.web_routers.tools`

| name | kind | new location |
|---|---|---|
| `LateState` | moved-lazy | `hermes_cli.web_deps` |
| `logging` | import | `logging` |

### `hermes_cli.web_server`

| name | kind | new location |
|---|---|---|
| `AudioTranscriptionRequest` | moved-lazy | `hermes_cli.web_models` |
| `AutomationBlueprintInstantiate` | moved-lazy | `hermes_cli.web_models` |
| `BackupRequest` | moved-lazy | `hermes_cli.web_models` |
| `BaseModel` | unrestorable | `no top-level definition on BASE` |
| `BulkDeleteSessions` | moved-lazy | `hermes_cli.web_models` |
| `CONFIG_SCHEMA` | moved-lazy | `hermes_cli.web_server_config` |
| `ChatImageUpload` | moved-lazy | `hermes_cli.web_models` |
| `ConfigUpdate` | moved-lazy | `hermes_cli.web_models` |
| `CredentialPoolAdd` | moved-lazy | `hermes_cli.web_models` |
| `CronJobCreate` | moved-lazy | `hermes_cli.web_models` |
| `CronJobUpdate` | moved-lazy | `hermes_cli.web_models` |
| `CuratorPause` | moved-lazy | `hermes_cli.web_models` |
| `CustomEndpointUpdate` | moved-lazy | `hermes_cli.web_models` |
| `DEFAULT_CONFIG` | moved-lazy | `hermes_cli.config` |
| `DebugShareRequest` | moved-lazy | `hermes_cli.web_models` |
| `EnvVarDelete` | moved-lazy | `hermes_cli.web_models` |
| `EnvVarReveal` | moved-lazy | `hermes_cli.web_models` |
| `EnvVarUpdate` | moved-lazy | `hermes_cli.web_models` |
| `File` | unrestorable | `no top-level definition on BASE` |
| `FileResponse` | unrestorable | `no top-level definition on BASE` |
| `FontSetBody` | moved-lazy | `hermes_cli.web_models` |
| `Form` | unrestorable | `no top-level definition on BASE` |
| `FsWriteText` | moved-lazy | `hermes_cli.web_models` |
| `GitBranchSwitchBody` | moved-lazy | `hermes_cli.web_models` |
| `GitCommitBody` | moved-lazy | `hermes_cli.web_models` |
| `GitFileBody` | moved-lazy | `hermes_cli.web_models` |
| `GitPathBody` | moved-lazy | `hermes_cli.web_models` |
| `GitWorktreeAddBody` | moved-lazy | `hermes_cli.web_models` |
| `GitWorktreeRemoveBody` | moved-lazy | `hermes_cli.web_models` |
| `HTMLResponse` | unrestorable | `no top-level definition on BASE` |
| `HookCreate` | moved-lazy | `hermes_cli.web_models` |
| `HookDelete` | moved-lazy | `hermes_cli.web_models` |
| `ImportRequest` | moved-lazy | `hermes_cli.web_models` |
| `LearningNodeEdit` | moved-lazy | `hermes_cli.web_models` |
| `LearningNodeRef` | moved-lazy | `hermes_cli.web_models` |
| `List` | import | `typing` |
| `Literal` | import | `typing` |
| `MCPCatalogInstall` | moved-lazy | `hermes_cli.web_models` |
| `MCPEnabledToggle` | moved-lazy | `hermes_cli.web_models` |
| `MCPServerCreate` | moved-lazy | `hermes_cli.web_models` |
| `MCPServersReplace` | moved-lazy | `hermes_cli.web_models` |
| `ManagedDirectoryCreate` | moved-lazy | `hermes_cli.web_models` |
| `ManagedFileDelete` | moved-lazy | `hermes_cli.web_models` |
| `ManagedFileUpload` | moved-lazy | `hermes_cli.web_models` |
| `ManagedFilesPolicy` | moved-lazy | `hermes_cli.web_server_files` |
| `MemoryProviderConfigUpdate` | moved-lazy | `hermes_cli.web_models` |
| `MemoryProviderSelect` | moved-lazy | `hermes_cli.web_models` |
| `MemoryProviderSetupRequest` | moved-lazy | `hermes_cli.web_models` |
| `MemoryReset` | moved-lazy | `hermes_cli.web_models` |
| `MessagingPlatformUpdate` | moved-lazy | `hermes_cli.web_models` |
| `MoaConfigPayload` | moved-lazy | `hermes_cli.web_models` |
| `MoaModelSlot` | moved-lazy | `hermes_cli.web_models` |
| `MoaPresetPayload` | moved-lazy | `hermes_cli.web_models` |
| `ModelAssignment` | moved-lazy | `hermes_cli.web_models` |
| `OAuthSubmitBody` | moved-lazy | `hermes_cli.web_models` |
| `OPTIONAL_ENV_VARS` | moved-lazy | `hermes_cli.config` |
| `PairingApprove` | moved-lazy | `hermes_cli.web_models` |
| `PairingRevoke` | moved-lazy | `hermes_cli.web_models` |
| `ProfileActiveUpdate` | moved-lazy | `hermes_cli.web_models` |
| `ProfileCreate` | moved-lazy | `hermes_cli.web_models` |
| `ProfileDescribeAuto` | moved-lazy | `hermes_cli.web_models` |
| `ProfileDescriptionUpdate` | moved-lazy | `hermes_cli.web_models` |
| `ProfileModelUpdate` | moved-lazy | `hermes_cli.web_models` |
| `ProfileRename` | moved-lazy | `hermes_cli.web_models` |
| `ProfileSoulUpdate` | moved-lazy | `hermes_cli.web_models` |
| `ProviderConfigSchema` | moved-lazy | `plugins.memory.config_schema` |
| `ProviderField` | moved-lazy | `plugins.memory.config_schema` |
| `PtyBridge` | moved-lazy | `hermes_cli.pty_bridge` |
| `PtySessionRegistry` | moved-lazy | `hermes_cli.pty_session` |
| `PtyUnavailableError` | moved-lazy | `hermes_cli.pty_bridge` |
| `Query` | unrestorable | `no top-level definition on BASE` |
| `RawConfigUpdate` | moved-lazy | `hermes_cli.web_models` |
| `RegistryFull` | moved-lazy | `hermes_cli.pty_session` |
| `Response` | unrestorable | `no top-level definition on BASE` |
| `STORAGE_HONCHO_HOST_BLOCK` | moved-lazy | `plugins.memory.config_schema` |
| `SecretStr` | unrestorable | `no top-level definition on BASE` |
| `SessionImport` | moved-lazy | `hermes_cli.web_models` |
| `SessionPrune` | moved-lazy | `hermes_cli.web_models` |
| `SessionRename` | moved-lazy | `hermes_cli.web_models` |
| `SkillContentUpdate` | moved-lazy | `hermes_cli.web_models` |
| `SkillCreate` | moved-lazy | `hermes_cli.web_models` |
| `SkillInstallRequest` | moved-lazy | `hermes_cli.web_models` |
| `SkillToggle` | moved-lazy | `hermes_cli.web_models` |
| `SkillUninstallRequest` | moved-lazy | `hermes_cli.web_models` |
| `SkillsUpdateRequest` | moved-lazy | `hermes_cli.web_models` |
| `StaticFiles` | unrestorable | `no top-level definition on BASE` |
| `TTSLeaseRequest` | moved-lazy | `hermes_cli.web_models` |
| `TTSSpeakRequest` | moved-lazy | `hermes_cli.web_models` |
| `TelegramOnboardingApply` | moved-lazy | `hermes_cli.web_models` |
| `TelegramOnboardingStart` | moved-lazy | `hermes_cli.web_models` |
| `TerminalBackendSelect` | moved-lazy | `hermes_cli.web_models` |
| `ThemeSetBody` | moved-lazy | `hermes_cli.web_models` |
| `ToolsetEnvUpdate` | moved-lazy | `hermes_cli.web_models` |
| `ToolsetModelSelect` | moved-lazy | `hermes_cli.web_models` |
| `ToolsetPostSetup` | moved-lazy | `hermes_cli.web_models` |
| `ToolsetProviderSelect` | moved-lazy | `hermes_cli.web_models` |
| `ToolsetToggle` | moved-lazy | `hermes_cli.web_models` |
| `UploadFile` | unrestorable | `no top-level definition on BASE` |
| `WebSocket` | unrestorable | `no top-level definition on BASE` |
| `WebSocketDisconnect` | unrestorable | `no top-level definition on BASE` |
| `WebhookCreate` | moved-lazy | `hermes_cli.web_models` |
| `WebhookEnabledToggle` | moved-lazy | `hermes_cli.web_models` |
| `WhatsAppOnboardingApply` | moved-lazy | `hermes_cli.web_models` |
| `WhatsAppOnboardingStart` | moved-lazy | `hermes_cli.web_models` |
| `activate_custom_endpoint` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `add_credential_pool_entry` | moved-lazy | `hermes_cli.web_routers.ops` |
| `add_mcp_server` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `apply_telegram_onboarding` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `apply_whatsapp_onboarding` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `approve_pairing` | moved-lazy | `hermes_cli.web_routers.ops` |
| `atexit` | import | `atexit` |
| `auth_mcp_server` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `base64` | import | `base64` |
| `binascii` | import | `binascii` |
| `build_cron_model_impact` | moved-lazy | `hermes_cli.config` |
| `bulk_delete_sessions_endpoint` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `cancel_oauth_session` | moved-lazy | `hermes_cli.web_routers.oauth` |
| `cancel_telegram_onboarding` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `cancel_whatsapp_onboarding` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `cfg_get` | moved-lazy | `hermes_cli.config` |
| `check_config_version` | moved-lazy | `hermes_cli.config` |
| `check_hermes_update` | moved-lazy | `hermes_cli.web_routers.actions` |
| `clear_model_endpoint_credentials` | moved-lazy | `hermes_cli.config` |
| `clear_pending_pairing` | moved-lazy | `hermes_cli.web_routers.ops` |
| `coerce_provider_id` | moved-lazy | `hermes_cli.config` |
| `concurrent` | import | `concurrent.futures` |
| `console_ws` | moved-lazy | `hermes_cli.web_routers.chat_ws` |
| `contextlib` | import | `contextlib` |
| `contextmanager` | import | `contextlib` |
| `count_empty_sessions_endpoint` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `create_cron_job` | moved-lazy | `hermes_cli.web_routers.cron` |
| `create_hook` | moved-lazy | `hermes_cli.web_routers.ops` |
| `create_managed_directory` | moved-lazy | `hermes_cli.web_routers.files` |
| `create_profile_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `create_skill` | moved-lazy | `hermes_cli.web_routers.skills` |
| `create_webhook` | moved-lazy | `hermes_cli.web_routers.ops` |
| `cron_fire_webhook` | moved-lazy | `hermes_cli.web_routers.cron` |
| `custom_endpoint_key_env` | moved-lazy | `hermes_cli.config` |
| `dataclass` | import | `dataclasses` |
| `datetime` | import | `datetime` |
| `delete_agent_plugin` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `delete_cron_job` | moved-lazy | `hermes_cli.web_routers.cron` |
| `delete_custom_endpoint` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `delete_empty_sessions_endpoint` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `delete_hook` | moved-lazy | `hermes_cli.web_routers.ops` |
| `delete_learning_node` | moved-lazy | `hermes_cli.web_routers.status` |
| `delete_managed_file` | moved-lazy | `hermes_cli.web_routers.files` |
| `delete_profile_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `delete_session_endpoint` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `delete_webhook` | moved-lazy | `hermes_cli.web_routers.ops` |
| `derive_gateway_busy` | moved-lazy | `gateway.status` |
| `derive_gateway_drainable` | moved-lazy | `gateway.status` |
| `describe_profile_auto_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `detect_install_method` | moved-lazy | `hermes_cli.config` |
| `disconnect_oauth_provider` | moved-lazy | `hermes_cli.web_routers.oauth` |
| `download_dashboard_backup` | moved-lazy | `hermes_cli.web_routers.ops` |
| `download_managed_file` | moved-lazy | `hermes_cli.web_routers.files` |
| `enable_webhooks` | moved-lazy | `hermes_cli.web_routers.ops` |
| `env_var_enabled` | moved-lazy | `utils` |
| `events_ws` | moved-lazy | `hermes_cli.web_routers.chat_ws` |
| `export_session_endpoint` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `field_validator` | unrestorable | `no top-level definition on BASE` |
| `find_provider_entry` | moved-lazy | `hermes_cli.config` |
| `format_docker_update_message` | moved-lazy | `hermes_cli.config` |
| `fs_default_cwd` | moved-lazy | `hermes_cli.web_routers.files` |
| `fs_download` | moved-lazy | `hermes_cli.web_routers.files` |
| `fs_git_root` | moved-lazy | `hermes_cli.web_routers.files` |
| `fs_list` | moved-lazy | `hermes_cli.web_routers.files` |
| `fs_read_data_url` | moved-lazy | `hermes_cli.web_routers.files` |
| `fs_read_text` | moved-lazy | `hermes_cli.web_routers.files` |
| `fs_write_text` | moved-lazy | `hermes_cli.web_routers.files` |
| `functools` | import | `functools` |
| `gateway_drain` | moved-lazy | `hermes_cli.web_routers.actions` |
| `gateway_ws` | moved-lazy | `hermes_cli.web_routers.chat_ws` |
| `get_action_status` | moved-lazy | `hermes_cli.web_routers.actions` |
| `get_active_profile_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `get_auxiliary_models` | moved-lazy | `hermes_cli.web_routers.models` |
| `get_client_voice_config` | moved-lazy | `hermes_cli.web_routers.audio` |
| `get_computer_use_status` | moved-lazy | `hermes_cli.web_routers.tools` |
| `get_config` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `get_config_path` | moved-lazy | `hermes_cli.config` |
| `get_config_raw` | moved-lazy | `hermes_cli.web_routers.analytics` |
| `get_cron_delivery_targets` | moved-lazy | `hermes_cli.web_routers.cron` |
| `get_cron_job` | moved-lazy | `hermes_cli.web_routers.cron` |
| `get_curator_status` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_dashboard_font` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `get_dashboard_plugins` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `get_dashboard_themes` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `get_defaults` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `get_egress_status` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `get_elevenlabs_voices` | moved-lazy | `hermes_cli.web_routers.audio` |
| `get_env_path` | moved-lazy | `hermes_cli.config` |
| `get_env_vars` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `get_health` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_hermes_home` | moved-lazy | `hermes_cli.config` |
| `get_learning_graph` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_learning_node` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_logs` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_media` | moved-lazy | `hermes_cli.web_routers.files` |
| `get_memory_provider_config` | moved-lazy | `hermes_cli.web_routers.memory_providers` |
| `get_memory_status` | moved-lazy | `hermes_cli.web_routers.ops` |
| `get_messaging_platforms` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `get_moa_models` | moved-lazy | `hermes_cli.web_routers.models` |
| `get_model_info` | moved-lazy | `hermes_cli.web_routers.models` |
| `get_model_options` | moved-lazy | `hermes_cli.web_routers.models` |
| `get_models_analytics` | moved-lazy | `hermes_cli.web_routers.analytics` |
| `get_plugins_hub` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `get_portal_status` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_process_hermes_home` | moved-lazy | `hermes_cli.config` |
| `get_profile_setup_command` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `get_profile_soul` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `get_profiles_sessions` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `get_profiles_sessions_sidebar` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `get_provider_config_schema` | moved-lazy | `plugins.memory.config_schema` |
| `get_recommended_default_model` | moved-lazy | `hermes_cli.web_routers.models` |
| `get_running_pid` | moved-lazy | `gateway.status` |
| `get_running_pid_cached` | moved-lazy | `gateway.status` |
| `get_runtime_status_running_pid` | moved-lazy | `gateway.status` |
| `get_schema` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `get_session_detail` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `get_session_latest_descendant` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `get_session_messages` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `get_session_stats` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `get_sessions` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `get_skill_content` | moved-lazy | `hermes_cli.web_routers.skills` |
| `get_skills` | moved-lazy | `hermes_cli.web_routers.skills` |
| `get_ssh_ownership` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_status` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_system_stats` | moved-lazy | `hermes_cli.web_routers.status` |
| `get_telegram_onboarding_status` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `get_terminal_backends` | moved-lazy | `hermes_cli.web_routers.tools` |
| `get_toolset_config` | moved-lazy | `hermes_cli.web_routers.tools` |
| `get_toolset_models` | moved-lazy | `hermes_cli.web_routers.tools` |
| `get_toolsets` | moved-lazy | `hermes_cli.web_routers.tools` |
| `get_update_receipt` | moved-lazy | `hermes_cli.web_routers.actions` |
| `get_usage_analytics` | moved-lazy | `hermes_cli.web_routers.analytics` |
| `get_whatsapp_onboarding_status` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `git_base_branches_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_branch_switch_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_branches_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_commit_context_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_commit_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_create_pr_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_file_diff_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_push_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_rev_parse_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_revert_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_review_diff_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_review_list_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_ship_info_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_stage_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_status_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_unstage_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_worktree_add_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_worktree_remove_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `git_worktrees_route` | moved-lazy | `hermes_cli.web_routers.git` |
| `grant_computer_use_permissions` | moved-lazy | `hermes_cli.web_routers.tools` |
| `hashlib` | import | `hashlib` |
| `import_sessions_endpoint` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `importlib` | import | `importlib.util` |
| `inspect` | import | `inspect` |
| `install_mcp_catalog_entry` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `install_skill_hub` | moved-lazy | `hermes_cli.web_routers.skills` |
| `instantiate_blueprint` | moved-lazy | `hermes_cli.web_routers.cron` |
| `ipaddress` | import | `ipaddress` |
| `is_nix_install_method` | moved-lazy | `hermes_cli.config` |
| `json` | import | `json` |
| `list_checkpoints` | moved-lazy | `hermes_cli.web_routers.ops` |
| `list_credential_pool` | moved-lazy | `hermes_cli.web_routers.ops` |
| `list_cron_blueprints` | moved-lazy | `hermes_cli.web_routers.cron` |
| `list_cron_job_runs` | moved-lazy | `hermes_cli.web_routers.cron` |
| `list_cron_jobs` | moved-lazy | `hermes_cli.web_routers.cron` |
| `list_custom_endpoints` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `list_hooks` | moved-lazy | `hermes_cli.web_routers.ops` |
| `list_managed_files` | moved-lazy | `hermes_cli.web_routers.files` |
| `list_mcp_catalog` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `list_mcp_servers` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `list_oauth_providers` | moved-lazy | `hermes_cli.web_routers.oauth` |
| `list_pairing` | moved-lazy | `hermes_cli.web_routers.ops` |
| `list_profiles_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `list_skills_hub_sources` | moved-lazy | `hermes_cli.web_routers.skills` |
| `list_webhooks` | moved-lazy | `hermes_cli.web_routers.ops` |
| `load_env` | moved-lazy | `hermes_cli.config` |
| `math` | import | `math` |
| `mcp_oauth_callback` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `mcp_oauth_flow_status` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `mimetypes` | import | `mimetypes` |
| `normalize_updated_at` | moved-lazy | `gateway.status` |
| `open_profile_terminal_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `parse_active_agents` | moved-lazy | `gateway.status` |
| `pause_cron_job` | moved-lazy | `hermes_cli.web_routers.cron` |
| `poll_oauth_session` | moved-lazy | `hermes_cli.web_routers.oauth` |
| `post_agent_plugin_disable` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `post_agent_plugin_enable` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `post_agent_plugin_install` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `post_agent_plugin_update` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `post_plugin_visibility` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `preview_skill_hub` | moved-lazy | `hermes_cli.web_routers.skills` |
| `prune_checkpoints` | moved-lazy | `hermes_cli.web_routers.ops` |
| `prune_sessions_endpoint` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `pty_ws` | moved-lazy | `hermes_cli.web_routers.chat_ws` |
| `pub_ws` | moved-lazy | `hermes_cli.web_routers.chat_ws` |
| `put_plugin_providers` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `queue` | import | `queue` |
| `read_managed_file` | moved-lazy | `hermes_cli.web_routers.files` |
| `read_raw_config` | moved-lazy | `hermes_cli.config` |
| `read_runtime_status` | moved-lazy | `gateway.status` |
| `recommended_update_command_for_method` | moved-lazy | `hermes_cli.config` |
| `redact_key` | moved-lazy | `hermes_cli.config` |
| `remove_credential_pool_entry` | moved-lazy | `hermes_cli.web_routers.ops` |
| `remove_env_value` | moved-lazy | `hermes_cli.config` |
| `remove_env_var` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `remove_mcp_server` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `rename_profile_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `rename_session_endpoint` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `replace_mcp_servers` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `rescan_dashboard_plugins` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `reset_memory` | moved-lazy | `hermes_cli.web_routers.ops` |
| `resolve_cron_model_drift_defaults` | moved-lazy | `hermes_cli.config` |
| `resolve_gateway_liveness` | moved-lazy | `gateway.status` |
| `restart_gateway` | moved-lazy | `hermes_cli.web_routers.actions` |
| `resume_cron_job` | moved-lazy | `hermes_cli.web_routers.cron` |
| `reveal_env_var` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `revoke_pairing` | moved-lazy | `hermes_cli.web_routers.ops` |
| `run_backup` | moved-lazy | `hermes_cli.web_routers.ops` |
| `run_config_migrate` | moved-lazy | `hermes_cli.web_routers.status` |
| `run_curator` | moved-lazy | `hermes_cli.web_routers.status` |
| `run_debug_share_endpoint` | moved-lazy | `hermes_cli.web_routers.status` |
| `run_doctor` | moved-lazy | `hermes_cli.doctor` |
| `run_dump` | moved-lazy | `hermes_cli.dump` |
| `run_import` | moved-lazy | `hermes_cli.web_routers.ops` |
| `run_import_upload` | moved-lazy | `hermes_cli.web_routers.ops` |
| `run_in_threadpool` | unrestorable | `no top-level definition on BASE` |
| `run_prompt_size` | moved-lazy | `hermes_cli.web_routers.status` |
| `run_security_audit` | moved-lazy | `hermes_cli.web_routers.ops` |
| `run_toolset_post_setup` | moved-lazy | `hermes_cli.web_routers.tools` |
| `save_config` | moved-lazy | `hermes_cli.config` |
| `save_env_value` | moved-lazy | `hermes_cli.config` |
| `save_toolset_env` | moved-lazy | `hermes_cli.web_routers.tools` |
| `scan_skill_hub` | moved-lazy | `hermes_cli.web_routers.skills` |
| `search_sessions` | moved-lazy | `hermes_cli.web_routers.sessions` |
| `search_skills_hub` | moved-lazy | `hermes_cli.web_routers.skills` |
| `select_terminal_backend` | moved-lazy | `hermes_cli.web_routers.tools` |
| `select_toolset_model` | moved-lazy | `hermes_cli.web_routers.tools` |
| `select_toolset_provider` | moved-lazy | `hermes_cli.web_routers.tools` |
| `serve_plugin_asset` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `set_active_profile_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `set_curator_paused` | moved-lazy | `hermes_cli.web_routers.status` |
| `set_dashboard_font` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `set_dashboard_theme` | moved-lazy | `hermes_cli.web_routers.dashboard_ui` |
| `set_env_var` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `set_mcp_server_enabled` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `set_memory_provider` | moved-lazy | `hermes_cli.web_routers.ops` |
| `set_moa_models` | moved-lazy | `hermes_cli.web_routers.models` |
| `set_model_assignment` | moved-lazy | `hermes_cli.web_routers.models` |
| `set_webhook_enabled` | moved-lazy | `hermes_cli.web_routers.ops` |
| `setup_memory_provider` | moved-lazy | `hermes_cli.web_routers.memory_providers` |
| `shlex` | import | `shlex` |
| `shutil` | import | `shutil` |
| `speak_stream_ws` | moved-lazy | `hermes_cli.web_routers.audio` |
| `speak_text` | moved-lazy | `hermes_cli.web_routers.audio` |
| `start_gateway` | moved-lazy | `hermes_cli.web_routers.ops` |
| `start_oauth_login` | moved-lazy | `hermes_cli.web_routers.oauth` |
| `start_telegram_onboarding` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `start_whatsapp_onboarding` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `stat` | import | `stat` |
| `stop_gateway` | moved-lazy | `hermes_cli.web_routers.ops` |
| `stream_managed_file` | moved-lazy | `hermes_cli.web_routers.files` |
| `submit_oauth_code` | moved-lazy | `hermes_cli.web_routers.oauth` |
| `tempfile` | import | `tempfile` |
| `test_mcp_server` | moved-lazy | `hermes_cli.web_routers.mcp` |
| `test_messaging_platform` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `timezone` | import | `datetime` |
| `toggle_skill` | moved-lazy | `hermes_cli.web_routers.skills` |
| `toggle_toolset` | moved-lazy | `hermes_cli.web_routers.tools` |
| `transcribe_audio_upload` | moved-lazy | `hermes_cli.web_routers.audio` |
| `trigger_cron_job` | moved-lazy | `hermes_cli.web_routers.cron` |
| `tts_lease` | moved-lazy | `hermes_cli.web_routers.audio` |
| `uninstall_skill_hub` | moved-lazy | `hermes_cli.web_routers.skills` |
| `update_config` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `update_config_raw` | moved-lazy | `hermes_cli.web_routers.analytics` |
| `update_cron_job` | moved-lazy | `hermes_cli.web_routers.cron` |
| `update_hermes` | moved-lazy | `hermes_cli.web_routers.actions` |
| `update_learning_node` | moved-lazy | `hermes_cli.web_routers.status` |
| `update_memory_provider_config` | moved-lazy | `hermes_cli.web_routers.memory_providers` |
| `update_messaging_platform` | moved-lazy | `hermes_cli.web_routers.messaging` |
| `update_profile_description_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `update_profile_model_endpoint` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `update_profile_soul` | moved-lazy | `hermes_cli.web_routers.profiles` |
| `update_skill_content` | moved-lazy | `hermes_cli.web_routers.skills` |
| `update_skills_hub` | moved-lazy | `hermes_cli.web_routers.skills` |
| `upload_chat_image` | moved-lazy | `hermes_cli.web_routers.files` |
| `upload_managed_file` | moved-lazy | `hermes_cli.web_routers.files` |
| `upload_managed_file_stream` | moved-lazy | `hermes_cli.web_routers.files` |
| `upsert_custom_endpoint` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `validate_custom_endpoint` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `validate_provider_credential` | moved-lazy | `hermes_cli.web_routers.config_env` |
| `windows_detach_flags` | moved-lazy | `hermes_cli._subprocess_compat` |
| `windows_hide_flags` | moved-lazy | `hermes_cli._subprocess_compat` |
| `write_platform_config_field` | moved-lazy | `hermes_cli.config` |
| `yaml` | import | `yaml` |
| `zipfile` | import | `zipfile` |

### `hermes_cli.webhook`

| name | kind | new location |
|---|---|---|
| `atomic_replace` | moved-lazy | `utils` |
| `os` | import | `os` |
| `tempfile` | import | `tempfile` |

### `hermes_cli.win_pty_bridge`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `hermes_logging`

| name | kind | new location |
|---|---|---|
| `rotating_file_handlers` | restored-def | `(deleted; BASE body restored)` |

### `hermes_state`

| name | kind | new location |
|---|---|---|
| `AUTO_VACUUM_MIN_FREELIST_RATIO` | moved-lazy | `hermes_state_common` |
| `ActivityProvenance` | moved-lazy | `agent.session_activity` |
| `CompressionSessionBusyError` | moved-lazy | `hermes_state_errors` |
| `CompressionSessionClosedError` | moved-lazy | `hermes_state_errors` |
| `DEFERRED_INDEX_SQL` | moved-lazy | `hermes_state_common` |
| `FTS_CJK_STALE_KEY` | moved-lazy | `hermes_state_common` |
| `FTS_CJK_TABLE_SQL` | moved-lazy | `hermes_state_fts` |
| `FTS_CJK_TRIGGER_SQL` | moved-lazy | `hermes_state_fts` |
| `FTS_REBUILD_DEFERRAL_KEY` | moved-lazy | `hermes_state_common` |
| `FTS_SQL` | moved-lazy | `hermes_state_common` |
| `FTS_STALE_KEY` | moved-lazy | `hermes_state_common` |
| `FTS_STORAGE_VERSION` | moved-lazy | `hermes_state_common` |
| `FTS_TRIGRAM_SQL` | moved-lazy | `hermes_state_common` |
| `LEGACY_FTS_SQL` | moved-lazy | `hermes_state_common` |
| `LEGACY_FTS_TRIGRAM_SQL` | moved-lazy | `hermes_state_common` |
| `MAX_FTS5_QUERY_CHARS` | moved-lazy | `hermes_state_common` |
| `MAX_SAFE_EXPORT_MESSAGES` | restored-def | `(deleted; BASE body restored)` |
| `MAX_SAFE_RESUME_MESSAGES` | restored-def | `(deleted; BASE body restored)` |
| `PERSISTENCE_ERROR_CAUSES` | moved-lazy | `hermes_state_errors` |
| `SCHEMA_SQL` | moved-lazy | `hermes_state_common` |
| `SCHEMA_VERSION` | moved-lazy | `hermes_state_common` |
| `SESSION_STATUS_COMPLETE` | moved-lazy | `hermes_state_sessions` |
| `SESSION_STATUS_EMPTY` | moved-lazy | `hermes_state_sessions` |
| `SESSION_STATUS_ERROR` | moved-lazy | `hermes_state_sessions` |
| `SESSION_STATUS_INTERRUPTED` | moved-lazy | `hermes_state_sessions` |
| `SKILL_EXCERPT_JOINT` | moved-lazy | `agent.skill_commands` |
| `SKILL_SCAFFOLD_SQL_LIKE` | moved-lazy | `agent.skill_commands` |
| `SessionTurnLeaseLostError` | moved-lazy | `hermes_state_errors` |
| `Set` | import | `typing` |
| `WalUnsupportedError` | moved-lazy | `hermes_state_wal` |
| `apply_durability_barriers` | moved-lazy | `hermes_state_repair` |
| `classify_session_status` | moved-lazy | `hermes_state_sessions` |
| `close_shared_session_dbs` | unrestorable | `no top-level definition on BASE` |
| `collect_state_db_stats` | moved-lazy | `hermes_state_dbfile` |
| `contextlib` | import | `contextlib` |
| `count_db_holders` | moved-lazy | `hermes_state_dbfile` |
| `describe_skill_invocation` | moved-lazy | `agent.skill_commands` |
| `errno` | import | `errno` |
| `fts5_cjk_so_path` | moved-lazy | `hermes_state_fts` |
| `get_shared_session_db` | unrestorable | `no top-level definition on BASE` |
| `is_advisory_lock_contention` | moved-lazy | `hermes_state_common` |
| `is_automatic_end_reason` | moved-lazy | `hermes_state_common` |
| `is_disk_full_error` | moved-lazy | `hermes_state_errors` |
| `is_sqlite_wal_reset_vulnerable` | moved-lazy | `hermes_state_wal` |
| `is_transient_sqlite_error` | moved-lazy | `hermes_state_errors` |
| `iter_deleted_sqlite_sidecar_holders` | moved-lazy | `hermes_state_dbfile` |
| `release_or_close` | moved-lazy | `hermes_state_registry` |
| `release_shared_session_db` | unrestorable | `no top-level definition on BASE` |
| `report_startup_progress` | moved-lazy | `hermes_startup_watchdog` |
| `resolve_journal_mode` | moved-lazy | `hermes_state_wal` |
| `resolve_synchronous_level` | moved-lazy | `hermes_state_wal` |
| `sanitize_context` | moved-lazy | `agent.memory_manager` |
| `sqlite_source_id` | moved-lazy | `hermes_state_wal` |
| `struct` | import | `struct` |
| `weakref` | import | `weakref` |
| `workspace_key` | moved-lazy | `hermes_state_sessions` |

### `hermes_state_registry`

| name | kind | new location |
|---|---|---|
| `close_shared_session_dbs` | restored-def | `(deleted; BASE body restored)` |
| `get_shared_session_db` | restored-def | `(deleted; BASE body restored)` |
| `release_shared_session_db` | restored-def | `(deleted; BASE body restored)` |

### `hermes_state_search`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |
| `os` | import | `os` |

### `plugins.browser.browser_use.provider`

| name | kind | new location |
|---|---|---|
| `BrowserProvider` | moved-lazy | `agent.browser_provider` |
| `os` | import | `os` |

### `plugins.browser.browserbase.provider`

| name | kind | new location |
|---|---|---|
| `BrowserProvider` | moved-lazy | `agent.browser_provider` |
| `requests` | import | `requests` |
| `uuid` | import | `uuid` |

### `plugins.browser.firecrawl.provider`

| name | kind | new location |
|---|---|---|
| `BrowserProvider` | moved-lazy | `agent.browser_provider` |
| `requests` | import | `requests` |
| `uuid` | import | `uuid` |

### `plugins.context_engine`

| name | kind | new location |
|---|---|---|
| `importlib` | import | `importlib.util` |
| `sys` | import | `sys` |

### `plugins.cron_providers`

| name | kind | new location |
|---|---|---|
| `importlib` | import | `importlib.util` |
| `sys` | import | `sys` |

### `plugins.cron_providers.chronos`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |

### `plugins.cron_providers.chronos._nas_client`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |

### `plugins.dashboard_auth.basic`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |
| `LoginStart` | moved-lazy | `hermes_cli.dashboard_auth` |

### `plugins.dashboard_auth.drain`

| name | kind | new location |
|---|---|---|
| `LoginStart` | moved-lazy | `hermes_cli.dashboard_auth` |

### `plugins.dashboard_auth.nous`

| name | kind | new location |
|---|---|---|
| `DashboardAuthProvider` | moved-lazy | `hermes_cli.dashboard_auth` |
| `InvalidCodeError` | moved-lazy | `hermes_cli.dashboard_auth` |
| `RefreshExpiredError` | moved-lazy | `hermes_cli.dashboard_auth` |
| `base64` | import | `base64` |
| `classify_jwks_lookup_error` | moved-lazy | `hermes_cli.dashboard_auth` |
| `hashlib` | import | `hashlib` |
| `httpx` | import | `httpx` |
| `os` | import | `os` |
| `secrets` | import | `secrets` |
| `urllib` | import | `urllib.parse` |

### `plugins.dashboard_auth.self_hosted`

| name | kind | new location |
|---|---|---|
| `DashboardAuthProvider` | moved-lazy | `hermes_cli.dashboard_auth` |
| `InvalidCodeError` | moved-lazy | `hermes_cli.dashboard_auth` |
| `RefreshExpiredError` | moved-lazy | `hermes_cli.dashboard_auth` |
| `classify_jwks_lookup_error` | moved-lazy | `hermes_cli.dashboard_auth` |
| `hashlib` | import | `hashlib` |
| `os` | import | `os` |
| `secrets` | import | `secrets` |

### `plugins.google_meet.audio_bridge`

| name | kind | new location |
|---|---|---|
| `chrome_fake_audio_flags` | restored-def | `(deleted; BASE body restored)` |

### `plugins.google_meet.meet_bot`

| name | kind | new location |
|---|---|---|
| `SAY_PCM_FILENAME` | restored-def | `(deleted; BASE body restored)` |
| `SAY_QUEUE_FILENAME` | restored-def | `(deleted; BASE body restored)` |
| `json` | import | `json` |

### `plugins.google_meet.node.cli`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |

### `plugins.google_meet.node.registry`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |

### `plugins.image_gen.deepinfra`

| name | kind | new location |
|---|---|---|
| `ImageGenProvider` | moved-lazy | `agent.image_gen_provider` |
| `error_response` | moved-lazy | `agent.image_gen_provider` |
| `save_b64_image` | moved-lazy | `agent.image_gen_provider` |
| `save_url_image` | moved-lazy | `agent.image_gen_provider` |

### `plugins.image_gen.fal`

| name | kind | new location |
|---|---|---|
| `ImageGenProvider` | moved-lazy | `agent.image_gen_provider` |
| `os` | import | `os` |

### `plugins.image_gen.krea`

| name | kind | new location |
|---|---|---|
| `ImageGenProvider` | moved-lazy | `agent.image_gen_provider` |
| `error_response` | moved-lazy | `agent.image_gen_provider` |
| `normalize_reference_images` | moved-lazy | `agent.image_gen_provider` |
| `os` | import | `os` |

### `plugins.image_gen.openai`

| name | kind | new location |
|---|---|---|
| `ImageGenProvider` | moved-lazy | `agent.image_gen_provider` |
| `error_response` | moved-lazy | `agent.image_gen_provider` |
| `normalize_reference_images` | moved-lazy | `agent.image_gen_provider` |
| `save_b64_image` | moved-lazy | `agent.image_gen_provider` |
| `save_url_image` | moved-lazy | `agent.image_gen_provider` |

### `plugins.image_gen.xai`

| name | kind | new location |
|---|---|---|
| `ImageGenProvider` | moved-lazy | `agent.image_gen_provider` |
| `error_response` | moved-lazy | `agent.image_gen_provider` |
| `normalize_reference_images` | moved-lazy | `agent.image_gen_provider` |
| `save_b64_image` | moved-lazy | `agent.image_gen_provider` |
| `save_url_image` | moved-lazy | `agent.image_gen_provider` |

### `plugins.memory.hindsight`

| name | kind | new location |
|---|---|---|
| `dataclass` | import | `dataclasses` |
| `importlib` | import | `importlib` |

### `plugins.memory.honcho`

| name | kind | new location |
|---|---|---|
| `CONCLUDE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `CONTEXT_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `PROFILE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `REASONING_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `SEARCH_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `TRIVIAL_PROMPT_RE` | moved-lazy | `agent.memory_provider` |

### `plugins.memory.honcho.client`

| name | kind | new location |
|---|---|---|
| `SingletonSlot` | moved-lazy | `plugins.plugin_utils` |

### `plugins.memory.honcho.oauth`

| name | kind | new location |
|---|---|---|
| `Callable` | import | `typing` |

### `plugins.memory.honcho.session`

| name | kind | new location |
|---|---|---|
| `Callable` | import | `typing` |
| `Path` | import | `pathlib` |
| `hashlib` | import | `hashlib` |
| `re` | import | `re` |

### `plugins.memory.mem0`

| name | kind | new location |
|---|---|---|
| `ADD_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `DELETE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `SEARCH_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `UPDATE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |

### `plugins.memory.mem0._setup`

| name | kind | new location |
|---|---|---|
| `has_oss_flags` | restored-def | `(deleted; BASE body restored)` |

### `plugins.memory.retaindb`

| name | kind | new location |
|---|---|---|
| `CONTEXT_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `Dict` | import | `typing` |
| `FILE_DELETE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `FILE_INGEST_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `FILE_LIST_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `FILE_READ_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `FILE_UPLOAD_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `FORGET_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `List` | import | `typing` |
| `PROFILE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `REMEMBER_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `SEARCH_SCHEMA` | restored-def | `(deleted; BASE body restored)` |

### `plugins.memory.supermemory`

| name | kind | new location |
|---|---|---|
| `FORGET_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `PROFILE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `SEARCH_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `STORE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |

### `plugins.platforms.a2a.protocol`

| name | kind | new location |
|---|---|---|
| `ERR_PUSH_NOT_SUPPORTED` | restored-def | `(deleted; BASE body restored)` |
| `STATE_AUTH_REQUIRED` | restored-def | `(deleted; BASE body restored)` |
| `copy` | import | `copy` |
| `data_part` | restored-def | `(deleted; BASE body restored)` |
| `file_part` | restored-def | `(deleted; BASE body restored)` |
| `message_with_parts` | restored-def | `(deleted; BASE body restored)` |
| `stream_message` | restored-def | `(deleted; BASE body restored)` |

### `plugins.platforms.a2a.security`

| name | kind | new location |
|---|---|---|
| `Path` | import | `pathlib` |
| `authenticate` | restored-def | `(deleted; BASE body restored)` |
| `get_bearer_token` | restored-def | `(deleted; BASE body restored)` |
| `get_peer_tokens` | restored-def | `(deleted; BASE body restored)` |
| `get_push_secret` | restored-def | `(deleted; BASE body restored)` |
| `get_trusted_peers` | restored-def | `(deleted; BASE body restored)` |
| `is_trusted_peer` | restored-def | `(deleted; BASE body restored)` |
| `resolve_bind_host` | restored-def | `(deleted; BASE body restored)` |
| `sign_push_payload` | restored-def | `(deleted; BASE body restored)` |

### `plugins.platforms.a2a.tools`

| name | kind | new location |
|---|---|---|
| `TypedDict` | import | `typing` |

### `plugins.platforms.dingtalk.adapter`

| name | kind | new location |
|---|---|---|
| `DINGTALK_TYPE_MAPPING` | moved-lazy | `plugins.platforms.dingtalk.inbound` |
| `EXT_MAP` | restored-def | `(deleted; BASE body restored)` |
| `MessageType` | moved-lazy | `gateway.platforms.base` |

### `plugins.platforms.discord.adapter`

| name | kind | new location |
|---|---|---|
| `env_int` | moved-lazy | `utils` |

### `plugins.platforms.feishu.feishu_comment`

| name | kind | new location |
|---|---|---|
| `add_comment_reaction` | restored-def | `(deleted; BASE body restored)` |
| `delete_comment_reaction` | restored-def | `(deleted; BASE body restored)` |

### `plugins.platforms.google_chat.oauth`

| name | kind | new location |
|---|---|---|
| `atomic_replace` | moved-lazy | `utils` |
| `secrets` | import | `secrets` |
| `subprocess` | import | `subprocess` |

### `plugins.platforms.irc.adapter`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `plugins.platforms.line.adapter`

| name | kind | new location |
|---|---|---|
| `field` | import | `dataclasses` |

### `plugins.platforms.matrix.adapter`

| name | kind | new location |
|---|---|---|
| `MAX_MESSAGE_LENGTH` | restored-def | `(deleted; BASE body restored)` |
| `PaginationDirection` | unrestorable | `no top-level definition on BASE` |
| `SyncToken` | unrestorable | `no top-level definition on BASE` |
| `_MATRIX_CAPABILITIES` | restored-helper | `(deleted; restored as a dependency of get_matrix_capabilities)` |
| `get_matrix_capabilities` | restored-def | `(deleted; BASE body restored)` |

### `plugins.platforms.ntfy.adapter`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `plugins.platforms.photon.adapter`

| name | kind | new location |
|---|---|---|
| `ProcessingOutcome` | moved-lazy | `gateway.platforms.base` |
| `resolve_sidecar_dir` | moved-lazy | `plugins.platforms.photon.sidecar_paths` |

### `plugins.platforms.photon.auth`

| name | kind | new location |
|---|---|---|
| `credential_summary` | restored-def | `(deleted; BASE body restored)` |
| `get_session` | restored-def | `(deleted; BASE body restored)` |

### `plugins.platforms.photon.cli`

| name | kind | new location |
|---|---|---|
| `Path` | import | `pathlib` |
| `resolve_sidecar_dir` | moved-lazy | `plugins.platforms.photon.sidecar_paths` |

### `plugins.platforms.raft.adapter`

| name | kind | new location |
|---|---|---|
| `asyncio` | import | `asyncio` |

### `plugins.platforms.teams.adapter`

| name | kind | new location |
|---|---|---|
| `TeamsSummaryWriter` | moved-lazy | `plugins.platforms.teams.summary_writer` |
| `html` | import | `html` |
| `quote` | import | `urllib.parse` |

### `plugins.platforms.telegram.adapter`

| name | kind | new location |
|---|---|---|
| `atomic_replace` | moved-lazy | `utils` |
| `cache_document_from_bytes` | moved-lazy | `gateway.platforms.base` |
| `threading` | import | `threading` |

### `plugins.platforms.telegram.telegram_ids`

| name | kind | new location |
|---|---|---|
| `telegram_chat_id_key` | restored-def | `(deleted; BASE body restored)` |

### `plugins.platforms.wecom.adapter`

| name | kind | new location |
|---|---|---|
| `ABSOLUTE_MAX_BYTES` | moved-lazy | `plugins.platforms.wecom.media` |
| `APP_CMD_UPLOAD_MEDIA_CHUNK` | moved-lazy | `plugins.platforms.wecom.media` |
| `APP_CMD_UPLOAD_MEDIA_FINISH` | moved-lazy | `plugins.platforms.wecom.media` |
| `APP_CMD_UPLOAD_MEDIA_INIT` | moved-lazy | `plugins.platforms.wecom.media` |
| `FILE_MAX_BYTES` | moved-lazy | `plugins.platforms.wecom.media` |
| `IMAGE_MAX_BYTES` | moved-lazy | `plugins.platforms.wecom.media` |
| `MAX_INTERMEDIATE_FRAMES` | moved-lazy | `plugins.platforms.wecom.streaming` |
| `MAX_UPLOAD_CHUNKS` | moved-lazy | `plugins.platforms.wecom.media` |
| `Path` | import | `pathlib` |
| `ReplyFrame` | moved-lazy | `plugins.platforms.wecom.streaming` |
| `STREAM_EXPIRED_ERRCODE` | moved-lazy | `plugins.platforms.wecom.streaming` |
| `STREAM_REQUEST_EXPIRED_ERRCODE` | moved-lazy | `plugins.platforms.wecom.streaming` |
| `STREAM_VERSION_CONFLICT_ERRCODE` | moved-lazy | `plugins.platforms.wecom.streaming` |
| `UPLOAD_CHUNK_SIZE` | moved-lazy | `plugins.platforms.wecom.media` |
| `VIDEO_MAX_BYTES` | moved-lazy | `plugins.platforms.wecom.media` |
| `VOICE_MAX_BYTES` | moved-lazy | `plugins.platforms.wecom.media` |
| `VOICE_SUPPORTED_MIMES` | moved-lazy | `plugins.platforms.wecom.media` |
| `WeComStreamExpiredError` | moved-lazy | `plugins.platforms.wecom.streaming` |
| `base64` | import | `base64` |
| `cache_document_from_bytes_async` | moved-lazy | `gateway.platforms.base` |
| `cache_image_from_bytes_async` | moved-lazy | `gateway.platforms.base` |
| `dataclass` | import | `dataclasses` |
| `deque` | import | `collections` |
| `hashlib` | import | `hashlib` |
| `mimetypes` | import | `mimetypes` |
| `os` | import | `os` |
| `unquote` | import | `urllib.parse` |
| `urlparse` | import | `urllib.parse` |

### `plugins.spotify`

| name | kind | new location |
|---|---|---|
| `SPOTIFY_ALBUMS_SCHEMA` | moved-lazy | `plugins.spotify.tools` |
| `SPOTIFY_DEVICES_SCHEMA` | moved-lazy | `plugins.spotify.tools` |
| `SPOTIFY_LIBRARY_SCHEMA` | moved-lazy | `plugins.spotify.tools` |
| `SPOTIFY_PLAYBACK_SCHEMA` | moved-lazy | `plugins.spotify.tools` |
| `SPOTIFY_PLAYLISTS_SCHEMA` | moved-lazy | `plugins.spotify.tools` |
| `SPOTIFY_QUEUE_SCHEMA` | moved-lazy | `plugins.spotify.tools` |
| `SPOTIFY_SEARCH_SCHEMA` | moved-lazy | `plugins.spotify.tools` |

### `plugins.spotify.client`

| name | kind | new location |
|---|---|---|
| `compact_json` | restored-def | `(deleted; BASE body restored)` |
| `json` | restored-import | `json` |
| `json` | import | `json` |

### `plugins.spotify.tools`

| name | kind | new location |
|---|---|---|
| `SpotifyAPIError` | moved-lazy | `plugins.spotify.client` |
| `SpotifyAuthRequiredError` | moved-lazy | `plugins.spotify.client` |

### `plugins.teams_pipeline.cli`

| name | kind | new location |
|---|---|---|
| `GraphSubscription` | moved-lazy | `plugins.teams_pipeline.models` |
| `Path` | import | `pathlib` |
| `datetime` | import | `datetime` |
| `timedelta` | import | `datetime` |
| `timezone` | import | `datetime` |

### `plugins.teams_pipeline.meetings`

| name | kind | new location |
|---|---|---|
| `fetch_call_record_artifact` | restored-def | `(deleted; BASE body restored)` |

### `plugins.teams_pipeline.pipeline`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `plugins.teams_pipeline.subscriptions`

| name | kind | new location |
|---|---|---|
| `build_store` | restored-def | `(deleted; BASE body restored)` |
| `resolve_store_path` | restored-helper | `(deleted; restored as a dependency of build_store)` |
| `resolve_store_path` | restored-def | `(deleted; BASE body restored)` |
| `resolve_teams_pipeline_store_path` | restored-import | `plugins.teams_pipeline.store` |
| `resolve_teams_pipeline_store_path` | moved-lazy | `plugins.teams_pipeline.store` |

### `plugins.video_gen.fal`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `plugins.video_gen.xai`

| name | kind | new location |
|---|---|---|
| `_run_xai_video_coroutine` | restored-helper | `(deleted; restored as a dependency of run_xai_video_generation)` |
| `run_xai_video_generation` | restored-def | `(deleted; BASE body restored)` |

### `plugins.web.brave_free.provider`

| name | kind | new location |
|---|---|---|
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |
| `os` | import | `os` |

### `plugins.web.ddgs.provider`

| name | kind | new location |
|---|---|---|
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |

### `plugins.web.exa.provider`

| name | kind | new location |
|---|---|---|
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |
| `os` | import | `os` |

### `plugins.web.firecrawl.provider`

| name | kind | new location |
|---|---|---|
| `NoReturn` | import | `typing` |
| `TYPE_CHECKING` | import | `typing` |
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |
| `os` | import | `os` |

### `plugins.web.keenable.provider`

| name | kind | new location |
|---|---|---|
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |

### `plugins.web.parallel.provider`

| name | kind | new location |
|---|---|---|
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |

### `plugins.web.searxng.provider`

| name | kind | new location |
|---|---|---|
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |
| `os` | import | `os` |

### `plugins.web.tavily.provider`

| name | kind | new location |
|---|---|---|
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |

### `plugins.web.xai.provider`

| name | kind | new location |
|---|---|---|
| `WebSearchProvider` | moved-lazy | `agent.web_search_provider` |

### `providers`

| name | kind | new location |
|---|---|---|
| `OMIT_TEMPERATURE` | moved-lazy | `providers.base` |

### `run_agent`

| name | kind | new location |
|---|---|---|
| `COMPRESSED_SUMMARY_METADATA_KEY` | moved-lazy | `agent.context_compressor` |
| `ContextCompressor` | moved-lazy | `agent.context_compressor` |
| `DEFAULT_AGENT_IDENTITY` | moved-lazy | `agent.prompt_builder` |
| `FailoverReason` | moved-lazy | `agent.error_classifier` |
| `OpenAI` | moved-lazy | `agent.process_bootstrap` |
| `SimpleNamespace` | import | `types` |
| `asyncio` | import | `asyncio` |
| `atomic_json_write` | moved-lazy | `utils` |
| `base64` | import | `base64` |
| `build_context_files_prompt` | moved-lazy | `agent.prompt_builder` |
| `build_environment_hints` | moved-lazy | `agent.prompt_builder` |
| `build_skills_system_prompt` | moved-lazy | `agent.prompt_builder` |
| `check_toolset_requirements` | moved-lazy | `model_tools` |
| `convert_scratchpad_to_think` | moved-lazy | `agent.trajectory` |
| `copy` | import | `copy` |
| `estimate_request_tokens_rough` | moved-lazy | `agent.model_metadata` |
| `file_mutation_result_landed` | moved-lazy | `agent.tool_result_classification` |
| `flatten_message_text` | moved-lazy | `agent.message_content` |
| `get_tool_definitions` | moved-lazy | `model_tools` |
| `handle_function_call` | moved-lazy | `model_tools` |
| `hashlib` | import | `hashlib` |
| `is_truthy_value` | moved-lazy | `utils` |
| `jittered_backoff` | moved-lazy | `agent.retry_utils` |
| `load_soul_md` | moved-lazy | `agent.prompt_builder` |
| `normalize_usage` | moved-lazy | `agent.usage_pricing` |
| `redact_sensitive_text` | moved-lazy | `agent.redact` |
| `request_hard_interrupt` | moved-lazy | `agent.interrupt_compat` |
| `sanitize_context` | moved-lazy | `agent.memory_manager` |
| `tempfile` | import | `tempfile` |
| `user_originated_turn_view` | moved-lazy | `agent.context_compressor` |

### `tools`

| name | kind | new location |
|---|---|---|
| `mcp_tool` | unrestorable | `no top-level definition on BASE` |

### `tools.apply_layout_tool`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |

### `tools.approval`

| name | kind | new location |
|---|---|---|
| `DANGEROUS_PATTERNS` | moved-lazy | `tools.approval_detection` |
| `DANGEROUS_PATTERNS_COMPILED` | moved-lazy | `tools.approval_detection` |
| `HARDLINE_PATTERNS` | moved-lazy | `tools.approval_detection` |
| `HARDLINE_PATTERNS_COMPILED` | moved-lazy | `tools.approval_detection` |
| `HUMAN_WAIT_MARGIN_S` | moved-lazy | `tools.approval_human_wait` |
| `cfg_get` | moved-lazy | `hermes_cli.config` |
| `contextlib` | import | `contextlib` |
| `contextvars` | import | `contextvars` |
| `fnmatch` | import | `fnmatch` |
| `functools` | import | `functools` |
| `get_plugin_manager` | moved-lazy | `tools.approval_prompt` |
| `human_wait_ceiling` | moved-lazy | `tools.approval_human_wait` |
| `human_wait_seconds` | moved-lazy | `tools.approval_human_wait` |
| `human_wait_window` | moved-lazy | `tools.approval_human_wait` |
| `is_interrupted` | moved-lazy | `tools.interrupt` |
| `re` | import | `re` |
| `request_elicitation_consent` | moved-lazy | `tools.approval_prompt` |
| `reset_current_observability_context` | moved-lazy | `tools.approval_context` |
| `reset_current_session_key` | moved-lazy | `tools.approval_context` |
| `reset_hermes_interactive_context` | moved-lazy | `tools.approval_context` |
| `set_current_observability_context` | moved-lazy | `tools.approval_context` |
| `set_current_session_key` | moved-lazy | `tools.approval_context` |
| `set_hermes_interactive_context` | moved-lazy | `tools.approval_context` |
| `shlex` | import | `shlex` |
| `sys` | import | `sys` |
| `tempfile` | import | `tempfile` |
| `time` | import | `time` |
| `unicodedata` | import | `unicodedata` |
| `uuid` | import | `uuid` |

### `tools.async_delegation`

| name | kind | new location |
|---|---|---|
| `active_for_session` | restored-def | `(deleted; BASE body restored)` |

### `tools.browser_camofox_state`

| name | kind | new location |
|---|---|---|
| `CAMOFOX_STATE_DIR_NAME` | restored-def | `(deleted; BASE body restored)` |
| `CAMOFOX_STATE_SUBDIR` | restored-def | `(deleted; BASE body restored)` |

### `tools.browser_dialog_tool`

| name | kind | new location |
|---|---|---|
| `logger` | moved-lazy | `tools.approval` |
| `logging` | import | `logging` |

### `tools.browser_supervisor`

| name | kind | new location |
|---|---|---|
| `CONSOLE_HISTORY_MAX` | restored-def | `(deleted; BASE body restored)` |
| `ConsoleEvent` | restored-def | `(deleted; BASE body restored)` |
| `DIALOG_BRIDGE_HOST` | moved-lazy | `tools.browser_supervisor_dialogs` |
| `DIALOG_BRIDGE_URL_PATTERN` | moved-lazy | `tools.browser_supervisor_dialogs` |
| `DIALOG_POLICY_AUTO_ACCEPT` | moved-lazy | `tools.browser_supervisor_dialogs` |
| `DIALOG_POLICY_AUTO_DISMISS` | moved-lazy | `tools.browser_supervisor_dialogs` |
| `DIALOG_POLICY_MUST_RESPOND` | moved-lazy | `tools.browser_supervisor_dialogs` |
| `FRAME_TREE_MAX_ENTRIES` | moved-lazy | `tools.browser_supervisor_frames` |
| `FRAME_TREE_MAX_OOPIF_DEPTH` | moved-lazy | `tools.browser_supervisor_frames` |

### `tools.browser_tool`

| name | kind | new location |
|---|---|---|
| `BrowserUseProvider` | moved-lazy | `plugins.browser.browser_use.provider.BrowserUseBrowserProvider` |
| `BrowserbaseProvider` | moved-lazy | `plugins.browser.browserbase.provider.BrowserbaseBrowserProvider` |
| `CloudBrowserProvider` | moved-lazy | `agent.browser_provider.BrowserProvider` |
| `FirecrawlProvider` | moved-lazy | `plugins.browser.firecrawl.provider.FirecrawlBrowserProvider` |
| `List` | import | `typing` |
| `SNAPSHOT_SUMMARIZE_THRESHOLD` | restored-def | `(deleted; BASE body restored)` |
| `Tuple` | import | `typing` |
| `agent_browser_runnable` | moved-lazy | `hermes_constants` |
| `check_browser_requirements` | moved-lazy | `tools.browser_tool_install` |
| `check_browser_vision_requirements` | moved-lazy | `tools.browser_tool_install` |
| `cleanup_all_browsers` | moved-lazy | `tools.browser_tool_lifecycle` |
| `cleanup_browser` | moved-lazy | `tools.browser_tool_lifecycle` |
| `contextlib` | import | `contextlib` |
| `datetime` | import | `datetime` |
| `functools` | import | `functools` |
| `get_hermes_home_override` | moved-lazy | `hermes_constants` |
| `hermes_home_key` | moved-lazy | `hermes_constants` |
| `is_truthy_value` | moved-lazy | `utils` |
| `lightpanda_engine_status` | moved-lazy | `tools.browser_tool_lightpanda_fallback` |
| `node_tool_runnable` | moved-lazy | `hermes_constants` |
| `normalize_browser_cloud_provider` | moved-lazy | `tools.tool_backend_helpers` |
| `re` | import | `re` |
| `reset_hermes_home_override` | moved-lazy | `hermes_constants` |
| `set_hermes_home_override` | moved-lazy | `hermes_constants` |
| `shutil` | import | `shutil` |
| `signal` | import | `signal` |
| `timezone` | import | `datetime` |
| `warm_agent_browser_npx_cache` | moved-lazy | `tools.browser_tool_install` |
| `windows_hide_flags` | moved-lazy | `hermes_cli._subprocess_compat` |

### `tools.clarify_gateway`

| name | kind | new location |
|---|---|---|
| `get_notify` | restored-def | `(deleted; BASE body restored)` |
| `register_notify` | restored-def | `(deleted; BASE body restored)` |
| `unregister_notify` | restored-def | `(deleted; BASE body restored)` |

### `tools.close_preview_tool`

| name | kind | new location |
|---|---|---|
| `CLOSE_PREVIEW_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `json` | import | `json` |
| `registry` | moved-lazy | `tools.registry` |
| `tool_error` | moved-lazy | `tools.registry` |

### `tools.code_execution_tool`

| name | kind | new location |
|---|---|---|
| `DEFAULT_KERNEL_MODE` | restored-def | `(deleted; BASE body restored)` |
| `KERNEL_MODES` | restored-def | `(deleted; BASE body restored)` |
| `platform` | import | `platform` |
| `socket` | import | `socket` |
| `sys` | import | `sys` |
| `thread_scoped_silence` | moved-lazy | `agent.thread_scoped_output` |

### `tools.code_kernel_remote`

| name | kind | new location |
|---|---|---|
| `base64` | import | `base64` |

### `tools.computer_use`

| name | kind | new location |
|---|---|---|
| `annotations` | import | `__future__` |
| `check_computer_use_requirements` | moved-lazy | `tools.computer_use.tool` |
| `get_computer_use_schema` | moved-lazy | `tools.computer_use.tool` |
| `handle_computer_use` | moved-lazy | `tools.computer_use.tool` |
| `release_computer_use_session` | moved-lazy | `tools.computer_use.tool` |
| `set_approval_callback` | moved-lazy | `tools.computer_use.tool` |

### `tools.computer_use.cua_backend`

| name | kind | new location |
|---|---|---|
| `CaptureResult` | moved-lazy | `tools.computer_use.backend` |
| `PureWindowsPath` | import | `pathlib` |
| `Tuple` | import | `typing` |
| `UIElement` | moved-lazy | `tools.computer_use.backend` |
| `asyncio` | import | `asyncio` |
| `base64` | import | `base64` |
| `concurrent` | import | `concurrent.futures` |
| `cua_driver_install_hint` | moved-lazy | `tools.computer_use.cua_backend_driver` |
| `cua_driver_update_check` | moved-lazy | `tools.computer_use.cua_backend_driver` |
| `deque` | import | `collections` |
| `functools` | import | `functools` |
| `json` | import | `json` |
| `re` | import | `re` |
| `shutil` | import | `shutil` |
| `tempfile` | import | `tempfile` |
| `time` | import | `time` |

### `tools.computer_use.permissions`

| name | kind | new location |
|---|---|---|
| `List` | import | `typing` |

### `tools.computer_use.tool`

| name | kind | new location |
|---|---|---|
| `struct` | import | `struct` |

### `tools.cronjob_tools`

| name | kind | new location |
|---|---|---|
| `effective_job_state` | moved-lazy | `cron.jobs` |
| `re` | import | `re` |

### `tools.delegate_tool`

| name | kind | new location |
|---|---|---|
| `DEFAULT_CHILD_TIMEOUT` | moved-lazy | `tools.delegate_tool_config` |
| `DEFAULT_MAX_SUMMARY_CHARS` | moved-lazy | `tools.delegate_tool_results` |
| `DEFAULT_TOOLSETS` | moved-lazy | `tools.delegate_tool_toolsets` |
| `FuturesTimeoutError` | import | `concurrent.futures` |
| `MAX_DEPTH` | moved-lazy | `tools.delegate_tool_config` |
| `TOOLSETS` | moved-lazy | `toolsets` |
| `base_url_hostname` | moved-lazy | `utils` |
| `contextvars` | import | `contextvars` |
| `enum` | import | `enum` |
| `file_state` | moved-lazy | `tools` |
| `json` | import | `json` |
| `os` | import | `os` |
| `re` | import | `re` |
| `request_hard_interrupt` | moved-lazy | `agent.interrupt_compat` |
| `threading` | import | `threading` |
| `urlsplit` | import | `urllib.parse` |
| `urlunsplit` | import | `urllib.parse` |

### `tools.delegation_live_log`

| name | kind | new location |
|---|---|---|
| `new_live_delegation_id` | restored-def | `(deleted; BASE body restored)` |

### `tools.delegation_output_schema`

| name | kind | new location |
|---|---|---|
| `MAX_SCHEMA_RETRIES` | restored-def | `(deleted; BASE body restored)` |

### `tools.discord_tool`

| name | kind | new location |
|---|---|---|
| `TYPE_CHECKING` | import | `typing` |
| `Tuple` | import | `typing` |
| `get_dynamic_schema` | restored-def | `(deleted; BASE body restored)` |

### `tools.drive_preview_tool`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |

### `tools.env_probe`

| name | kind | new location |
|---|---|---|
| `os` | import | `os` |

### `tools.environments`

| name | kind | new location |
|---|---|---|
| `modal_utils` | unrestorable | `no top-level definition on BASE` |

### `tools.environments.base`

| name | kind | new location |
|---|---|---|
| `IO` | import | `typing` |
| `Protocol` | import | `typing` |
| `codecs` | import | `codecs` |
| `deque` | import | `collections` |
| `re` | import | `re` |
| `sanitize_task_id_for_path` | moved-lazy | `tools.environments.path_utils` |
| `select` | import | `select` |
| `subprocess` | import | `subprocess` |
| `windows_hide_flags` | moved-lazy | `hermes_cli._subprocess_compat` |

### `tools.environments.managed_modal`

| name | kind | new location |
|---|---|---|
| `BaseModalExecutionEnvironment` | unrestorable | `no top-level definition on BASE` |
| `ModalExecStart` | unrestorable | `no top-level definition on BASE` |
| `PreparedModalExec` | unrestorable | `no top-level definition on BASE` |
| `dataclass` | import | `dataclasses` |

### `tools.environments.modal_utils`

| name | kind | new location |
|---|---|---|
| `*` | module-stub | `(deleted)` |

### `tools.environments.vercel_sandbox`

| name | kind | new location |
|---|---|---|
| `dataclass` | import | `dataclasses` |

### `tools.feishu_doc_tool`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |
| `logger` | moved-lazy | `tools.approval` |
| `logging` | import | `logging` |
| `threading` | import | `threading` |

### `tools.feishu_drive_tool`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |
| `threading` | import | `threading` |

### `tools.file_operations`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |
| `ClassVar` | import | `typing` |
| `DEFAULT_READ_LIMIT` | moved-lazy | `tools.file_operations_common` |
| `DEFAULT_READ_OFFSET` | moved-lazy | `tools.file_operations_common` |
| `DEFAULT_SEARCH_LIMIT` | moved-lazy | `tools.file_operations_common` |
| `DEFAULT_SEARCH_OFFSET` | moved-lazy | `tools.file_operations_common` |
| `LINTERS` | moved-lazy | `tools.file_operations_lint` |
| `LintResult` | moved-lazy | `tools.file_operations_common` |
| `List` | import | `typing` |
| `MAX_FILE_SIZE` | moved-lazy | `tools.transcription_common` |
| `MAX_LINES` | restored-def | `(deleted; BASE body restored)` |
| `MAX_LINE_LENGTH` | restored-def | `(deleted; BASE body restored)` |
| `SEARCH_PRUNE_DIR_NAMES` | moved-lazy | `agent.search_policy` |
| `SearchMatch` | moved-lazy | `tools.file_operations_common` |
| `WRITE_DENIED_PATHS` | restored-def | `(deleted; BASE body restored)` |
| `WRITE_DENIED_PREFIXES` | restored-def | `(deleted; BASE body restored)` |
| `build_write_denied_paths` | restored-import | `agent.file_safety` |
| `build_write_denied_paths` | moved-lazy | `agent.file_safety` |
| `build_write_denied_prefixes` | restored-import | `agent.file_safety` |
| `build_write_denied_prefixes` | moved-lazy | `agent.file_safety` |
| `dataclass` | import | `dataclasses` |
| `field` | import | `dataclasses` |
| `posixpath` | import | `posixpath` |
| `threading` | import | `threading` |
| `tool_interrupt` | moved-lazy | `tools.interrupt` |

### `tools.file_tools`

| name | kind | new location |
|---|---|---|
| `PurePosixPath` | import | `pathlib` |
| `has_opaque_document_extension` | moved-lazy | `tools.binary_extensions` |
| `is_pdf_path` | moved-lazy | `tools.binary_extensions` |
| `notify_other_tool_call` | moved-lazy | `tools.file_tools_read_tracking` |
| `posixpath` | import | `posixpath` |
| `reset_file_dedup` | moved-lazy | `tools.file_tools_read_tracking` |
| `sys` | import | `sys` |

### `tools.focus_pane_tool`

| name | kind | new location |
|---|---|---|
| `FOCUS_PANE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `json` | import | `json` |

### `tools.fuzzy_match`

| name | kind | new location |
|---|---|---|
| `List` | import | `typing` |
| `Tuple` | import | `typing` |

### `tools.image_generation_tool`

| name | kind | new location |
|---|---|---|
| `is_krea_model` | restored-def | `(deleted; BASE body restored)` |

### `tools.lazy_deps`

| name | kind | new location |
|---|---|---|
| `feature_specs` | restored-def | `(deleted; BASE body restored)` |

### `tools.managed_tool_gateway`

| name | kind | new location |
|---|---|---|
| `_MANAGED_GATEWAY_VENDOR` | restored-helper | `(deleted; restored as a dependency of build_managed_media_uploader)` |
| `_MEDIA_UPLOAD_PRESIGN_TIMEOUT_SECONDS` | restored-helper | `(deleted; restored as a dependency of build_managed_media_uploader)` |
| `_MEDIA_UPLOAD_PUT_READ_TIMEOUT_SECONDS` | restored-helper | `(deleted; restored as a dependency of build_managed_media_uploader)` |
| `_MEDIA_UPLOAD_PUT_WRITE_TIMEOUT_SECONDS` | restored-helper | `(deleted; restored as a dependency of build_managed_media_uploader)` |
| `_describe_media_upload_refusal` | restored-helper | `(deleted; restored as a dependency of build_managed_media_uploader)` |
| `build_managed_media_uploader` | restored-def | `(deleted; BASE body restored)` |
| `is_managed_nous_gateway_url` | restored-helper | `(deleted; restored as a dependency of build_managed_media_uploader)` |
| `is_managed_nous_gateway_url` | restored-def | `(deleted; BASE body restored)` |
| `managed_gateway_auth_headers` | restored-helper | `(deleted; restored as a dependency of build_managed_media_uploader)` |
| `managed_gateway_auth_headers` | restored-def | `(deleted; BASE body restored)` |
| `managed_vendor_base_path` | restored-def | `(deleted; BASE body restored)` |
| `managed_vendor_endpoints` | restored-def | `(deleted; BASE body restored)` |
| `managed_vendor_upload_path` | restored-helper | `(deleted; restored as a dependency of managed_vendor_endpoints)` |
| `managed_vendor_upload_path` | restored-def | `(deleted; BASE body restored)` |
| `urlsplit` | restored-import | `urllib.parse` |
| `urlsplit` | import | `urllib.parse` |

### `tools.mcp_oauth`

| name | kind | new location |
|---|---|---|
| `OAuthClientInformationFull` | restored-def | `(deleted; BASE body restored)` |
| `OAuthClientMetadata` | restored-def | `(deleted; BASE body restored)` |
| `OAuthClientProvider` | restored-def | `(deleted; BASE body restored)` |
| `OAuthMetadata` | restored-def | `(deleted; BASE body restored)` |
| `OAuthToken` | restored-def | `(deleted; BASE body restored)` |
| `contextmanager` | import | `contextlib` |

### `tools.mcp_schema_cache`

| name | kind | new location |
|---|---|---|
| `clear_cache_entry` | restored-def | `(deleted; BASE body restored)` |
| `has_cached_entry` | restored-def | `(deleted; BASE body restored)` |

### `tools.mcp_tool`

| name | kind | new location |
|---|---|---|
| `Coroutine` | import | `typing` |
| `InvalidMcpUrlError` | moved-lazy | `tools.mcp_tool_errors` |
| `MCP_TOOL_NAME_PREFIX` | moved-lazy | `tools.mcp_tool_schema` |
| `NonMcpEndpointError` | moved-lazy | `tools.mcp_tool_errors` |
| `SimpleNamespace` | import | `types` |
| `Tuple` | import | `typing` |
| `asynccontextmanager` | import | `contextlib` |
| `concurrent` | import | `concurrent.futures` |
| `datetime` | import | `datetime` |
| `discover_mcp_tools` | moved-lazy | `tools.mcp_tool_discovery` |
| `errno` | import | `errno` |
| `fnmatch` | import | `fnmatch` |
| `get_mcp_status` | moved-lazy | `tools.mcp_tool_discovery` |
| `get_registered_mcp_server_names` | moved-lazy | `tools.mcp_tool_discovery` |
| `has_registered_mcp_tools` | moved-lazy | `tools.mcp_tool_discovery` |
| `is_mcp_tool_parallel_safe` | moved-lazy | `tools.mcp_tool_discovery` |
| `json` | import | `json` |
| `matches_name_filter` | moved-lazy | `tools.mcp_tool_schema` |
| `math` | import | `math` |
| `mcp_prefixed_tool_name` | moved-lazy | `tools.mcp_tool_schema` |
| `persist_agent_tool_names` | moved-lazy | `tools.mcp_tool_agent` |
| `probe_mcp_server_tools` | moved-lazy | `tools.mcp_tool_discovery` |
| `random` | import | `random` |
| `re` | import | `re` |
| `reconnect_mcp_server` | moved-lazy | `tools.mcp_tool_loop` |
| `refresh_agent_mcp_tools` | moved-lazy | `tools.mcp_tool_agent` |
| `register_mcp_servers` | moved-lazy | `tools.mcp_tool_discovery` |
| `reprobe_tool_availability` | moved-lazy | `tools.mcp_tool_agent` |
| `restore_agent_tool_prefix` | moved-lazy | `tools.mcp_tool_agent` |
| `sanitize_mcp_name_component` | moved-lazy | `tools.mcp_tool_schema` |
| `shutdown_mcp_servers` | moved-lazy | `tools.mcp_tool_lifecycle` |
| `shutil` | import | `shutil` |
| `strip_unicode_tags` | moved-lazy | `tools.ansi_strip` |
| `tool_error` | moved-lazy | `tools.registry` |
| `urlparse` | import | `urllib.parse` |

### `tools.memory_tool`

| name | kind | new location |
|---|---|---|
| `atomic_write_text` | moved-lazy | `utils` |
| `contextmanager` | import | `contextlib` |
| `time` | import | `time` |

### `tools.microsoft_graph_client`

| name | kind | new location |
|---|---|---|
| `AsyncIterator` | import | `typing` |
| `GraphCredentials` | moved-lazy | `tools.microsoft_graph_auth` |

### `tools.open_preview_tool`

| name | kind | new location |
|---|---|---|
| `OPEN_PREVIEW_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `json` | import | `json` |
| `registry` | moved-lazy | `tools.registry` |

### `tools.openrouter_client`

| name | kind | new location |
|---|---|---|
| `get_async_client` | restored-def | `(deleted; BASE body restored)` |

### `tools.path_security`

| name | kind | new location |
|---|---|---|
| `logger` | moved-lazy | `tools.approval` |
| `logging` | import | `logging` |

### `tools.preview_tool`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |
| `preview_open` | restored-def | `(deleted; BASE body restored)` |

### `tools.process_registry`

| name | kind | new location |
|---|---|---|
| `MAX_ACTIVE_PROCESS_AGE` | restored-def | `(deleted; BASE body restored)` |

### `tools.react_to_message_tool`

| name | kind | new location |
|---|---|---|
| `env_var_enabled` | moved-lazy | `utils` |

### `tools.read_extract`

| name | kind | new location |
|---|---|---|
| `MAX_XLSX_BYTES` | restored-def | `(deleted; BASE body restored)` |

### `tools.read_preview_tool`

| name | kind | new location |
|---|---|---|
| `READ_PREVIEW_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `json` | import | `json` |
| `registry` | moved-lazy | `tools.registry` |
| `tool_error` | moved-lazy | `tools.registry` |

### `tools.read_terminal_tool`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |

### `tools.read_window_tool`

| name | kind | new location |
|---|---|---|
| `json` | import | `json` |
| `tool_error` | moved-lazy | `tools.registry` |

### `tools.send_message_tool`

| name | kind | new location |
|---|---|---|
| `SEND_MESSAGE_SCHEMA` | restored-def | `(deleted; BASE body restored)` |
| `re` | import | `re` |
| `redact_sensitive_text` | moved-lazy | `agent.redact` |
| `time` | import | `time` |

### `tools.skill_ledger`

| name | kind | new location |
|---|---|---|
| `ACTOR_AGENT` | restored-def | `(deleted; BASE body restored)` |
| `ACTOR_CURATOR` | restored-def | `(deleted; BASE body restored)` |
| `ACTOR_USER` | restored-def | `(deleted; BASE body restored)` |

### `tools.skill_linter`

| name | kind | new location |
|---|---|---|
| `format_findings` | restored-def | `(deleted; BASE body restored)` |
| `has_errors` | restored-def | `(deleted; BASE body restored)` |

### `tools.skill_manager_tool`

| name | kind | new location |
|---|---|---|
| `mark_background_review_skill_read` | moved-lazy | `tools.skill_manager_guards` |

### `tools.skill_usage`

| name | kind | new location |
|---|---|---|
| `_suppressed_file` | restored-helper | `(deleted; restored as a dependency of add_suppressed_name)` |
| `_write_suppressed_names` | restored-helper | `(deleted; restored as a dependency of add_suppressed_name)` |
| `add_suppressed_name` | restored-def | `(deleted; BASE body restored)` |
| `agent_created_report` | restored-def | `(deleted; BASE body restored)` |
| `os` | restored-import | `os` |
| `os` | import | `os` |
| `remove_suppressed_name` | restored-def | `(deleted; BASE body restored)` |
| `tempfile` | restored-import | `tempfile` |
| `tempfile` | import | `tempfile` |

### `tools.skillevaluator_scan`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |
| `SCANNER_NAME` | restored-def | `(deleted; BASE body restored)` |
| `scanner_available` | restored-def | `(deleted; BASE body restored)` |

### `tools.skills_guard`

| name | kind | new location |
|---|---|---|
| `full_content_hash` | restored-def | `(deleted; BASE body restored)` |

### `tools.skills_hub`

| name | kind | new location |
|---|---|---|
| `ABC` | import | `abc` |
| `BrowseShSource` | moved-lazy | `tools.skills_hub_sources` |
| `ClawHubSource` | moved-lazy | `tools.skills_hub_clawhub` |
| `GITHUB_TAP_PROVIDERS` | moved-lazy | `tools.skills_hub_github` |
| `GitHubAuth` | moved-lazy | `tools.skills_hub_github` |
| `GitHubSource` | moved-lazy | `tools.skills_hub_github` |
| `HERMES_INDEX_TTL` | moved-lazy | `tools.skills_hub_search` |
| `HERMES_INDEX_URL` | moved-lazy | `tools.skills_hub_search` |
| `HermesIndexSource` | moved-lazy | `tools.skills_hub_official` |
| `LobeHubSource` | moved-lazy | `tools.skills_hub_sources` |
| `OptionalSkillSource` | moved-lazy | `tools.skills_hub_official` |
| `PurePosixPath` | import | `pathlib` |
| `ScanResult` | moved-lazy | `tools.skills_guard` |
| `SkillBundle` | moved-lazy | `tools.skills_hub_models` |
| `SkillMeta` | moved-lazy | `tools.skills_hub_models` |
| `SkillSource` | moved-lazy | `tools.skills_hub_models` |
| `SkillsShSource` | moved-lazy | `tools.skills_hub_skillssh` |
| `TRUSTED_REPOS` | moved-lazy | `tools.skills_guard` |
| `Tuple` | import | `typing` |
| `Union` | import | `typing` |
| `UrlSource` | moved-lazy | `tools.skills_hub_sources` |
| `WellKnownSkillSource` | moved-lazy | `tools.skills_hub_sources` |
| `abstractmethod` | import | `abc` |
| `bundle_content_hash` | moved-lazy | `tools.skills_hub_install` |
| `check_for_skill_updates` | moved-lazy | `tools.skills_hub_install` |
| `content_hash` | moved-lazy | `tools.skills_guard` |
| `create_source_router` | moved-lazy | `tools.skills_hub_search` |
| `dataclass` | import | `dataclasses` |
| `field` | import | `dataclasses` |
| `github_provider_for` | moved-lazy | `tools.skills_hub_github` |
| `hashlib` | import | `hashlib` |
| `install_from_quarantine` | moved-lazy | `tools.skills_hub_install` |
| `is_excluded_skill_path` | moved-lazy | `agent.skill_utils` |
| `os` | import | `os` |
| `parallel_search_sources` | moved-lazy | `tools.skills_hub_search` |
| `quarantine_bundle` | moved-lazy | `tools.skills_hub_install` |
| `quote` | import | `urllib.parse` |
| `re` | import | `re` |
| `shutil` | import | `shutil` |
| `source_url_for_bundle` | moved-lazy | `tools.skills_hub_models` |
| `subprocess` | import | `subprocess` |
| `unified_search` | moved-lazy | `tools.skills_hub_search` |
| `uninstall_skill` | moved-lazy | `tools.skills_hub_install` |
| `unquote` | import | `urllib.parse` |
| `urlparse` | import | `urllib.parse` |
| `urlsplit` | import | `urllib.parse` |
| `urlunparse` | import | `urllib.parse` |
| `windows_hide_flags` | moved-lazy | `hermes_cli._subprocess_compat` |
| `yaml` | import | `yaml` |

### `tools.skills_sync`

| name | kind | new location |
|---|---|---|
| `PurePosixPath` | import | `pathlib` |
| `atomic_replace` | moved-lazy | `utils` |
| `datetime` | import | `datetime` |
| `diff_bundled_skill` | moved-lazy | `tools.skills_sync_bundled_ops` |
| `is_bundled_skills_opt_out` | restored-def | `(deleted; BASE body restored)` |
| `json` | import | `json` |
| `list_user_modified_bundled_skills` | moved-lazy | `tools.skills_sync_bundled_ops` |
| `remove_pristine_bundled_skills` | moved-lazy | `tools.skills_sync_bundled_ops` |
| `reset_bundled_skill` | moved-lazy | `tools.skills_sync_bundled_ops` |
| `restore_official_optional_skill` | moved-lazy | `tools.skills_sync_optional` |
| `set_bundled_skills_opt_out` | moved-lazy | `tools.skills_sync_bundled_ops` |
| `timezone` | import | `datetime` |

### `tools.skills_sync_client`

| name | kind | new location |
|---|---|---|
| `ARTIFACT_TYPE_SKILL` | moved-lazy | `tools.skills_sync_client_wire` |
| `KIND_COMMIT` | restored-def | `(deleted; BASE body restored)` |
| `KIND_TREE` | restored-def | `(deleted; BASE body restored)` |
| `MODE_DIR` | restored-def | `(deleted; BASE body restored)` |
| `MODE_EXEC` | restored-def | `(deleted; BASE body restored)` |
| `MODE_FILE` | restored-def | `(deleted; BASE body restored)` |
| `SYNC_MANIFEST_ENTRY_NAME` | moved-lazy | `tools.skills_sync_client_wire` |
| `SYNC_MANIFEST_TYPE` | restored-def | `(deleted; BASE body restored)` |
| `SYNC_MANIFEST_VERSION` | moved-lazy | `tools.skills_sync_client_wire` |
| `WIRE_VERSION` | moved-lazy | `tools.skills_sync_client_wire` |
| `canonical_json_bytes` | moved-lazy | `tools.skills_sync_client_wire` |
| `datetime` | import | `datetime` |
| `dev_gate_open` | restored-def | `(deleted; BASE body restored)` |
| `hashlib` | import | `hashlib` |
| `maybe_pull_org_skills` | moved-lazy | `tools.skills_sync_client_org` |
| `org_head_ref` | moved-lazy | `tools.skills_sync_client_org` |
| `org_skill_is_locally_modified` | moved-lazy | `tools.skills_sync_client_org` |
| `org_sync_available` | restored-def | `(deleted; BASE body restored)` |
| `parse_sync_manifest` | moved-lazy | `tools.skills_sync_client_wire` |
| `propose_skill` | moved-lazy | `tools.skills_sync_client_org` |
| `pull_org_skills` | moved-lazy | `tools.skills_sync_client_org` |
| `time` | import | `time` |
| `timezone` | import | `datetime` |
| `user_conflict_ref` | restored-def | `(deleted; BASE body restored)` |
| `wire_address` | moved-lazy | `tools.skills_sync_client_wire` |

### `tools.skills_tool`

| name | kind | new location |
|---|---|---|
| `Enum` | import | `enum` |
| `Set` | import | `typing` |
| `display_hermes_home` | moved-lazy | `hermes_constants` |
| `env_var_enabled` | moved-lazy | `utils` |
| `re` | import | `re` |
| `threading` | import | `threading` |

### `tools.slash_confirm`

| name | kind | new location |
|---|---|---|
| `asyncio` | import | `asyncio` |
| `resolve_sync_compat` | restored-def | `(deleted; BASE body restored)` |

### `tools.terminal_scope`

| name | kind | new location |
|---|---|---|
| `install_refusal_scope` | restored-def | `(deleted; BASE body restored)` |
| `terminal_scope` | restored-def | `(deleted; BASE body restored)` |

### `tools.terminal_tool`

| name | kind | new location |
|---|---|---|
| `Path` | import | `pathlib` |
| `cleanup_vm` | moved-lazy | `tools.terminal_tool_lifecycle` |
| `env_var_enabled` | moved-lazy | `utils` |
| `get_active_env` | moved-lazy | `tools.terminal_tool_lifecycle` |
| `has_direct_modal_credentials` | moved-lazy | `tools.tool_backend_helpers` |
| `importlib` | import | `importlib.util` |
| `is_interrupted` | moved-lazy | `tools.interrupt` |
| `is_managed_tool_gateway_ready` | moved-lazy | `tools.managed_tool_gateway` |
| `is_persistent_env` | moved-lazy | `tools.terminal_tool_lifecycle` |
| `nous_tool_gateway_unavailable_message` | moved-lazy | `tools.tool_backend_helpers` |
| `platform` | import | `platform` |
| `re` | import | `re` |
| `resolve_modal_backend_state` | moved-lazy | `tools.tool_backend_helpers` |
| `shlex` | import | `shlex` |
| `shutil` | import | `shutil` |
| `stat` | import | `stat` |
| `strip_inert_heredoc_bodies` | moved-lazy | `tools.shell_heredoc` |
| `subprocess` | import | `subprocess` |
| `sys` | import | `sys` |

### `tools.tool_result_storage`

| name | kind | new location |
|---|---|---|
| `HEREDOC_MARKER` | restored-def | `(deleted; BASE body restored)` |
| `uuid` | import | `uuid` |

### `tools.tool_search`

| name | kind | new location |
|---|---|---|
| `Literal` | import | `typing` |
| `build_catalog_listing` | restored-def | `(deleted; BASE body restored)` |
| `copy` | import | `copy` |
| `field` | import | `dataclasses` |
| `re` | import | `re` |
| `snowballstemmer` | import | `snowballstemmer` |
| `threading` | import | `threading` |

### `tools.transcription_tools`

| name | kind | new location |
|---|---|---|
| `COMMAND_STT_OUTPUT_FORMATS` | moved-lazy | `tools.transcription_command` |
| `COMMON_LOCAL_BIN_DIRS` | moved-lazy | `tools.transcription_common` |
| `DEFAULT_COMMAND_STT_LANGUAGE` | moved-lazy | `tools.transcription_command` |
| `DEFAULT_COMMAND_STT_OUTPUT_FORMAT` | moved-lazy | `tools.transcription_command` |
| `DEFAULT_COMMAND_STT_TIMEOUT_SECONDS` | moved-lazy | `tools.transcription_command` |
| `DEFAULT_LOCAL_STT_LANGUAGE` | moved-lazy | `tools.transcription_common` |
| `ELEVENLABS_STT_BASE_URL` | moved-lazy | `tools.transcription_common` |
| `GROQ_BASE_URL` | moved-lazy | `tools.transcription_common` |
| `GROQ_MODELS` | moved-lazy | `tools.transcription_common` |
| `LOCAL_NATIVE_AUDIO_FORMATS` | moved-lazy | `tools.transcription_common` |
| `MAX_FILE_SIZE` | moved-lazy | `tools.transcription_common` |
| `OPENAI_BASE_URL` | moved-lazy | `tools.transcription_common` |
| `OPENAI_MODELS` | moved-lazy | `tools.transcription_common` |
| `SUPPORTED_FORMATS` | moved-lazy | `tools.transcription_common` |
| `XAI_STT_BASE_URL` | moved-lazy | `tools.transcription_common` |
| `managed_nous_tools_enabled` | moved-lazy | `tools.tool_backend_helpers` |
| `nous_tool_gateway_unavailable_message` | moved-lazy | `tools.tool_backend_helpers` |
| `platform` | import | `platform` |
| `queue` | import | `queue` |
| `re` | import | `re` |
| `resolve_managed_tool_gateway` | moved-lazy | `tools.managed_tool_gateway` |
| `resolve_openai_audio_api_key` | moved-lazy | `tools.tool_backend_helpers` |
| `shlex` | import | `shlex` |
| `subprocess` | import | `subprocess` |
| `tempfile` | import | `tempfile` |
| `urljoin` | import | `urllib.parse` |
| `windows_hide_flags` | moved-lazy | `hermes_cli._subprocess_compat` |

### `tools.tts_tool`

| name | kind | new location |
|---|---|---|
| `AudioDeliveryProfile` | moved-lazy | `tools.tts_tool_delivery` |
| `COMMAND_TTS_OUTPUT_FORMATS` | moved-lazy | `tools.tts_command_provider` |
| `DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH` | moved-lazy | `tools.tts_command_provider` |
| `DEFAULT_COMMAND_TTS_OUTPUT_FORMAT` | moved-lazy | `tools.tts_command_provider` |
| `DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS` | moved-lazy | `tools.tts_command_provider` |
| `DEFAULT_DEEPINFRA_TTS_VOICE` | moved-lazy | `tools.tts_tool_openai` |
| `DEFAULT_EDGE_VOICE` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_ELEVENLABS_MODEL_ID` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_ELEVENLABS_STREAMING_MODEL_ID` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_ELEVENLABS_VOICE_ID` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_GEMINI_AUDIO_TAGS` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_GEMINI_TTS_BASE_URL` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_GEMINI_TTS_MODEL` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_GEMINI_TTS_VOICE` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_KITTENTTS_MODEL` | moved-lazy | `tools.tts_tool_local` |
| `DEFAULT_KITTENTTS_VOICE` | moved-lazy | `tools.tts_tool_local` |
| `DEFAULT_MINIMAX_BASE_URL` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_MINIMAX_CN_BASE_URL` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_MINIMAX_MODEL` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_MINIMAX_VOICE_ID` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_MISTRAL_TTS_MODEL` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_MISTRAL_TTS_VOICE_ID` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_OPENAI_BASE_URL` | moved-lazy | `tools.tts_tool_openai` |
| `DEFAULT_OPENAI_MODEL` | moved-lazy | `tools.tts_tool_openai` |
| `DEFAULT_OPENAI_VOICE` | moved-lazy | `tools.tts_tool_openai` |
| `DEFAULT_PIPER_VOICE` | moved-lazy | `tools.tts_tool_local` |
| `DEFAULT_XAI_AUTO_SPEECH_TAGS` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_BASE_URL` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_BIT_RATE` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_LANGUAGE` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_OPTIMIZE_STREAMING_LATENCY_DEFAULT` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_SAMPLE_RATE` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_SPEED_DEFAULT` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_SPEED_MAX` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_SPEED_MIN` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_TEXT_NORMALIZATION_DEFAULT` | moved-lazy | `tools.tts_tool_providers` |
| `DEFAULT_XAI_VOICE_ID` | moved-lazy | `tools.tts_tool_providers` |
| `ELEVENLABS_MODEL_MAX_TEXT_LENGTH` | moved-lazy | `tools.tts_tool_delivery` |
| `FALLBACK_MAX_TEXT_LENGTH` | moved-lazy | `tools.tts_tool_delivery` |
| `FALLBACK_MAX_TEXT_LENGTH` | restored-helper | `(deleted; restored as a dependency of MAX_TEXT_LENGTH)` |
| `Future` | import | `concurrent.futures` |
| `GEMINI_AUDIO_TAG_REWRITE_TASK` | moved-lazy | `tools.tts_tool_providers` |
| `GEMINI_TTS_CHANNELS` | restored-def | `(deleted; BASE body restored)` |
| `GEMINI_TTS_SAMPLE_RATE` | restored-def | `(deleted; BASE body restored)` |
| `GEMINI_TTS_SAMPLE_WIDTH` | restored-def | `(deleted; BASE body restored)` |
| `Iterator` | import | `typing` |
| `MANAGED_OPENAI_TTS_MODELS` | moved-lazy | `tools.tts_tool_openai` |
| `MAX_TEXT_LENGTH` | restored-def | `(deleted; BASE body restored)` |
| `PROVIDER_MAX_TEXT_LENGTH` | moved-lazy | `tools.tts_tool_delivery` |
| `TTS_RESPONSE_BODY_CHUNK_BYTES` | moved-lazy | `tools.tts_tool_providers` |
| `TTS_RESPONSE_BODY_LIMIT_BYTES` | moved-lazy | `tools.tts_tool_providers` |
| `ThreadPoolExecutor` | import | `concurrent.futures` |
| `Tuple` | import | `typing` |
| `acquire_tts_lease` | moved-lazy | `tools.tts_tool_lifecycle` |
| `base64` | import | `base64` |
| `dataclass` | import | `dataclasses` |
| `field` | import | `dataclasses` |
| `hermes_xai_user_agent` | moved-lazy | `tools.xai_http` |
| `managed_nous_tools_enabled` | moved-lazy | `tools.tool_backend_helpers` |
| `nous_tool_gateway_unavailable_message` | moved-lazy | `tools.tool_backend_helpers` |
| `platform` | import | `platform` |
| `queue` | import | `queue` |
| `re` | import | `re` |
| `read_selection` | moved-lazy | `tools.tool_backend_helpers` |
| `release_tts_lease` | moved-lazy | `tools.tts_tool_lifecycle` |
| `release_tts_provider` | moved-lazy | `tools.tts_tool_lifecycle` |
| `resolve_managed_tool_gateway` | moved-lazy | `tools.managed_tool_gateway` |
| `resolve_openai_audio_api_key` | moved-lazy | `tools.tool_backend_helpers` |
| `selection_error` | moved-lazy | `tools.tool_backend_helpers` |
| `shlex` | import | `shlex` |
| `shutil` | import | `shutil` |
| `stream_tts_to_speaker` | moved-lazy | `tools.tts_tool_speaker` |
| `subprocess` | import | `subprocess` |
| `threading` | import | `threading` |
| `time` | import | `time` |
| `tts_lease_holders` | moved-lazy | `tools.tts_tool_lifecycle` |
| `urljoin` | import | `urllib.parse` |
| `urlparse` | import | `urllib.parse` |
| `uuid` | import | `uuid` |
| `warm_tts_provider` | moved-lazy | `tools.tts_tool_lifecycle` |
| `windows_hide_flags` | moved-lazy | `hermes_cli._subprocess_compat` |

### `tools.url_safety`

| name | kind | new location |
|---|---|---|
| `has_sensitive_query_params` | restored-def | `(deleted; BASE body restored)` |
| `ssrf_safe_async_http_transport` | restored-def | `(deleted; BASE body restored)` |
| `ssrf_safe_http_transport` | restored-def | `(deleted; BASE body restored)` |

### `tools.vision_tools`

| name | kind | new location |
|---|---|---|
| `contextlib` | import | `contextlib` |
| `sys` | import | `sys` |
| `threading` | import | `threading` |

### `tools.voice_mode`

| name | kind | new location |
|---|---|---|
| `DEFAULT_TTS_ECHO_SIMILARITY_THRESHOLD` | moved-lazy | `tools.voice_mode_transcript` |
| `DEFAULT_VOICE_STOP_PHRASES` | moved-lazy | `tools.voice_mode_transcript` |
| `MIN_FRAGMENT_LENGTH_FOR_ECHO` | moved-lazy | `tools.voice_mode_transcript` |
| `WHISPER_HALLUCINATIONS` | restored-def | `(deleted; BASE body restored)` |
| `difflib` | import | `difflib` |
| `is_tts_echo` | moved-lazy | `tools.voice_mode_transcript` |
| `re` | import | `re` |
| `voice_stop_hint` | moved-lazy | `tools.voice_mode_transcript` |

### `tools.web_result_cache`

| name | kind | new location |
|---|---|---|
| `Any` | import | `typing` |
| `List` | import | `typing` |

### `tools.web_tools`

| name | kind | new location |
|---|---|---|
| `DEFAULT_EXTRACT_CHAR_LIMIT` | moved-lazy | `tools.web_tools_truncate` |
| `Dict` | import | `typing` |
| `Firecrawl` | moved-lazy | `plugins.web.firecrawl.provider` |
| `MAX_STORED_TEXT_CHARS` | moved-lazy | `tools.web_tools_truncate` |
| `TYPE_CHECKING` | import | `typing` |
| `asyncio` | import | `asyncio` |
| `build_vendor_gateway_url` | moved-lazy | `tools.managed_tool_gateway` |
| `httpx` | import | `httpx` |
| `managed_nous_tools_enabled` | moved-lazy | `tools.tool_backend_helpers` |
| `normalize_url_for_request` | moved-lazy | `tools.url_safety` |
| `nous_tool_gateway_unavailable_message` | moved-lazy | `tools.tool_backend_helpers` |
| `prefers_gateway` | moved-lazy | `tools.tool_backend_helpers` |
| `re` | import | `re` |
| `resolve_managed_tool_gateway` | moved-lazy | `tools.managed_tool_gateway` |
| `sensitive_query_param_name` | moved-lazy | `tools.url_safety` |
| `sys` | import | `sys` |

### `tools.website_policy`

| name | kind | new location |
|---|---|---|
| `invalidate_cache` | restored-def | `(deleted; BASE body restored)` |

### `tools.write_approval`

| name | kind | new location |
|---|---|---|
| `is_background` | restored-def | `(deleted; BASE body restored)` |

### `tools.yuanbao_tools`

| name | kind | new location |
|---|---|---|
| `List` | import | `typing` |
| `Optional` | import | `typing` |

### `toolsets`

| name | kind | new location |
|---|---|---|
| `resolve_multiple_toolsets` | restored-def | `(deleted; BASE body restored)` |

### `tui_gateway.compute_host`

| name | kind | new location |
|---|---|---|
| `HostSession` | restored-def | `(deleted; BASE body restored)` |
| `SpikeAgent` | restored-helper | `(deleted; restored as a dependency of HostSession)` |
| `SpikeAgent` | restored-def | `(deleted; BASE body restored)` |
| `dataclass` | restored-import | `dataclasses` |
| `dataclass` | import | `dataclasses` |
| `field` | restored-import | `dataclasses` |
| `field` | import | `dataclasses` |
| `request_hard_interrupt` | moved-lazy | `agent.interrupt_compat` |

### `tui_gateway.hosted_room_driver`

| name | kind | new location |
|---|---|---|
| `contextlib` | import | `contextlib` |
| `null_turn_lock` | restored-def | `(deleted; BASE body restored)` |

### `tui_gateway.hosted_room_service`

| name | kind | new location |
|---|---|---|
| `hashlib` | import | `hashlib` |

### `tui_gateway.mcp_rpc_helpers`

| name | kind | new location |
|---|---|---|
| `Optional` | import | `typing` |
| `Tuple` | import | `typing` |
| `resolve_profile` | restored-def | `(deleted; BASE body restored)` |

### `tui_gateway.methods_browser_control`

| name | kind | new location |
|---|---|---|
| `BROWSER_CONTROL_PROTOCOL_VERSION` | moved-lazy | `gateway.browser_control_broker` |
| `browser_control_protocol_supported` | moved-lazy | `gateway.browser_control_broker` |
| `filter_browser_control_capabilities` | moved-lazy | `gateway.browser_control_broker` |

### `tui_gateway.methods_prompt`

| name | kind | new location |
|---|---|---|
| `types` | import | `types` |

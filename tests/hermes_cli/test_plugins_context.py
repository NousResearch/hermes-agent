"""PluginContext lives in hermes_cli/plugins_context.py and stays importable and patchable through
the hermes_cli.plugins facade."""

import logging
import typing

import pytest

import hermes_cli.plugins as plugins_mod
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def test_plugin_context_is_defined_in_its_own_module_and_reexported():
    import hermes_cli.plugins_context as plugin_context_mod

    assert plugin_context_mod.PluginContext is PluginContext
    assert plugin_context_mod._UNSET is plugins_mod._UNSET
    assert plugin_context_mod._ignore_after_abandoned_load is plugins_mod._ignore_after_abandoned_load


def test_ctx_methods_still_read_patches_on_the_facade(monkeypatch):
    config = {"plugins": {"entries": {"demo": {"settings": {"color": "blue"}}}}}
    monkeypatch.setattr(plugins_mod, "load_config_readonly", lambda *a, **k: config)
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    assert ctx.get_config("color") == "blue"


def test_ctx_logs_through_the_facade_logger(monkeypatch, caplog):
    patched = logging.getLogger("tests.plugin_context.facade_logger")
    monkeypatch.setattr(plugins_mod, "logger", patched)
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    with caplog.at_level(logging.WARNING, logger=patched.name):
        ctx.register_middleware("not_a_middleware_kind", lambda *a, **k: None)
    assert [r.name for r in caplog.records if "unknown middleware" in r.getMessage()] == [patched.name]


def test_ctx_leases_through_the_facade_replacement_coordinator(monkeypatch):
    real = plugins_mod.replacement_coordinator
    acquired = []

    class Recording:
        def acquire(self, slot, **kwargs):
            acquired.append(slot[0])
            return real.acquire(slot, **kwargs)

    monkeypatch.setattr(plugins_mod, "replacement_coordinator", Recording())
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    ctx.register_command("demo-cmd", lambda raw: None).dispose()
    assert acquired == ["manager_mapping"]


def test_ctx_reads_the_section_limit_from_the_facade(monkeypatch):
    monkeypatch.setattr(plugins_mod, "MAX_SYSTEM_PROMPT_SECTION_CHARS", 1)
    ctx = PluginContext(PluginManifest(name="demo"), PluginManager())
    with pytest.raises(ValueError, match="between 1 and 1"):
        ctx.register_system_prompt_section("demo.section", "x", max_chars=2)


def test_init_annotations_still_resolve():
    hints = typing.get_type_hints(PluginContext.__init__)
    assert hints == {"manifest": PluginManifest, "manager": PluginManager}


# Every non-dunder name in dir(hermes_cli.plugins) on main ca436069, private helpers included.
# Anything importable from the facade on main must stay importable, so a move can't drop one.
_MAIN_FACADE_NAMES = """
    Any Callable DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS Dict ENTRY_POINTS_GROUP HERMES_EVENT_NAMESPACE
    List LoadedPlugin MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS MAX_SYSTEM_PROMPT_SECTION_CHARS Mapping
    Optional PLUGIN_SECTIONS_END PLUGIN_SECTIONS_START Path PluginCapabilityCtxMixin PluginContext
    PluginDispatchMixin PluginLedgerMixin PluginLoaderMixin PluginManager PluginManifest
    PluginRegistration PluginState PluginSystemPromptSection PluginToolOverrideError
    RELAY_PLUGINS_CONFIG_ENV RenderedPluginSystemPromptSection SHELL_UNSUPPORTED_HOOKS
    SUPPORTED_MANIFEST_VERSION SYSTEM_PROMPT_SECTION_POSITIONS Set Tuple Union VALID_HOOKS
    VALID_MIDDLEWARE _BARE_MODULE_SCOPE _CONFIG_SCHEMA_TYPES _DEBUG_HANDLER_INSTALLED
    _EVENT_EMIT_DEPTH_CAP _EVENT_PENDING_CAP _EventSubscription _HOOK_CALLBACK_TIMEOUT_SECS
    _HOOK_TIMEOUT_SUPPRESSION_SECONDS _MAX_HOOK_CALLBACK_TIMEOUT_SECS _MODULE_NAMESPACE_LOCK _NS_PARENT
    _PLUGINS_DEBUG _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE _PreToolCallDirective
    _SCOPED_PROVIDER_REGISTRARS _UNSET _attach_published_gateway_host _attach_published_tui_host
    _background_discovery_lock _background_discovery_thread _clear_plugin_submodules _delivery_manager
    _dispatch_pre_tool_call_hooks _ensure_plugins_discovered _env_enabled _evict_modules
    _get_disabled_plugins _get_enabled_plugins _get_pre_tool_call_directive_details
    _ignore_after_abandoned_load _install_plugin_debug_handler _join_background_discovery
    _known_plugin_managers _locked_plugin_state _nested_plugin_mapping _nested_plugin_value
    _nowait_plugin_set _persist_plugin_toolset_keys _plugin_home_key _plugin_home_scope _plugin_manager
    _plugin_managers_by_home _plugin_managers_lock _plugin_relative_segments _plugin_settings_entry
    _plugin_toolset_keys_cache_path _portable_data_namespace _published_gateway_host_lock
    _published_gateway_message_injector _published_tui_host_lock _published_tui_message_injector
    _reset_plugin_managers_for_tests _resolve_block_from_details _resolve_hook_callback_timeout
    _serialized_replacement _thread_tool_whitelist ainvoke_hook annotations asyncio cached_property
    clear_published_gateway_message_host clear_published_tui_message_host clear_thread_tool_whitelist
    collect_directory_manifests dataclass discover_entrypoint_manifests discover_plugins env_var_enabled
    field fire_pre_command_hook format_system_prompt_sections gate_manifest get_bundled_plugins_dir
    get_hermes_home get_plugin_auxiliary_tasks get_plugin_command_handler get_plugin_commands
    get_plugin_context_engine get_plugin_error_classification get_plugin_manager
    get_plugin_toolset_keys_nowait get_plugin_toolsets get_portable_mcp_server_names_nowait
    get_pre_tool_call_block_message get_pre_tool_call_directive get_pre_verify_continue_message
    get_process_hermes_home has_enabled_agent_plugin_mcp has_hook has_middleware hermes_home_key
    importlib in_plugin_load_worker invoke_hook invoke_middleware is_valid_system_prompt_section_id
    iter_hook_callbacks json legacy_relay_plugin_keys load_config_readonly logger logging
    make_scoped_provider_registrar manifest_key os parse_manifest_file plugin_capability_granted
    plugin_discovery_suppressed publish_gateway_message_host publish_tui_message_host queue re
    render_system_prompt_sections replacement_coordinator resolve_manifest_winners resolve_module_origin
    resolve_plugin_command_result resolve_plugin_load_order resolve_pre_tool_block save_plugin_setting
    scan_directory set_thread_tool_whitelist start_background_plugin_discovery suppress sys threading
    types validate_config_schema wraps
""".split()


def test_every_name_importable_from_the_facade_on_main_still_is():
    missing = [name for name in _MAIN_FACADE_NAMES if not hasattr(plugins_mod, name)]
    assert missing == []

"""Compatibility facade for the canonical plugin runtime.

First-party code imports runtime owners directly. This module exists only to preserve
documented/external plugin imports while those compatibility paths remain supported.
"""

from __future__ import annotations

from plugin_runtime.loading import LoadedPlugin, PluginLoaderMixin
from plugin_runtime.ownership import PluginOwnershipMixin
from plugin_runtime.registration import PluginRegistration
from plugin_runtime.context import PluginContext, PluginToolOverrideError
from plugin_runtime.manager import PluginManager
from hermes_cli.plugin_policy import (
    _PreToolCallDirective,
    _dispatch_pre_tool_call_hooks,
    _get_pre_tool_call_directive_details,
    _resolve_block_from_details,
    clear_thread_tool_whitelist,
    fire_pre_command_hook,
    get_plugin_error_classification,
    get_pre_tool_call_block_message,
    get_pre_tool_call_directive,
    get_pre_verify_continue_message,
    resolve_pre_tool_block,
    set_thread_tool_whitelist,
)
from plugin_runtime.api import (
    ainvoke_hook,
    get_plugin_auxiliary_tasks,
    get_plugin_command_handler,
    get_plugin_commands,
    get_plugin_context_engine,
    get_plugin_subscriptions,
    get_plugin_toolsets,
    has_hook,
    has_middleware,
    invoke_hook,
    invoke_middleware,
    iter_hook_callbacks,
    render_system_prompt_sections,
    unload_plugins,
)
from plugin_runtime.lifecycle import (
    clear_published_tui_message_host,
    discover_plugins,
    ensure_plugins_discovered as _ensure_plugins_discovered,
    get_plugin_manager,
    get_plugin_toolset_keys_nowait,
    get_portable_mcp_server_names_nowait,
    has_enabled_agent_plugin_mcp,
    publish_tui_message_host,
    start_background_plugin_discovery,
)
from plugin_runtime.discovery import (
    ENTRY_POINTS_GROUP,
    _get_disabled_plugins,
    _get_enabled_plugins,
    collect_directory_manifests,
    discover_entrypoint_manifests,
    gate_manifest,
    get_bundled_plugins_dir,
    resolve_manifest_winners,
    scan_directory,
)
from plugin_runtime.relay_policy import RELAY_PLUGINS_CONFIG_ENV, legacy_relay_plugin_keys
from plugin_runtime.manifest import (
    _CONFIG_SCHEMA_TYPES,
    SUPPORTED_MANIFEST_VERSION,
    PluginManifest,
    _portable_skill_namespace,
    manifest_key,
    parse_manifest_file,
    resolve_module_origin,
    resolve_plugin_load_order,
    validate_config_schema,
)
from plugin_runtime.scope import plugin_home_scope as _plugin_home_scope
from plugin_runtime.dispatch import (
    DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    HERMES_EVENT_NAMESPACE,
    MAX_SYSTEM_PROMPT_SECTION_CHARS,
    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS,
    PLUGIN_SECTIONS_END,
    PLUGIN_SECTIONS_START,
    SHELL_UNSUPPORTED_HOOKS,
    SYSTEM_PROMPT_SECTION_POSITIONS,
    VALID_HOOKS,
    VALID_MIDDLEWARE,
    _EVENT_EMIT_DEPTH_CAP,
    _EVENT_PENDING_CAP,
    _HOOK_CALLBACK_TIMEOUT_SECS,
    _HOOK_TIMEOUT_SUPPRESSION_SECONDS,
    _MAX_HOOK_CALLBACK_TIMEOUT_SECS,
    _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE,
    _resolve_hook_callback_timeout,
    PluginDispatchMixin,
    PluginSystemPromptSection,
    RenderedPluginSystemPromptSection,
    _EventSubscription,
    format_system_prompt_sections,
    is_valid_system_prompt_section_id,
    resolve_plugin_command_result,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
from typing import Iterable, Type  # noqa: F401,E402
from contextlib import contextmanager  # noqa: F401,E402
import contextvars  # noqa: F401,E402
import copy  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import time  # noqa: F401,E402
from functools import wraps  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    "CAPABILITY_REGISTRY": ("plugin_runtime.capabilities", "CAPABILITY_REGISTRY"),
    "ENTRY_POINT_CAPABILITIES_GROUP": ("plugin_runtime.discovery", "ENTRY_POINT_CAPABILITIES_GROUP"),
    "LEGACY_RELAY_PLUGIN_KEYS": ("plugin_runtime.relay_policy", "LEGACY_RELAY_PLUGIN_KEYS"),
    "MAX_SYSTEM_PROMPT_SECTIONS": ("plugin_runtime.dispatch", "MAX_SYSTEM_PROMPT_SECTIONS"),
    "OBSERVER_SCHEMA_VERSION": ("plugin_runtime.dispatch", "OBSERVER_SCHEMA_VERSION"),
    "VALID_CAPABILITY_IDS": ("plugin_runtime.capabilities", "VALID_CAPABILITY_IDS"),
    "cfg_get": ("hermes_cli.config", "cfg_get"),
    "fast_safe_load": ("utils", "fast_safe_load"),
    "format_system_prompt_section": ("plugin_runtime.dispatch", "format_system_prompt_section"),
    "reset_hermes_home_override": ("hermes_constants", "reset_hermes_home_override"),
    "set_hermes_home_override": ("hermes_constants", "set_hermes_home_override"),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    from plugin_runtime.compat import warn_once

    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

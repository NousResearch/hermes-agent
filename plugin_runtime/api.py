"""Stateless public runtime helpers over the canonical plugin manager lifecycle."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from plugin_runtime.dispatch import RenderedPluginSystemPromptSection
from plugin_runtime.lifecycle import (
    delivery_manager,
    ensure_plugins_discovered,
    get_plugin_manager,
    join_background_discovery,
)
from plugin_runtime.loading import LoadedPlugin
from plugin_runtime.manifest import PluginManifest


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook after lazy plugin discovery."""
    return delivery_manager().invoke_hook(hook_name, **kwargs)


async def ainvoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Async hook invocation through the canonical delivery manager."""
    return await delivery_manager().ainvoke_hook(hook_name, **kwargs)


def render_system_prompt_sections(
    session_info: Mapping[str, Any],
) -> List[RenderedPluginSystemPromptSection]:
    """Render plugin prompt sections after idempotent discovery."""
    return ensure_plugins_discovered().render_system_prompt_sections(session_info)


def invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    """Invoke registered middleware after lazy plugin discovery."""
    return delivery_manager().invoke_middleware(kind, **kwargs)


def has_middleware(kind: str) -> bool:
    """Return whether middleware is registered for ``kind`` after lazy discovery."""
    manager = delivery_manager()
    method = getattr(manager, "has_middleware", None)
    if callable(method):
        return bool(method(kind))
    return bool(getattr(manager, "_middleware", {}).get(kind))


def has_hook(hook_name: str) -> bool:
    """Return whether a loaded plugin handles ``hook_name`` after lazy discovery."""
    return delivery_manager().has_hook(hook_name)


def iter_hook_callbacks(hook_name: str) -> tuple[Callable, ...]:
    """Return a stable callback snapshot for ``hook_name``."""
    return get_plugin_manager().iter_hook_callbacks(hook_name)


def get_plugin_context_engine():
    """Return the plugin-registered context engine, or ``None``."""
    return ensure_plugins_discovered()._context_engine


def get_plugin_command_handler(name: str) -> Optional[Callable]:
    """Return the handler for a plugin-registered slash command, or ``None``."""
    entry = ensure_plugins_discovered()._plugin_commands.get(name)
    return entry["handler"] if entry else None


def get_plugin_commands() -> Dict[str, dict]:
    """Return plugin commands after idempotent discovery."""
    return ensure_plugins_discovered()._plugin_commands


def get_plugin_auxiliary_tasks() -> List[Dict[str, Any]]:
    """Return plugin auxiliary-task registrations sorted by key."""
    manager = ensure_plugins_discovered()
    return [manager._aux_tasks[key] for key in sorted(manager._aux_tasks)]


def get_plugin_toolsets() -> List[tuple]:
    """Return plugin toolsets as ``(key, label, description)`` tuples."""
    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []
    try:
        from tools.registry import registry
    except Exception:
        return []

    toolset_tools: Dict[str, List[str]] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if entry:
            toolset_tools.setdefault(entry.toolset, []).append(entry.name)

    toolset_plugin: Dict[str, LoadedPlugin] = {}
    for loaded in manager._plugins.values():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)

    result = []
    for toolset_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(toolset_key)
        description = (
            plugin.manifest.description if plugin else ""
        ) or ", ".join(sorted(toolset_tools[toolset_key]))
        result.append(
            (
                toolset_key,
                f"🔌 {toolset_key.replace('_', ' ').title()}",
                description,
            )
        )
    return result


def get_plugin_subscriptions() -> Dict[str, List[Callable]]:
    """Return an event-name -> callback snapshot after idempotent discovery."""
    manager = ensure_plugins_discovered()
    with manager._event_lock:
        return {
            event: [entry.callback for entry in entries]
            for event, entries in manager._subscriptions.items()
        }


def unload_plugins(
    plugin: Union[str, PluginManifest, LoadedPlugin, None] = None,
) -> bool:
    """Unload one plugin or all plugins after joining background discovery."""
    join_background_discovery()
    return get_plugin_manager().unload(plugin)

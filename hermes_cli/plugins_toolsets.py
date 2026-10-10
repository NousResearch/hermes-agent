"""Plugin toolsets: ``PluginContext`` toolset verbs and the ``hermes tools`` listing of plugin toolsets."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from hermes_cli.plugins import LoadedPlugin
    from hermes_cli.plugins_ledger import PluginRegistration

logger = logging.getLogger("hermes_cli.plugins")


class PluginToolsetMixin:
    """Toolset verbs of ``PluginContext``; reads ``manifest``, ``_manager`` and ``_track`` from it."""

    def register_toolset(
        self, name: str, description: str, tools: List[str] | None = None, includes: List[str] | None = None,
    ) -> Optional[PluginRegistration]:
        """Define a toolset: ``tools`` by name plus every tool of the toolsets in ``includes`` (and any
        tool later registered into ``name``). Selectable wherever a built-in toolset is. Scoped to this
        plugin's profile; a name a built-in toolset, an MCP server alias or another plugin in this profile
        holds is rejected. Unloading the plugin removes it."""
        from toolsets import TOOLSETS
        from tools.registry import registry
        clean = (name or "").strip()
        if not clean or clean in TOOLSETS or registry.get_toolset_alias_target(clean):
            logger.warning("Plugin '%s' tried to register toolset %r, which is empty, a built-in toolset or an "
                           "MCP server alias. Skipping.", self.manifest.name, name)
            return None
        scope = self._manager.scope_key
        definition = registry.register_toolset_definition(
            clean, description, list(tools or []), list(includes or []), scope=scope)
        if definition is None:
            logger.warning("Plugin '%s' tried to register toolset %r, already defined in this profile. "
                           "Skipping.", self.manifest.name, clean)
            return None
        return self._track("toolset", clean,
                           lambda: registry.remove_toolset_definition(clean, definition, scope=scope))

    def add_to_toolset(self, toolset: str, tool_name: str) -> Optional[PluginRegistration]:
        """Add ``tool_name`` to an existing toolset -- typically a built-in bundle such as ``hermes-cli``
        whose tool list is static, which ``register_tool(toolset=...)`` (one toolset per tool) cannot
        reach. Scoped to this plugin's profile; unloading the plugin removes the membership."""
        from toolsets import validate_toolset
        from tools.registry import registry
        if not validate_toolset(toolset):
            logger.warning("Plugin '%s' tried to add %r to unknown toolset %r. Skipping.",
                           self.manifest.name, tool_name, toolset)
            return None
        scope = self._manager.scope_key
        member = registry.add_toolset_member(toolset, tool_name, scope=scope)
        return self._track("toolset_member", f"{toolset}:{tool_name}",
                           lambda: registry.remove_toolset_member(toolset, member, scope=scope))


def get_plugin_toolsets() -> List[tuple]:
    """Plugin toolsets as ``(key, label, description)`` tuples for the ``hermes tools`` TUI."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []
    try:
        from tools.registry import registry
    except Exception:
        return []
    # Group plugin tool names by their toolset, then map each toolset back to the plugin that
    # registered it (first owner wins) for the description.
    toolset_tools: Dict[str, List[str]] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if entry:
            toolset_tools.setdefault(entry.toolset, []).append(entry.name)
    toolset_plugin: Dict[str, "LoadedPlugin"] = {}
    for loaded in manager._plugins.values():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)
    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        desc = (plugin.manifest.description if plugin else "") or ", ".join(sorted(toolset_tools[ts_key]))
        result.append((ts_key, f"🔌 {ts_key.replace('_', ' ').title()}", desc))
    return result

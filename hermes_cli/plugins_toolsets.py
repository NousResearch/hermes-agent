"""Plugin toolset catalog helpers split from :mod:`hermes_cli.plugins`.

The facade still exports ``get_plugin_toolsets`` and resolves its manager at
call time, so profile-manager and test monkeypatch seams remain live.
"""

from __future__ import annotations

from typing import Any, List


def build_plugin_toolsets(manager: Any) -> List[tuple]:
    """Build ``(key, label, description)`` tuples for the ``hermes tools`` TUI."""
    if not manager._plugin_tool_names:
        return []
    try:
        from tools.registry import registry
    except ImportError:
        return []
    # Group plugin tool names by their toolset, then map each toolset back to the plugin that
    # registered it (first owner wins) for the description.
    toolset_tools: dict[str, list[str]] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if entry:
            toolset_tools.setdefault(entry.toolset, []).append(entry.name)
    toolset_plugin: dict[str, Any] = {}
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

"""Compatibility facade for Agent Plugins v1 portable packages.

Runtime-neutral parsing, validation, translation, and liveness live in
:mod:`plugin_runtime.portable`. Startup probing remains CLI-owned here.
"""

from __future__ import annotations

from typing import Any, Mapping

from plugin_runtime.portable import (
    MCP_SCHEMA_V1,
    PLUGIN_SCHEMA_V1,
    AgentPluginDiagnostic,
    AgentPluginError,
    AgentPluginPackage,
    AgentPluginServerDeclaration,
    AgentPluginSkill,
    _clear_liveness,
    _discover_mcp,
    _server_declarations,
    _set_liveness,
    liveness_for,
    load_agent_plugin,
    read_agent_plugin_manifest,
)


def has_enabled_agent_plugin_mcp(raw_config: Mapping[str, Any]) -> bool:
    """Import-compatible wrapper for the shared PluginManager MCP probe."""
    from plugin_runtime.lifecycle import has_enabled_agent_plugin_mcp as _probe

    return _probe(raw_config)

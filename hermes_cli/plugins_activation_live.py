"""Make an installed plugin's MCP servers and skills usable in THIS process now (#87770 follow-up).

:func:`hermes_cli.plugins_activation.activate_plugin_now` loads the plugin; skills are live from that
point (``skill_view`` resolves ``<plugin>:<skill>``). MCP servers are not: nothing connects them until
``reload.mcp``. :func:`connect_plugin_mcp` connects the plugin's servers one by one (the connector
flow's ``register_mcp_servers({name: config})``), and :func:`live_notice` renders what became usable
as the note an open chat receives on its next turn.

MCP tools are deferred behind ``tool_search`` / ``tool_describe`` / ``tool_call``, so adding them to a
live session leaves the model-facing tool array (and the cached prompt prefix) byte-identical.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def connect_plugin_mcp(activation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Connect every portable MCP server ``activation`` defers, under the caller's profile scope.

    Returns ``[{name, connected, tools, error?}]``, one row per server; a server that fails to
    connect is reported, never retried. Never raises."""
    names = list((activation.get("deferred") or {}).get("mcp_servers") or ())
    if not names:
        return []
    try:
        from hermes_cli.plugins import get_plugin_manager
        from tools.mcp_tool_config import _filter_suspicious_mcp_servers, _load_mcp_config
        from tools.mcp_tool_discovery import register_mcp_servers
        configured = _load_mcp_config()
        portable = get_plugin_manager().get_portable_mcp_servers()
    except Exception as exc:
        logger.warning("plugin MCP activation could not read server config: %s", exc)
        return [{"name": n, "connected": False, "tools": [], "error": str(exc)} for n in names]
    rows = []
    for name in names:
        config = configured.get(name) or _filter_suspicious_mcp_servers(
            {name: portable[name]} if name in portable else {}).get(name)
        if not isinstance(config, dict):
            rows.append({"name": name, "connected": False, "tools": [], "error": "no MCP configuration"})
            continue
        try:
            register_mcp_servers({name: config})
            from tools.connectors.mcp import _registered_tool_names
            tools = [t for t in _registered_tool_names(name) if not t.endswith(_utility_suffixes())]
        except Exception as exc:
            rows.append({"name": name, "connected": False, "tools": [], "error": str(exc)})
            continue
        row: Dict[str, Any] = {"name": name, "connected": bool(tools), "tools": tools}
        if not tools:
            row["error"] = _server_error(name) or "the server did not connect"
        rows.append(row)
    return rows


def _utility_suffixes() -> tuple:
    """The resource/prompt wrappers every MCP server gets (``mcp__x__list_resources``, ...): callable,
    but they say nothing about what the plugin does, so they are left out of counts and listings."""
    from tools.mcp_tool_registration import _UTILITY_HANDLER_FACTORIES
    return tuple(f"_{kind}" for kind in _UTILITY_HANDLER_FACTORIES)


def _server_error(name: str) -> Optional[str]:
    try:
        from tools.mcp_tool_discovery import get_mcp_status
        for entry in get_mcp_status():
            if entry.get("name") == name:
                return entry.get("error")
    except Exception:
        return None
    return None


def plugin_skills(plugin_key: str) -> List[Dict[str, str]]:
    """``[{name, description}]`` for the skills ``plugin_key`` registered (qualified ``<ns>:<skill>``)."""
    try:
        from hermes_cli.plugins import get_plugin_manager
        skills = get_plugin_manager()._plugin_skills
    except Exception:
        return []
    return [{"name": qualified, "description": str(entry.get("description") or "")}
            for qualified, entry in sorted(skills.items()) if entry.get("plugin_key") == plugin_key]


def live_notice(activation: Dict[str, Any]) -> str:
    """The note an open chat gets on its next turn: connected MCP servers with their tools, skills,
    and what waits for the next session. Empty when nothing became usable."""
    live = activation.get("live_now") or {}
    servers, skills = live.get("mcp_servers") or [], live.get("skills") or []
    deferred = activation.get("deferred") or {}
    if not servers and not skills:
        return ""
    lines = [f"[plugin installed: {activation.get('name') or activation.get('key')}]"]
    connected = [s for s in servers if s.get("connected")]
    if connected:
        lines.append("MCP servers now connected; call their tools with tool_describe / tool_call:")
        for server in connected:
            lines.append(f"- {server['name']}")
            listing = _tool_listing(server["tools"])
            if listing:
                lines.extend(f"    {line}" for line in listing.splitlines())
    for server in (s for s in servers if not s.get("connected")):
        lines.append(f"- {server['name']}: not connected ({server.get('error') or 'unknown error'})")
    if skills:
        lines.append("Skills now available; load one with skill_view:")
        lines.extend(f"- {s['name']}: {s['description']}" if s.get("description") else f"- {s['name']}"
                     for s in skills)
    later = [f"{len(deferred['tools'])} Python tools" if deferred.get("tools") else "",
             f"{len(deferred['prompt'])} prompt sections" if deferred.get("prompt") else ""]
    if any(later):
        lines.append("Next session: " + ", ".join(p for p in later if p) + ".")
    return "\n".join(lines)


def _tool_listing(names: List[str]) -> str:
    """``name: first sentence`` per tool."""
    try:
        from tools.registry import registry
        from tools.tool_search_catalog import _short_desc
        definitions = registry.get_definitions(set(names), quiet=True)
    except Exception:
        return ""
    rows = []
    for definition in definitions:
        fn = definition.get("function") or definition
        name = str(fn.get("name") or "")
        if name:
            desc = _short_desc(fn.get("description") or "")
            rows.append(f"{name}: {desc}" if desc else name)
    return "\n".join(sorted(rows))

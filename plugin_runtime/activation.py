"""Runtime-owned plugin activation state and in-process go-live lifecycle."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_GO_LIVE_LOCK = threading.Lock()

# Hooks the gateway consults per inbound/outbound message: live as soon as the registry holds them.
_GATEWAY_TRANSFORM_HOOKS = frozenset({
    "transform_llm_output", "transform_tool_result", "transform_terminal_output", "pre_gateway_dispatch",
    "gateway_platform_event", "pre_command",
})


def plugin_activation_summary(manager: Any, plugin_key: str) -> Dict[str, Any]:
    """``{name, key, activated_now: {kind: [names]}, deferred: {kind: [names]}}`` for one loaded plugin,
    read from what it actually registered (ownership ledger + handler registries), not from what its
    manifest promises. Keys appear only when non-empty.

    ``activated_now``: ``gateway_commands`` (slash names), ``gateway_transforms`` / ``hooks`` (hook names),
    ``callbacks`` (platforms with a ``register_platform_handler`` factory / Slack action ids).
    ``deferred``: ``tools`` (tool names; next session), ``prompt`` (section ids; next session),
    ``mcp_servers`` (the plugin's mcp.json server names exactly as registered; until ``mcp.reload``)."""
    loaded = manager._plugins.get(plugin_key)
    manifest = getattr(loaded, "manifest", None)
    name = getattr(manifest, "name", None) or plugin_key
    regs = [r for r in manager._ownership_ledger.get(plugin_key, []) if getattr(r, "active", True)]
    kinds: Dict[str, List[str]] = {}
    for reg in regs:
        kinds.setdefault(reg.kind, []).append(str(reg.key))
    now: Dict[str, List[str]] = {}
    if kinds.get("command"):
        now["gateway_commands"] = sorted(kinds["command"])
    hooks = set(kinds.get("hook", ()))
    if hooks & _GATEWAY_TRANSFORM_HOOKS:
        now["gateway_transforms"] = sorted(hooks & _GATEWAY_TRANSFORM_HOOKS)
    if hooks - _GATEWAY_TRANSFORM_HOOKS:
        now["hooks"] = sorted(hooks - _GATEWAY_TRANSFORM_HOOKS)
    callbacks = sorted(platform for platform, factories in manager._platform_handler_factories.items()
                       if any(plugin == name for _f, plugin in factories))
    callbacks += [f"slack:{a}" for a in kinds.get("slack_action_handler", ())]
    if callbacks:
        now["callbacks"] = callbacks
    deferred: Dict[str, List[str]] = {}
    tools = sorted(set(kinds.get("tool", ())) | set(getattr(loaded, "tools_registered", None) or ())
                   | set(getattr(manifest, "provides_tools", None) or ()))
    if tools:
        deferred["tools"] = tools
    if kinds.get("system_prompt_section"):
        deferred["prompt"] = sorted(kinds["system_prompt_section"])
    servers = sorted(set(kinds.get("portable_mcp", ())) | {
        s for s, owner in manager._portable_mcp_server_plugins.items() if owner == plugin_key})
    if servers:
        deferred["mcp_servers"] = servers
    return {"name": name, "key": plugin_key, "activated_now": now, "deferred": deferred}


def activation_summaries(manager: Any) -> List[Dict[str, Any]]:
    """One summary per loaded (non-deferred-platform, non-errored) plugin — the ``on_plugin_loaded`` payload."""
    out = []
    for key, loaded in list(manager._plugins.items()):
        if getattr(loaded, "deferred", False) or getattr(loaded, "error", None):
            continue
        out.append(plugin_activation_summary(manager, key))
    return out


def find_activation(summaries: Optional[List[Dict[str, Any]]], name: str) -> Optional[Dict[str, Any]]:
    """The summary for ``name`` (manifest name, canonical key, or bare leaf of the key)."""
    for entry in summaries or ():
        key = str(entry.get("key") or "")
        if name in (entry.get("name"), key, key.rsplit("/", 1)[-1]):
            return entry
    return None


def load_and_go_live(name: str) -> Optional[Dict[str, Any]]:
    """Force-rediscover plugins in THIS process under the caller's profile scope, connect ``name``'s
    MCP servers and hand them (and its skills) to this profile's open chats with a turn note. Returns
    the activation summary with ``live_now: {mcp_servers, skills}``; ``deferred`` then keeps only what
    waits for the next session (Python ``tools``, ``prompt``). None when the plugin did not load."""
    # A forced rediscovery unloads every plugin before it loads them again, which drops their server
    # configs, skills and liveness declarations for the length of the pass. Two installs finishing
    # together (one card, two rows) each go live; the second one's pass must not run while the first
    # reads or connects, or the first plugin comes up with no tools. One go-live at a time.
    with _GO_LIVE_LOCK:
        return _go_live(name)


def _go_live(name: str) -> Optional[Dict[str, Any]]:
    from plugin_runtime.activation_live import connect_plugin_mcp, live_notice, plugin_skills
    try:
        from plugin_runtime.lifecycle import (
            get_plugin_manager,
            join_background_discovery,
            refresh_tui_plugin_sessions,
        )
        join_background_discovery()
        manager = get_plugin_manager()
        # Other forced passes (a reload-plugins verb, the dashboard) do not take the go-live lock, so the
        # reads share the discovery lock with the pass that produced them.
        with manager._discovery_lock:
            manager.discover_and_load(force=True)
            activation = find_activation(activation_summaries(manager), name)
            portable = manager.get_portable_mcp_servers()
            skills = plugin_skills(activation["key"]) if activation else []
    except Exception:
        logger.debug("in-process plugin reload after change to %r failed", name, exc_info=True)
        return None
    if activation is None:
        return None
    servers = connect_plugin_mcp(activation, portable)
    activation["live_now"] = {"mcp_servers": servers, "skills": skills}
    activation["deferred"] = {k: v for k, v in (activation.get("deferred") or {}).items() if k != "mcp_servers"}
    note = live_notice(activation)
    if servers or note:
        try:
            refresh_tui_plugin_sessions(note)
        except Exception:
            logger.warning("open chats were not refreshed after activating %r", name, exc_info=True)
    return activation
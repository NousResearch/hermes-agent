"""Mid-run plugin activation: what a just-loaded plugin does NOW vs next session (#87770).

One seam for every surface: ``PluginManager.on_plugin_loaded`` fires with one
:func:`plugin_activation_summary` per loaded plugin. Gateway handlers (slash commands, transform hooks,
platform callbacks) are live as soon as the plugin loads; tools, system-prompt sections and portable MCP
servers stay deferred — tools/prompt to the next session (prompt-cache invariant, same as
``/skills install``), MCP servers until ``mcp.reload``. :func:`activate_plugin_now` is what the install /
enable / update surfaces call after a successful config or tree change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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


def activate_plugin_now(name: str, *, in_process: bool = True) -> Dict[str, Any]:
    """After an install/enable/update: load the plugin in THIS process (so ``on_plugin_loaded``
    subscribers here — the TUI/Desktop server — see it) and nudge the running gateway to do the same
    over its control socket so live adapters re-wire their handlers. Never raises.

    Returns ``{"gateway_reloaded": bool, "activation": summary | None, "restart_required": bool}``.
    ``restart_required`` is True only when no gateway answered (old gateway, not running): with a
    reload, nothing needs a restart — tools/prompt wait for the next session, MCP servers for
    ``mcp.reload``, and that is what ``activation`` says."""
    from hermes_constants import get_hermes_home
    activation: Optional[Dict[str, Any]] = None
    if in_process:
        try:
            from hermes_cli.plugins import discover_plugins, get_plugin_manager
            discover_plugins(force=True)
            activation = find_activation(activation_summaries(get_plugin_manager()), name)
        except Exception:
            logger.debug("in-process plugin reload after change to %r failed", name, exc_info=True)
    answer = None
    try:
        from gateway.control_socket import reload_gateway_plugins
        from hermes_constants import get_default_hermes_root
        home = Path(get_hermes_home())
        answer = reload_gateway_plugins(home)
        if answer is None:
            root = Path(get_default_hermes_root())
            if root.resolve() != home.resolve():  # a served secondary: the multiplexer's socket
                answer = reload_gateway_plugins(root, profile_home=home)
    except Exception:
        logger.debug("gateway reload-plugins nudge for %r failed", name, exc_info=True)
    reloaded = bool(answer and answer.get("reloaded"))
    if reloaded and activation is None:
        activation = find_activation((answer or {}).get("activations"), name)
    return {"gateway_reloaded": reloaded, "activation": activation, "restart_required": not reloaded}


def activation_hint(result: Dict[str, Any]) -> str:
    """One honest sentence for CLI surfaces from an :func:`activate_plugin_now` result."""
    if not result.get("gateway_reloaded"):
        return "Restart the gateway for the plugin to take effect:\n  hermes gateway restart"
    act = result.get("activation") or {}
    now, deferred = act.get("activated_now") or {}, act.get("deferred") or {}
    parts = []
    if now:
        parts.append("active in the running gateway now: " + ", ".join(sorted(now)))
    if deferred:
        labels: Dict[str, str] = {"tools": "tools (next session)", "prompt": "system prompt (next session)",
                                  "mcp_servers": "MCP servers (mcp.reload / next session)"}
        parts.append("deferred: " + ", ".join(labels.get(k, k) for k in sorted(deferred)))
    if not parts:
        return "Gateway reloaded plugins; nothing of this plugin needs a session or restart."
    return "Gateway reloaded plugins — " + "; ".join(parts) + "."

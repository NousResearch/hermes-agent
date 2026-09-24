"""Mid-run plugin activation: what a just-loaded plugin does NOW vs next session (#87770).

One seam for every surface: ``PluginManager.on_plugin_loaded`` fires with one
:func:`plugin_activation_summary` per loaded plugin. Gateway handlers (slash commands, transform hooks,
platform callbacks) are live as soon as the plugin loads; portable MCP servers and skills go live in
the open chats of the plugin's profile (:func:`load_and_go_live`; MCP tools are deferred behind
tool_search, so the cached prompt prefix is unchanged); Python tools and system-prompt sections stay
deferred to the next session (prompt-cache invariant, same as ``/skills install``).
:func:`activate_plugin_now` is what the install / enable / update surfaces call after a successful
config or tree change.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
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


def activate_plugin_now(name: str, *, in_process: bool = True) -> Dict[str, Any]:
    """After an install/enable/update: load the plugin in THIS process (so ``on_plugin_loaded``
    subscribers here — the TUI/Desktop server — see it), connect its MCP servers and hand them plus
    its skills to the open chats of this profile (:func:`load_and_go_live`), and nudge the running
    gateway to load it too over its control socket so live adapters re-wire their handlers. A caller
    in another process (``hermes plugins install``) passes ``in_process=False``; the running Desktop /
    dashboard backend is then asked to do the in-process half (:func:`notify_serve_backend`). Never raises.

    Returns ``{"gateway_reloaded": bool, "activation": summary | None, "restart_required": bool}``.
    ``activation.live_now`` lists what is usable in open chats now; ``activation.deferred`` what waits
    for the next session. ``restart_required`` is True only when no gateway answered (old gateway, not
    running)."""
    from hermes_constants import get_hermes_home
    activation: Optional[Dict[str, Any]] = load_and_go_live(name) if in_process else None
    if not in_process:
        activation = (notify_serve_backend(name, Path(get_hermes_home())) or {}).get("activation")
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
    from hermes_cli.plugins_activation_live import connect_plugin_mcp, live_notice, plugin_skills
    try:
        from hermes_cli.plugins import _join_background_discovery, get_plugin_manager
        _join_background_discovery()
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
    import sys
    server = sys.modules.get("tui_gateway.server")  # loaded == this process hosts chats
    note = live_notice(activation)
    if server is not None and (servers or note):
        from hermes_constants import get_hermes_home
        try:
            server.refresh_plugin_sessions(Path(get_hermes_home()), note)
        except Exception:
            logger.warning("open chats were not refreshed after activating %r", name, exc_info=True)
    return activation


def notify_serve_backend(name: str, home: Path) -> Optional[Dict[str, Any]]:
    """Ask the running Desktop / dashboard backend (``hermes serve``, found through its host record) to
    run :func:`load_and_go_live` for ``name`` in ``home``. None when no backend answers. Never raises."""
    try:
        import json
        import urllib.request
        from gateway import host_rendezvous as hr
        record = hr.read_record(hr.ROLE_SERVE)
        if record is None or not record.port or not hr.record_token_is_consistent(record):
            return None
        token = hr.read_token(hr.ROLE_SERVE)
        request = urllib.request.Request(
            f"http://{hr.dial_host(record)}:{record.port}/api/dashboard/agent-plugins/activate",
            data=json.dumps({"name": name, "home": str(home)}).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "X-Hermes-Session-Token": token})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 — loopback http
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        logger.debug("serve backend activation for %r failed", name, exc_info=True)
        return None


def activation_hint(result: Dict[str, Any]) -> str:
    """One honest sentence for CLI surfaces from an :func:`activate_plugin_now` result."""
    act = result.get("activation") or {}
    live = act.get("live_now") or {}
    connected = [s for s in live.get("mcp_servers") or () if s.get("connected")]
    lines = []
    if connected or live.get("skills"):
        tools = sum(len(s.get("tools") or ()) for s in connected)
        lines.append(f"Live in open chats now: {tools} MCP tools, {len(live.get('skills') or ())} skills.")
    lines.extend(f"MCP server {s['name']} not connected: {s.get('error') or 'unknown error'}"
                 for s in live.get("mcp_servers") or () if not s.get("connected"))
    if not result.get("gateway_reloaded"):
        if lines:  # live in open chats; a messaging gateway that starts later loads it at boot
            return "\n".join(lines)
        return "\n".join([*lines, "Restart the gateway for the plugin to take effect:\n  hermes gateway restart"])
    now, deferred = act.get("activated_now") or {}, act.get("deferred") or {}
    parts = []
    if now:
        parts.append("active in the running gateway now: " + ", ".join(sorted(now)))
    if deferred:
        labels: Dict[str, str] = {"tools": "tools (next session)", "prompt": "system prompt (next session)",
                                  "mcp_servers": "MCP servers (next session)"}
        parts.append("deferred: " + ", ".join(labels.get(k, k) for k in sorted(deferred)))
    if not parts:
        return "\n".join([*lines, "Gateway reloaded plugins; nothing of this plugin needs a session or restart."])
    return "\n".join([*lines, "Gateway reloaded plugins — " + "; ".join(parts) + "."])

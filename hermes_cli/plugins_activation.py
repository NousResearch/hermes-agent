"""Cross-process plugin activation orchestration and CLI-facing activation hints.

Runtime-owned activation state and in-process go-live mechanics live in
``plugin_runtime.activation``. This module is the process-boundary edge used by install / enable /
update surfaces: it coordinates the local runtime activation with the running Gateway and, when the
caller is a separate CLI process, with the serving Dashboard/Desktop backend.

``PluginManager.on_plugin_loaded`` emits runtime activation summaries. Gateway handlers (slash
commands, transform hooks, platform callbacks) are live as soon as the plugin loads; portable MCP
servers and skills can be made live in open chats without rebuilding the cached prompt prefix; Python
tools and system-prompt sections stay deferred to the next session.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from plugin_runtime import activation as runtime_activation

logger = logging.getLogger(__name__)

def activate_plugin_now(name: str, *, in_process: bool = True) -> Dict[str, Any]:
    """After an install/enable/update: load the plugin in THIS process (so ``on_plugin_loaded``
    subscribers here — the TUI/Desktop server — see it), connect its MCP servers and hand them plus
    its skills to the open chats of this profile (``plugin_runtime.activation.load_and_go_live``), and nudge the running
    gateway to load it too over its control socket so live adapters re-wire their handlers. A caller
    in another process (``hermes plugins install``) passes ``in_process=False``; the running Desktop /
    dashboard backend is then asked to do the in-process half (:func:`notify_serve_backend`). Never raises.

    Returns ``{"gateway_reloaded": bool, "activation": summary | None, "restart_required": bool}``.
    ``activation.live_now`` lists what is usable in open chats now; ``activation.deferred`` what waits
    for the next session. ``restart_required`` is True only when no gateway answered (old gateway, not
    running)."""
    from hermes_constants import get_hermes_home
    activation: Optional[Dict[str, Any]] = runtime_activation.load_and_go_live(name) if in_process else None
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
        activation = runtime_activation.find_activation((answer or {}).get("activations"), name)
    return {"gateway_reloaded": reloaded, "activation": activation, "restart_required": not reloaded}


def _serve_backend_record():
    """The host-owner record, else the Desktop child's (a Desktop-only box has no host owner)."""
    from gateway import host_rendezvous as hr
    for role in (hr.ROLE_SERVE, hr.ROLE_DESKTOP_SERVE):
        record = hr.read_record(role)
        if record is not None and record.port and hr.record_token_is_consistent(record):
            return record
    return None


def notify_serve_backend(name: str, home: Path) -> Optional[Dict[str, Any]]:
    """Ask the running dashboard / Desktop backend (``hermes serve``, found through its host record) to
    run ``plugin_runtime.activation.load_and_go_live`` for ``name`` in ``home``. None when no backend answers. Never raises."""
    try:
        import json
        import urllib.request
        from gateway import host_rendezvous as hr
        record = _serve_backend_record()
        if record is None:
            return None
        token = hr.read_token(record.role)
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
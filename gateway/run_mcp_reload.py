"""Gateway control verb for one MCP server, without a gateway-wide reconnect."""
from __future__ import annotations

from contextlib import nullcontext
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def reload_scoped_mcp_verb(runner: Any) -> Callable[[dict], dict]:
    """Local socket verb; server/profile must be configured and served by this process.

    Drain tracked calls before closing the old transport. A failed preflight leaves it
    intact; once shutdown begins the previous process cannot be restored atomically.
    """
    def _handler(params: dict) -> dict:
        from hermes_constants import get_hermes_home, hermes_home_key
        from gateway.run import _profile_runtime_scope
        from tools import mcp_tool as core
        from tools import mcp_tool_config as config
        from tools import mcp_tool_discovery as discovery
        from tools import mcp_tool_lifecycle as lifecycle
        from tools import mcp_tool_loop as mcp_loop
        from tools.mcp_tool_common import mcp_server_enabled
        from tools.mcp_tool_reload import drain_calls
        from tools.mcp_tool_scope import _resolve_server_key

        params = params or {}
        name = params.get("name")
        if not isinstance(name, str) or not name or len(name) > 128:
            return {"status": "rejected", "error": "name must be a configured server name"}
        gateway_home = Path(get_hermes_home())
        home_raw = params.get("home", str(gateway_home))
        if not isinstance(home_raw, str) or not home_raw:
            return {"status": "rejected", "error": "invalid home"}
        home = Path(home_raw).expanduser()
        key = hermes_home_key(home)
        primary = hermes_home_key(gateway_home)
        if key != primary:
            served = getattr(runner, "_served_profile_homes", None) or {}
            if not getattr(runner.config, "multiplex_profiles", False) or not any(
                    hermes_home_key(h) == key for h in served.values()):
                return {"status": "rejected", "error": "home is not served by this gateway"}
        raw_timeout = params.get("drain_timeout", 15.0)
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)) or not 0 <= raw_timeout <= 60:
            return {"status": "rejected", "error": "drain_timeout must be between 0 and 60 seconds"}
        scope_context = (_profile_runtime_scope(home) if getattr(runner.config, "multiplex_profiles", False)
                         else nullcontext())
        with scope_context:
            servers = config._load_mcp_config()
            if name not in servers or not mcp_server_enabled(servers[name]):
                return {"status": "rejected", "error": "server is not enabled in this profile"}
            scope = core._mcp_registry_scope()
            with drain_calls(name, timeout=float(raw_timeout)) as drained:
                if not drained:
                    return {"status": "pending", "name": name, "home": str(home), "retry": True}
                with core._lock:
                    server_key = _resolve_server_key(name)
                    if server_key in core._server_connecting:
                        return {"status": "pending", "name": name, "home": str(home), "retry": True}
                    if server_key in core._lazy_server_configs:
                        return {"status": "pending", "name": name, "home": str(home), "retry": False,
                                "error": "lazy schema-cache registration is not a live connection"}
                    owner = core._server_scope_keys.get(server_key, scope)
                    adopters = core._server_tool_scopes.get(server_key, set()) - {scope}
                    if owner != scope or adopters:
                        return {"status": "pending", "name": name, "home": str(home),
                                "error": "connection is shared with another profile", "retry": False}
                    old = core._servers.get(server_key)
                # A caller may have timed out or been interrupted before the SDK
                # acknowledged cancellation. Its sync admission has ended, but the RPC
                # can still be executing on the MCP loop: do not cancel it during swap.
                if old is not None and getattr(old, "_inflight_tasks", None):
                    return {"status": "pending", "name": name, "home": str(home), "retry": True}
                if old is not None:
                    # Connect the NEW runtime before touching the old one. The preview is not
                    # registered, and is always closed on its owning MCP loop.
                    if not core._ensure_mcp_sdk():
                        return {"status": "failed", "name": name, "error": "MCP SDK unavailable"}
                    mcp_loop._ensure_mcp_loop()

                    async def _preflight():
                        candidate = await discovery._connect_server(name, servers[name])
                        try:
                            if not candidate._tools:
                                raise RuntimeError("replacement server exposes no tools")
                        finally:
                            await candidate.shutdown()

                    try:
                        mcp_loop._run_on_mcp_loop(_preflight, timeout=float(servers[name].get("connect_timeout", 60)) + 5)
                    except Exception as exc:
                        logger.warning("MCP scoped reload preflight failed for %s: %s", name, exc)
                        return {"status": "failed", "name": name, "phase": "preflight", "old_preserved": True,
                                "error": type(exc).__name__}
                if old is not None and getattr(old, "_inflight_tasks", None):
                    return {"status": "pending", "name": name, "home": str(home), "retry": True}
                lifecycle.shutdown_mcp_servers(scope=scope, names={name})
                with core._lock:
                    if old is not None and core._servers.get(server_key) is old:
                        return {"status": "failed", "name": name, "phase": "shutdown", "old_preserved": False,
                                "error": "old connection did not close"}
                try:
                    from tools.mcp_oauth import suppress_interactive_oauth
                    with suppress_interactive_oauth():
                        discovery.discover_mcp_tools(allowed_mcp_names=[name])
                    with core._lock:
                        new = core._servers.get(server_key)
                        names = list(getattr(new, "_registered_tool_names", ()) or ()) if new else []
                    if new is None or new.session is None or not names:
                        raise RuntimeError("replacement server did not register live tools")
                    from tools.mcp_tool_agent import reprobe_tool_availability
                    reprobe_tool_availability()
                    # Active turns retain their tools[] snapshot. Their next turn refreshes
                    # content-aware; idle cached agents also refresh on their next turn.
                    return {"status": "reloaded", "name": name, "home": str(home), "tools": names}
                except Exception as exc:
                    logger.warning("MCP scoped reload failed after shutdown for %s: %s", name, exc)
                    return {"status": "failed", "name": name, "phase": "reconnect", "old_preserved": False,
                            "error": type(exc).__name__}
    return _handler

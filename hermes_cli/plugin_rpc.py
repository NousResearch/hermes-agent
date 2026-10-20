"""Plugin JSON-RPC methods: a plugin's frontend calling its own Python half over the gateway socket.

Bound into ``PluginContext`` (``from hermes_cli.plugin_rpc import ...``), so they are ordinary
``ctx.*`` methods and ``register_rpc`` gets the abandoned-load guard of every registrar. The session
authority serves the methods (``gateway/session_plugin_rpc.py``); unload drops them with the ledger.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from hermes_cli.plugin_events import PLUGIN_EVENT_PREFIX, emit_plugin_session_event, plugin_event_name
from hermes_cli.plugins_ledger import PluginRegistration

__all__ = ["emit_to_session", "plugin_rpc_handler", "register_rpc"]


def register_rpc(self, name: str, handler: Callable[..., Any]) -> PluginRegistration:
    """Serve ``handler(params, *, session_id, **kwargs)`` as ``plugin.<plugin_id>.<name>``: the namespace
    is FORCED (``name`` is the bare dotted name). The return value, JSON-encodable, is the result."""
    if not callable(handler):
        raise TypeError(f"Plugin '{self.plugin_id}' register_rpc() handler must be callable")
    method = plugin_event_name(self.plugin_id, name, kind="RPC method")
    return self._register_entry("rpc_method", method, self._manager._plugin_rpc_methods, handler,
                                "Plugin %s registered RPC method: %s", method)


def emit_to_session(self, session_id: str, event: str, payload: Optional[dict[str, Any]] = None) -> bool:
    """Push ``plugin.<plugin_id>.<event>`` to the viewers of ``session_id`` only."""
    return emit_plugin_session_event(self.plugin_id, session_id, event, payload)


def plugin_rpc_handler(method: Any) -> Optional[Callable[..., Any]]:
    """The loaded plugin handler serving wire ``method`` for the active profile, else None."""
    if not isinstance(method, str) or not method.startswith(PLUGIN_EVENT_PREFIX):
        return None
    from hermes_cli.plugins import _ensure_plugins_discovered
    return _ensure_plugins_discovered()._plugin_rpc_methods.get(method)

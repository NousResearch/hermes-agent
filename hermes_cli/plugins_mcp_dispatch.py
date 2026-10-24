"""Profile-local registration for mandatory MCP dispatch guards.

Unlike optional tool hooks, these callbacks are selected by an independent
profile policy and their absence is an authorization failure.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cached_property
from threading import Lock
from typing import Callable


@dataclass(frozen=True)
class PluginMCPCall:
    """Host-recorded provenance of one ``PluginContext.call_mcp`` invocation.

    This is execution-local authority, never a value supplied in MCP arguments
    or tool-dispatch kwargs. It is set only around the native call and reset even
    if the call fails. A required guard can use it to distinguish a plugin tool
    from a model's raw MCP alias for the same server and operation.
    """

    plugin_id: str
    server: str
    tool_name: str


class _PluginMCPCallLease:
    """One native dispatch can consume this context, even if a worker copied it."""

    def __init__(self, call: PluginMCPCall):
        self.call = call
        self._lock = Lock()
        self._consumed = False

    def peek(self) -> PluginMCPCall | None:
        with self._lock:
            return None if self._consumed else self.call

    def take(self) -> PluginMCPCall | None:
        with self._lock:
            if self._consumed:
                return None
            self._consumed = True
            return self.call


_plugin_mcp_call: ContextVar[_PluginMCPCallLease | None] = ContextVar("plugin_mcp_call", default=None)


def current_plugin_mcp_call() -> PluginMCPCall | None:
    """Return unconsumed host provenance, if this is inside ``call_mcp``."""
    lease = _plugin_mcp_call.get()
    return lease.peek() if lease is not None else None


def _take_plugin_mcp_call() -> PluginMCPCall | None:
    """Consume provenance at the one native dispatch boundary."""
    lease = _plugin_mcp_call.get()
    return lease.take() if lease is not None else None


@contextmanager
def _bind_plugin_mcp_call(plugin_id: str, server: str, tool_name: str):
    token = _plugin_mcp_call.set(_PluginMCPCallLease(PluginMCPCall(plugin_id, server, tool_name)))
    try:
        yield
    finally:
        _plugin_mcp_call.reset(token)


def call_plugin_mcp(plugin_id: str, server: str, tool_name: str, timeout: float, args: dict):
    """Run the existing native MCP handler with one host-owned provenance lease."""
    from tools.mcp_tool_handlers import _make_tool_handler
    with _bind_plugin_mcp_call(plugin_id, server, tool_name):
        return _make_tool_handler(server, tool_name, timeout)(args)


class PluginMCPDispatchManagerMixin:
    @cached_property
    def _required_mcp_dispatch_guards(self) -> dict[str, tuple[str, Callable]]:
        return {}


class PluginMCPDispatchContextMixin:
    def register_required_mcp_dispatch_guard(self, *, server: str, callback: Callable):
        """Register this plugin's one mandatory, profile-local guard for a raw MCP server.

        ``callback(server=..., tool_name=..., args=..., plugin_call=...,
        **call_ids)`` must return the complete canonical argument object.
        ``plugin_call`` is an immutable host record for the first native dispatch
        made by ``PluginContext.call_mcp``; raw aliases receive ``None``. A policy
        in config.yaml decides whether the guard is required; registration alone
        does not change permissions.
        """
        if self._load_abandoned:
            raise self._refuse("an MCP dispatch guard after load timed out")
        if (not isinstance(server, str) or not server or server.strip() != server
                or not callable(callback) or server in self._manager._required_mcp_dispatch_guards):
            raise self._refuse(f"a duplicate or invalid MCP dispatch guard for {server!r}")
        entry = (self.plugin_id, callback)
        return self._register_entry(
            "required_mcp_dispatch_guard", server, self._manager._required_mcp_dispatch_guards,
            entry, "Plugin %s registered required MCP dispatch guard: %s", server,
        )

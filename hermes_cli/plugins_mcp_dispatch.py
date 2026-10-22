"""Profile-local registration for mandatory MCP dispatch guards.

Unlike optional tool hooks, these callbacks are selected by an independent
profile policy and their absence is an authorization failure.
"""

from functools import cached_property
from typing import Callable


class PluginMCPDispatchManagerMixin:
    @cached_property
    def _required_mcp_dispatch_guards(self) -> dict[str, tuple[str, Callable]]:
        return {}


class PluginMCPDispatchContextMixin:
    def register_required_mcp_dispatch_guard(self, *, server: str, callback: Callable):
        """Register this plugin's one mandatory, profile-local guard for a raw MCP server.

        ``callback(server=..., tool_name=..., args=..., **call_ids)`` must return
        the complete canonical argument object. A policy in config.yaml decides
        whether it is required; registration alone does not change permissions.
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

"""MCP reload for GatewayRunner: reconnect MCP servers and refresh cached agents' tool lists.

Split out of ``gateway/run_turn.py``; bound onto ``GatewayRunner`` through ``GatewayTurnMixin``.
``gateway.run`` internals are imported lazily inside method bodies (import cycle).
"""

from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path

from agent.i18n import t
from gateway.platforms.event import MessageEvent
from gateway.session import _session_key_namespace
from hermes_constants import get_hermes_home_override

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayMcpReloadMixin:
    """``/reload-mcp`` execution for GatewayRunner (see module docstring)."""

    def _mcp_reload_refresh_cached_agents(self, multiplex: bool, profile) -> None:
        """Refresh cached agents so existing sessions see new MCP tools on their next turn without
        a history-destroying ``/new``. Each agent keeps its build-time toolset selection EXACTLY: a
        session built with restricted enabled_toolsets (e.g. ["safe"]) must NOT silently gain tools."""
        try:
            from tools.mcp_tool_agent import refresh_agent_mcp_tools
            _cache = getattr(self, "_agent_cache", None)
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            if _cache_lock is None or not _cache:
                return
            # Multiplex: only this profile's sessions (another profile's agent would get this registry).
            _ns_prefix = _session_key_namespace(profile) + ":" if multiplex else None
            with _cache_lock:
                for _sess_key, _entry in list(_cache.items()):
                    if _ns_prefix and not str(_sess_key).startswith(_ns_prefix):
                        continue
                    _agent = _entry[0] if isinstance(_entry, tuple) else _entry
                    if _agent is not None:
                        refresh_agent_mcp_tools(_agent, quiet_mode=True)
        except Exception as _exc:
            logger.debug("Failed to update cached agent tools after MCP reload: %s", _exc)

    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        """Disconnect, reconnect, and notify MCP tool changes (shared by button / text / no-confirm paths).

        Under multiplex the reload runs inside the requesting profile's runtime scope (entered here
        when the caller did not) and only that profile's servers are torn down and rediscovered.

        See #95518.
        """
        from gateway.run import _profile_runtime_scope
        multiplex = bool(getattr(self.config, "multiplex_profiles", False))
        if multiplex and not get_hermes_home_override():
            profile_home = self._resolve_profile_home_for_source(event.source)
            with _profile_runtime_scope(Path(profile_home)):
                return await self._execute_mcp_reload(event)
        try:
            from tools.mcp_tool_lifecycle import shutdown_mcp_servers
            from tools.mcp_tool_discovery import discover_mcp_tools
            from tools.mcp_tool import _servers, _lock, _server_visible_in_scope
            from tools.mcp_tool_agent import reprobe_tool_availability
            from tools.mcp_tool_scope import _key_name
            from tools.registry import registry

            reload_scope = registry.current_scope_key() if multiplex else None

            def _scoped_server_names() -> set:
                with _lock:
                    return {
                        _key_name(key) for key in _servers
                        if _server_visible_in_scope(key, reload_scope)
                    }

            old_servers = _scoped_server_names()
            await self._run_in_executor_with_context(lambda: shutdown_mcp_servers(scope=reload_scope))
            # Explicit reload also re-probes tool availability (check_fn).
            reprobe_tool_availability()
            # Reconnect by discovering tools (reads config.yaml fresh). A chat command cannot finish
            # a browser OAuth flow either: an expired token parks with a `hermes mcp login` hint.
            from tools.mcp_oauth import suppress_interactive_oauth
            with suppress_interactive_oauth():
                new_tools = await self._run_in_executor_with_context(discover_mcp_tools)

            connected_servers = _scoped_server_names()
            if reload_scope is not None:
                from tools.mcp_tool import _mcp_tool_server_names
                with _lock:
                    new_tools = [n for n in new_tools if _mcp_tool_server_names.get(n) in connected_servers]
            # (label, i18n key, names); i18n lines list reconnected first, the injected note added first.
            changes = (
                ("Reconnected", "gateway.reload_mcp.reconnected", connected_servers & old_servers),
                ("Added", "gateway.reload_mcp.added", connected_servers - old_servers),
                ("Removed", "gateway.reload_mcp.removed", old_servers - connected_servers),
            )
            lines = [t("gateway.reload_mcp.header")] + [
                t(key, names=", ".join(sorted(names))) for _label, key, names in changes if names
            ]
            if not connected_servers:
                lines.append(t("gateway.reload_mcp.none_connected"))
            else:
                lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

            self._mcp_reload_refresh_cached_agents(multiplex, event.source.profile)

            # Append a note at the END of the history (preserves the prompt-cache prefix).
            change_parts = [
                f"{label} servers: {', '.join(sorted(names))}"
                for label, _key, names in (changes[1], changes[2], changes[0]) if names
            ]
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            with suppress(Exception):  # Best-effort; don't fail the reload over a transcript write
                session_entry = await self.async_session_store.get_or_create_session(event.source)
                await self.async_session_store.append_to_transcript(session_entry.session_id, reload_msg)

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return t("gateway.reload_mcp.failed", error=e)

"""Profile and session toolset selection for gateway turns."""

from __future__ import annotations

import logging

from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class TurnToolsetsMixin:
    """Resolve configured and session-owned tools at the existing gateway boundary."""

    def _resolve_enabled_toolsets_for_source(
        self, user_config: dict, source: SessionSource, platform_key: str,
    ) -> list:
        """Enabled toolsets for an agent run, honoring an adapter ``toolsets_for_source()`` override
        validated through the SAME ``_get_platform_tools`` path (unknown / platform-restricted
        toolsets dropped, not trusted)."""
        from hermes_cli.tools_config import _get_platform_tools
        try:
            adapter = self._delivery_adapter_for(source)
            override = adapter.toolsets_for_source(source) if adapter is not None else None
        except Exception:
            logger.debug("Adapter toolset override failed; using configured selection", exc_info=True)
            override = None
        if override and isinstance(override, list):
            pts = dict(user_config.get("platform_toolsets") or {})
            pts[platform_key] = [str(x) for x in override]
            user_config = {**user_config, "platform_toolsets": pts}
        enabled = _get_platform_tools(user_config, platform_key)
        session_key = self._session_key_for_source(source)
        if session_key:
            from hermes_cli.plugin_session_toolsets import session_toolset_names
            from tools.registry import registry
            enabled.update(session_toolset_names(session_key, scope=registry.current_scope_key()))
        return sorted(enabled)


    def _resolve_turn_toolsets(self, user_config: dict, source: SessionSource, platform_key: str):
        """``(enabled_toolsets, disabled_toolsets)`` for an agent run on ``source``."""
        from agent.skill_utils import parse_config_string_list
        enabled = self._resolve_enabled_toolsets_for_source(user_config, source, platform_key)
        disabled = parse_config_string_list((user_config.get("agent") or {}).get("disabled_toolsets")) or None
        return enabled, disabled


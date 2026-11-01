"""Tool catalog selection for a gateway turn, including exact-channel overrides."""

from __future__ import annotations

import logging
from typing import Optional

from gateway.session import SessionSource

logger = logging.getLogger(__name__)


class GatewayTurnToolsMixin:
    def _resolve_enabled_toolsets_for_source(
        self, user_config: dict, source: SessionSource, platform_key: str,
    ) -> list:
        """Apply an exact-channel catalog, else the platform and adapter configuration."""
        protected = self._protected_channel_toolsets(source)
        if protected is not None:
            from gateway.protected_channel_tools import effective_toolsets
            return effective_toolsets(protected)
        from hermes_cli.tools_config import _get_platform_tools
        try:
            adapter = self._delivery_adapter_for(source)
            override = adapter.toolsets_for_source(source) if adapter is not None else None
        except Exception:
            logger.debug("Adapter toolset override failed; using platform selection", exc_info=True)
            override = None
        if override and isinstance(override, list):
            pts = dict(user_config.get("platform_toolsets") or {})
            pts[platform_key] = [str(x) for x in override]
            user_config = {**user_config, "platform_toolsets": pts}
        return sorted(_get_platform_tools(user_config, platform_key))

    def _protected_channel_toolsets(self, source: SessionSource):
        """Exact receiving-bot policy, including durable native restore provenance."""
        from gateway.protected_channel_tools import policy_for_source
        from gateway.session_identity import identity_of

        identity = identity_of(source)
        # Native admission persists a credential/home binding, then restore_native
        # verifies it and reattaches the registered adapter ref. Standalone gateways
        # do not repin RoutingIdentity on that restored source. Resolve only through
        # the registered receiver; never infer a bot from platform or channel.
        owner = self._transport_owner(source)
        if owner is None:
            return policy_for_source(getattr(self, "config", None), source,
                                     transport_profile="", live_adapter=False)
        adapter, owner_profile = owner
        primary = getattr(self, "_primary_profile_name", None)
        if not primary:
            active = getattr(self, "_active_profile_name", None)
            primary = active() if callable(active) else "default"
        receiver = owner_profile or primary
        verified = (identity.adapter() is adapter and identity.transport_profile == receiver
                    if identity is not None else getattr(source, "_native_provenance_verified", None) is True)
        return policy_for_source(
            getattr(self, "config", None), source,
            transport_profile=receiver, live_adapter=verified,
        )

    def _protected_channel_tool_names(self, source: SessionSource, disabled_toolsets: Optional[list[str]]):
        policy = self._protected_channel_toolsets(source)
        if policy is None:
            return None
        from gateway.protected_channel_tools import allowed_tool_names
        return allowed_tool_names(policy, disabled_toolsets)

    def _resolve_turn_toolsets(self, user_config: dict, source: SessionSource, platform_key: str):
        """``(enabled_toolsets, disabled_toolsets)`` for an agent run on ``source``."""
        from agent.skill_utils import parse_config_string_list
        enabled = self._resolve_enabled_toolsets_for_source(user_config, source, platform_key)
        disabled = parse_config_string_list((user_config.get("agent") or {}).get("disabled_toolsets")) or None
        return enabled, disabled

"""Resolve individual channel and topic fields through inherited settings."""
from __future__ import annotations
from typing import Any, Optional
from gateway.config import ChannelOverride, GatewayConfig, Platform


def _get_channel_override_field(
    config: GatewayConfig,
    platform: Platform,
    chat_id: str,
    field_name: str,
    *,
    thread_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Any:
    """Resolve one override field; omitted exact entries continue inheritance."""
    from gateway.run import _channel_override_lookup_keys

    platform_config = (getattr(config, "platforms", None) or {}).get(platform)
    overrides = getattr(platform_config, "channel_overrides", None) or {}
    for key in _channel_override_lookup_keys(chat_id, thread_id=thread_id, parent_id=parent_id):
        override = overrides.get(key)
        value = getattr(override, field_name, None) if override is not None else None
        if value is not None:
            return value
    return None


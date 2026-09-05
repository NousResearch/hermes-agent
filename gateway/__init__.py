"""Hermes Gateway - multi-platform messaging integration (sessions, context
injection, delivery routing, platform-specific toolsets)."""

from .config import GatewayConfig, PlatformConfig, HomeChannel, SessionResetPolicy, load_gateway_config
from .session import (
    SessionContext,
    SessionStore,
    build_session_context_prompt,
)
from .delivery import DeliveryRouter, DeliveryTarget
from .state_db_authority import install_gateway_state_db_authority

# Installation is passive: no path is resolved and no database is opened until
# a writable SessionDB is actually constructed by a gateway surface.
install_gateway_state_db_authority()

__all__ = [
    "GatewayConfig", "PlatformConfig", "HomeChannel", "load_gateway_config",
    "SessionContext", "SessionStore", "SessionResetPolicy", "build_session_context_prompt",
    "DeliveryRouter", "DeliveryTarget",
]

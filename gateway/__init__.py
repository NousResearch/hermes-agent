"""Hermes Gateway - multi-platform messaging integration (sessions, context
injection, delivery routing, platform-specific toolsets)."""

from .config import GatewayConfig, PlatformConfig, HomeChannel, SessionResetPolicy, load_gateway_config
from .session import (
    SessionContext,
    SessionStore,
    build_session_context_prompt,
)
from .delivery import DeliveryRouter, DeliveryTarget
from .scoped_reasoning import install_scoped_reasoning_runtime

# Gateway reasoning overrides are more specific than model/global defaults and
# must stay fixed through same-turn provider/model route changes.
install_scoped_reasoning_runtime()

__all__ = [
    "GatewayConfig", "PlatformConfig", "HomeChannel", "load_gateway_config",
    "SessionContext", "SessionStore", "SessionResetPolicy", "build_session_context_prompt",
    "DeliveryRouter", "DeliveryTarget",
]

"""Hermes Gateway - multi-platform messaging integration (sessions, context
injection, delivery routing, platform-specific toolsets)."""

# Must precede config/session imports: session's compression path can import
# agent.redact, which snapshots HERMES_REDACT_LEVEL at module import time.
from ._startup import bootstrap_gateway_redaction

bootstrap_gateway_redaction()

from .config import GatewayConfig, PlatformConfig, HomeChannel, load_gateway_config
from .session import (
    SessionContext,
    SessionStore,
    build_session_context_prompt,
)
from .delivery import DeliveryRouter, DeliveryTarget

__all__ = [
    "GatewayConfig", "PlatformConfig", "HomeChannel", "load_gateway_config",
    "SessionContext", "SessionStore", "build_session_context_prompt",
    "DeliveryRouter", "DeliveryTarget",
]

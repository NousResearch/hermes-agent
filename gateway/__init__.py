"""
Hermes Gateway - Multi-platform messaging integration.

This module provides a unified gateway for connecting the Hermes agent
to various messaging platforms (Telegram, Discord, WhatsApp) with:
- Session management (persistent conversations with reset policies)
- Dynamic context injection (agent knows where messages come from)
- Delivery routing (cron job outputs to appropriate channels)
- Platform-specific toolsets (different capabilities per platform)
"""

from .config import GatewayConfig, HomeChannel, PlatformConfig, load_gateway_config
from .delivery import DeliveryRouter, DeliveryTarget
from .session import (
    SessionContext,
    SessionResetPolicy,
    SessionStore,
    build_session_context_prompt,
)

__all__ = [
    # Config
    "GatewayConfig",
    "PlatformConfig",
    "HomeChannel",
    "load_gateway_config",
    # Session
    "SessionContext",
    "SessionStore",
    "SessionResetPolicy",
    "build_session_context_prompt",
    # Delivery
    "DeliveryRouter",
    "DeliveryTarget",
]

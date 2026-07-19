"""
Hermes Gateway - Multi-platform messaging integration.

This module provides a unified gateway for connecting the Hermes agent
to various messaging platforms (Telegram, Discord, WhatsApp, Weixin, and more) with:
- Session management (persistent conversations with reset policies)
- Dynamic context injection (agent knows where messages come from)
- Delivery routing (cron job outputs to appropriate channels)
- Platform-specific toolsets (different capabilities per platform)
"""

# ``python -m gateway.run`` imports this package before loading gateway.run.
# Run checkout-drift cleanup only for that documented entry point; a regular
# ``import gateway`` must remain free of gateway-startup side effects.
import sys


def _is_direct_gateway_module_execution() -> bool:
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    return getattr(main_spec, "name", None) == "gateway.run"


if _is_direct_gateway_module_execution():
    from hermes_cli.gateway_bootstrap import purge_stale_gateway_pycache_before_import

    purge_stale_gateway_pycache_before_import()

from .config import GatewayConfig, PlatformConfig, HomeChannel, load_gateway_config
from .session import (
    SessionContext,
    SessionStore,
    SessionResetPolicy,
    build_session_context_prompt,
)
from .delivery import DeliveryRouter, DeliveryTarget

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

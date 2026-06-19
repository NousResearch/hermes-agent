"""Send Gate API Server Layer 3: HTTP-level rejection of send requests.

This module implements Layer 3 of the send-gate feature: API server rejection.

Layer 1 (in gateway/platforms/base.py): send() raises SendGateDisabledException
Layer 2 (in tools/send_gate_tool.py): Tool registration filtering
Layer 3 (here): HTTP request handler rejection when send_gate=disabled

When send_gate is disabled on any enabled platform, the API server rejects
chat/completion and response API requests with a 403 Forbidden status,
preventing the agent from being invoked and messages being sent.

This provides defense-in-depth: even if Layers 1 and 2 fail, the API server
still blocks send requests at the HTTP level.
"""

import logging
from typing import Optional, Tuple

from gateway.config import GatewayConfig, Platform

logger = logging.getLogger(__name__)


def _get_gateway_config() -> Optional[GatewayConfig]:
    """Load the gateway config if available.

    Returns None if not in gateway context or config is unavailable.
    """
    try:
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        return config
    except (ImportError, FileNotFoundError, Exception):
        # Not in gateway context, or config loading failed
        return None


def check_send_gate_enabled_for_api(
    config: Optional[GatewayConfig] = None,
) -> Tuple[bool, Optional[str]]:
    """Check if send_gate is enabled for all platforms at the API server level.

    Args:
        config: Optional GatewayConfig. If not provided, will attempt to load
               from gateway context. If no config is available, defaults to
               allowing sends (fail-open).

    Returns:
        Tuple of (is_enabled, error_message):
        - (True, None): Sends are allowed
        - (False, str): Sends are blocked; error message explains why and how to fix

    The check examines all enabled platforms:
    - If any enabled platform has send_gate="disabled", returns (False, message)
    - If no send_gate setting or defaults to "enabled", returns (True, None)
    - Disabled platforms (enabled=False) are ignored
    - On config loading errors, defaults to allowing sends (fail-open)
    """
    if config is None:
        config = _get_gateway_config()

    if config is None:
        # Not in gateway context; allow sends
        return True, None

    try:
        platforms = getattr(config, "platforms", {})
        disabled_platforms = []

        for platform, platform_config in platforms.items():
            # Skip disabled platforms
            if not getattr(platform_config, "enabled", True):
                continue

            # Check send_gate setting
            extra = getattr(platform_config, "extra", {})
            send_gate = extra.get("send_gate", "enabled").lower()

            if send_gate == "disabled":
                platform_name = getattr(platform, "value", str(platform))
                disabled_platforms.append(platform_name)

        if disabled_platforms:
            # Build helpful error message
            platform_list = ", ".join(sorted(disabled_platforms))
            error_msg = (
                f"Send operations are disabled via send_gate configuration. "
                f"Disabled on: {platform_list}. "
                f"To re-enable sends, set platforms.<platform_name>.extra.send_gate "
                f"to 'enabled' (or remove the 'send_gate' setting from your config) "
                f"and restart the gateway."
            )
            logger.debug("API server blocking send request: %s", error_msg)
            return False, error_msg

    except Exception as e:
        logger.debug("Error checking send_gate config: %s; allowing sends", e)
        # On any error, allow sends (fail-open rather than blocking)
        return True, None

    return True, None

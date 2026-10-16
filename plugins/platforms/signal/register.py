"""Signal plugin registration: configured/validate checks and ``register``."""

import logging

from gateway.config import PlatformConfig
from gateway.platforms._shared import env_is_connected

# Absolute imports (see __init__.py): one canonical module per file.
from plugins.platforms.signal.adapter import (
    MAX_MESSAGE_LENGTH, SignalAdapter, check_signal_requirements, validate_signal_config)
from plugins.platforms.signal.setup import interactive_setup
from plugins.platforms.signal.standalone import standalone_send

logger = logging.getLogger(__name__)

_env_configured = env_is_connected("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT")


def _is_connected(config: PlatformConfig) -> bool:
    """URL + account from config/scoped env, else ``.env``: a plain CLI process never loads ``.env``
    into the environment, and the old static setup row read it, so a ``.env``-only profile stays
    "configured" in ``hermes gateway setup``."""
    return validate_signal_config(config) or _env_configured(config)


def _validate_config(config: PlatformConfig) -> bool:
    """``validate_config``: same warning the core instantiation path logged before the move."""
    if validate_signal_config(config):
        return True
    logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
    return False


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="signal", label="Signal", adapter_factory=SignalAdapter,
        check_fn=check_signal_requirements, validate_config=_validate_config,
        is_connected=_is_connected, required_env=["SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"],
        install_hint="Signal: runtime requirements not met", setup_fn=interactive_setup,
        allowed_users_env="SIGNAL_ALLOWED_USERS", allow_all_env="SIGNAL_ALLOW_ALL_USERS",
        cron_deliver_env_var="SIGNAL_HOME_CHANNEL",
        # Out-of-process sends speak raw JSON-RPC and bypass SignalAdapter.send()'s chunking, so
        # send_message splits at the adapter's limit through this field.
        standalone_sender_fn=standalone_send, max_message_length=MAX_MESSAGE_LENGTH,
        emoji="📡", pii_safe=True, allow_update_command=True)

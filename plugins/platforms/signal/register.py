"""Signal plugin registration: configured/validate checks and ``register``."""

import logging

from gateway.config import PlatformConfig
from gateway.platforms._shared import env_is_connected
from gateway.platforms._shared import config_connected_extra
from gateway.platforms.send_targets import NOT_EXPLICIT, StandaloneSendPolicy

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


# System-prompt platform hint (PlatformEntry.platform_hint); cached-prompt bytes, keep stable.
_PLATFORM_HINT = (
    "You are on Signal. Standard markdown (**bold**, *italic*, ~~strike~~, # headers, `code`) auto-converts to "
    "Signal formatting; bullets render as \u2022. No tables \u2014 use bullets or labeled lines. "
    "You can send files natively: write MEDIA:/absolute/path/to/file in your response. "
    "Images (.png, .jpg, .webp) send as photos, other files as documents; ![alt](url) sends as photos."
)
def _parse_send_target(ref):
    # "group:<id>" is a native group target; an empty id is not explicit.
    stripped = ref.strip()
    if not stripped.startswith("group:"):
        return None
    group_id = stripped[len("group:"):].strip()
    return (f"group:{group_id}", None) if group_id else NOT_EXPLICIT


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    from . import gateway_env  # env -> GatewayConfig bridge (gateway.config_env)
    ctx.register_platform(
        target_ref_syntax_fn=_parse_send_target,
        # Attachments ride the JSON-RPC param on the final chunk.
        standalone_send=StandaloneSendPolicy(label="Signal", media=True, empty_list_between_chunks=True,
                                             text_direct=True),
        name="signal", label="Signal", adapter_factory=SignalAdapter,
        dashboard_description="Connect through a signal-cli REST bridge.",
        docs_url="https://github.com/bbernhard/signal-cli-rest-api",
        dashboard_env_vars=("SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT", "SIGNAL_ALLOWED_USERS"),
        check_fn=check_signal_requirements, validate_config=_validate_config,
        is_connected=_is_connected, required_env=["SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"],
        config_connected_fn=config_connected_extra("http_url"),
        install_hint="Signal: runtime requirements not met", setup_fn=interactive_setup,
        cron_deliver_env_var="SIGNAL_HOME_CHANNEL",
        # Out-of-process sends speak raw JSON-RPC and bypass SignalAdapter.send()'s chunking, so
        # send_message splits at the adapter's limit through this field.
        standalone_sender_fn=standalone_send, max_message_length=MAX_MESSAGE_LENGTH,
        emoji="📡", pii_safe=True, allow_update_command=True,
        platform_hint=_PLATFORM_HINT,
        display_tier="low",
        env_overrides=gateway_env.ENV_OVERRIDES,
        env_enable_credentials=gateway_env.ENV_ENABLE_CREDENTIALS, **gateway_env.AUTHZ,
    )

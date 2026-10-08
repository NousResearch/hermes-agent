"""Keep the loopback api_server up while the cron provider fires through it.

A cron provider declaring ``fires_over_loopback`` (Chronos) delivers every fire to the dashboard,
which forwards it to this gateway's api_server at ``/api/cron/fire``. That listener is then
required infrastructure, not a user-facing messaging platform: a ``platforms.api_server.enabled:
false`` left by the dashboard toggle, or an ``API_SERVER_KEY`` lost in a ``.env`` rewrite, would
otherwise make every fire 503 forever while the gateway looks healthy.

:func:`ensure_cron_loopback_listener` runs once on the runner's loaded config (before adapters
connect). It force-enables the platform and, when no usable key exists, generates one and
persists it through ``save_env_value`` in the profile that owns the listener. Host/port are left
as configured (default bind is 127.0.0.1). The built-in ticker never needs it, so self-hosted
gateways keep honouring the user's disable.
"""
from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Optional

from gateway.config import GatewayConfig, Platform, PlatformConfig, _has_usable_api_server_key

logger = logging.getLogger(__name__)

_KEY_ENV = "API_SERVER_KEY"


def _operator_env_key(home: Path) -> Optional[str]:
    """``API_SERVER_KEY`` supplied by the process environment rather than the profile ``.env``
    (``docker run -e API_SERVER_KEY=…``). docker/stage2-hook.sh never generates over one, and
    neither do we: a generated ``.env`` key would shadow it (``.env`` loads with override)."""
    env_value = os.environ.get(_KEY_ENV)
    if not env_value:
        return None
    from agent.secret_scope import load_env_file

    file_value = load_env_file(home / ".env").get(_KEY_ENV)
    return env_value if env_value != file_value else None


def _ensure_key(platform_config: PlatformConfig, home: Path) -> bool:
    """Make sure the adapter has a usable key; True when one is in place."""
    extra = platform_config.extra
    if _has_usable_api_server_key(extra.get("key")):
        return True
    operator_key = _operator_env_key(home)
    if operator_key is not None:
        if _has_usable_api_server_key(operator_key):
            extra["key"] = operator_key
            return True
        logger.error(
            "Cron provider needs the loopback api_server, but the container-provided %s is shorter "
            "than 16 characters, so the listener cannot start and scheduled cron fires will fail. "
            "Not generating a replacement (it would shadow the operator value); set a strong key, "
            "e.g. `openssl rand -hex 32`.", _KEY_ENV)
        return False
    from hermes_cli.config import env_write_refusal, save_env_value

    refusal = env_write_refusal(_KEY_ENV, "set")
    if refusal:
        logger.error(
            "Cron provider needs the loopback api_server but no usable %s exists and .env is "
            "write-locked (%s); scheduled cron fires will fail until a key is provided.",
            _KEY_ENV, refusal)
        return False
    key = secrets.token_hex(32)
    try:
        save_env_value(_KEY_ENV, key)
    except Exception as exc:
        # Still start this run with the generated key; the next boot retries the write.
        logger.warning("Could not persist generated %s to %s: %s", _KEY_ENV, home / ".env", exc)
    extra["key"] = key
    logger.warning(
        "Generated a new %s in %s: the active cron provider fires through the loopback "
        "api_server and no usable key was configured.", _KEY_ENV, home / ".env")
    return True


def ensure_cron_loopback_listener(
    config: GatewayConfig, *, home: Path, home_count: Optional[int] = None,
) -> bool:
    """Force the api_server on in ``config`` when the active cron provider fires over loopback.

    Call inside the scope of the profile that owns the listener (``home``): the provider is
    resolved from that profile's config and a generated key lands in its ``.env``. Returns True
    when the platform is enabled because of this requirement.
    """
    from cron.loopback_fire import loopback_api_server_required

    if not loopback_api_server_required(home_count):
        return False
    platform_config = config.platforms.setdefault(Platform.API_SERVER, PlatformConfig(enabled=False))
    platform_config.extra.pop("_enabled_explicit", None)
    if not _ensure_key(platform_config, home):
        return False
    if not platform_config.enabled:
        logger.warning(
            "platforms.api_server is disabled in %s, but the active cron provider delivers "
            "scheduled fires over the loopback api_server; starting it anyway (it cannot be "
            "turned off while that provider is active).", home / "config.yaml")
        platform_config.enabled = True
    return True

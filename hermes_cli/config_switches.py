"""``config_switch``: the one boolean reader behind the switches a distribution turns off."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def config_switch(*keys: str, default: bool = True) -> bool:
    """A boolean feature switch at ``keys`` in the cached merged config (``load_config_readonly``).

    The one reader behind the switches a distribution turns off (``mcp.client``,
    ``mcp.stdio_servers``, ``terminal.external_backends``, ``gateway.platform_adapters``,
    ``voice.mode_enabled``, ``sessions.git_probe``): absent, ``null`` or an unreadable config is
    ``default`` (today's behaviour), a truthy word is on, anything else is off. Never raises."""
    from hermes_cli import config as _config
    from utils import is_truthy_value

    try:
        value = _config.cfg_get(_config.load_config_readonly(), *keys, default=default)
    except Exception:  # health: allow BLE001 -- a switch never raises; an unreadable config is the default
        logger.debug("config_switch %s: config unreadable; using default=%s", ".".join(keys), default,
                     exc_info=True)
        return default
    return is_truthy_value(value, default=default)

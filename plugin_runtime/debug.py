"""Shared plugin-runtime debug state and logging setup."""

from __future__ import annotations

import logging
import sys

from utils import env_var_enabled


_LOGGER_NAME = "hermes_cli.plugins"
_PLUGINS_DEBUG = env_var_enabled("HERMES_PLUGINS_DEBUG")
_DEBUG_HANDLER_INSTALLED = False


def plugin_debug_enabled() -> bool:
    """Return the process-local plugin debug flag."""
    return _PLUGINS_DEBUG


def refresh_plugin_debug() -> bool:
    """Refresh the plugin debug flag from the environment and return it."""
    global _PLUGINS_DEBUG
    _PLUGINS_DEBUG = env_var_enabled("HERMES_PLUGINS_DEBUG")
    return _PLUGINS_DEBUG


def install_plugin_debug_handler(force: bool = False) -> None:
    """Tee plugin-runtime logs to stderr at DEBUG once per process when enabled."""
    global _DEBUG_HANDLER_INSTALLED, _PLUGINS_DEBUG
    if force:
        _PLUGINS_DEBUG = refresh_plugin_debug()
    if not _PLUGINS_DEBUG or _DEBUG_HANDLER_INSTALLED:
        return

    logger = logging.getLogger(_LOGGER_NAME)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[plugins] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    _DEBUG_HANDLER_INSTALLED = True
    logger.debug("HERMES_PLUGINS_DEBUG=1 — verbose plugin discovery logging enabled")

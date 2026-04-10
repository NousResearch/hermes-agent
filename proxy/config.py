"""Credential proxy configuration helpers.

Reads ``credential_proxy`` section from config.yaml::

    credential_proxy:
      enabled: true
      socket: ~/.hermes/state/cred-proxy.sock   # optional, has default
      proxy_credentials:                          # env var names to passthrough
        - CLOUDFLARE_API_TOKEN
        - SLACK_BOT_TOKEN
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_SOCKET_NAME = "cred-proxy.sock"
_DEFAULT_PID_NAME = "cred-proxy.pid"


def _hermes_home() -> Path:
    """Resolve HERMES_HOME (profile-aware)."""
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def _load_config_section() -> dict:
    """Load the ``credential_proxy`` section from config.yaml."""
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not available, credential_proxy config disabled")
        return {}

    config_path = _hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("credential_proxy", {}) or {}
    except Exception as exc:
        logger.debug("Could not read credential_proxy config: %s", exc)
        return {}


def is_proxy_enabled() -> bool:
    """Check whether the credential proxy is enabled in config."""
    section = _load_config_section()
    return bool(section.get("enabled", False))


def get_proxy_socket_path() -> Path:
    """Return the resolved Unix socket path for the proxy.

    The path is profile-aware: ``{HERMES_HOME}/state/cred-proxy.sock``.
    """
    section = _load_config_section()
    custom = section.get("socket")
    if custom:
        path = Path(str(custom)).expanduser()
        if not path.is_absolute():
            path = _hermes_home() / "state" / path
        return path
    return _hermes_home() / "state" / _DEFAULT_SOCKET_NAME


def get_proxy_pid_path() -> Path:
    """Return the PID file path for the proxy daemon."""
    return _hermes_home() / "state" / _DEFAULT_PID_NAME


def get_proxy_credentials_list() -> List[str]:
    """Return env var names that should bypass the blocklist.

    These are vars whose values contain ``hermes-proxy://`` placeholders
    and must survive ``_sanitize_subprocess_env()``.
    """
    section = _load_config_section()
    raw = section.get("proxy_credentials", [])
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if isinstance(item, str) and item.strip()]

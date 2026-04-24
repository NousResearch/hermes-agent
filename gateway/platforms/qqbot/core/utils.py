# -*- coding: utf-8 -*-
"""QQBot shared utilities — User-Agent, HTTP headers, config coercion.

All functions are stateless and carry zero hermes dependencies.
``_get_hermes_version()`` uses ``importlib.metadata`` with a safe fallback,
so this module works outside the hermes environment.
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List

from .constants import QQBOT_VERSION


# ---------------------------------------------------------------------------
# User-Agent
# ---------------------------------------------------------------------------

def _get_hermes_version() -> str:
    """Return the hermes-agent package version, or ``'dev'`` if unavailable."""
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:
        return "dev"


def build_user_agent() -> str:
    """Build a descriptive User-Agent string.

    Format::

        QQBotAdapter/<qqbot_version> (Python/<py_version>; <os>; Hermes/<hermes_version>)

    Example::

        QQBotAdapter/1.1.0 (Python/3.11.9; darwin; Hermes/0.9.0)
    """
    py_ver = (
        f"{sys.version_info.major}"
        f".{sys.version_info.minor}"
        f".{sys.version_info.micro}"
    )
    os_name = platform.system().lower()
    hermes_ver = _get_hermes_version()
    return (
        f"QQBotAdapter/{QQBOT_VERSION}"
        f" (Python/{py_ver}; {os_name}; Hermes/{hermes_ver})"
    )


def get_api_headers() -> Dict[str, str]:
    """Return standard HTTP headers for QQBot API requests.

    ``q.qq.com`` requires ``Accept: application/json``; without it the server
    returns a JavaScript anti-bot challenge page.
    """
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": build_user_agent(),
    }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def coerce_list(value: Any) -> List[str]:
    """Coerce a config value into a trimmed, non-empty string list.

    Accepts comma-separated strings, lists, tuples, sets, or single values.

    :param value: Raw config value (any type).
    :returns: List of non-empty stripped strings.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    stripped = str(value).strip()
    return [stripped] if stripped else []

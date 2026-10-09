"""QQBot shared utilities — User-Agent, HTTP helpers, config coercion."""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, List

from .constants import QQBOT_VERSION


def _get_hermes_version() -> str:
    """Return the hermes-agent package version, or 'dev' if unavailable."""
    try:
        from importlib.metadata import version
        return version("hermes-agent")
    except Exception:
        return "dev"


def build_user_agent() -> str:
    """``QQBotAdapter/<qqbot_version> (Python/<py_version>; <os>; Hermes/<hermes_version>)``."""
    v = sys.version_info
    return (f"QQBotAdapter/{QQBOT_VERSION} (Python/{v.major}.{v.minor}.{v.micro}; "
            f"{platform.system().lower()}; Hermes/{_get_hermes_version()})")


def get_api_headers() -> Dict[str, str]:
    """Standard QQBot API headers. ``q.qq.com`` requires ``Accept: application/json``
    — without it the server returns a JavaScript anti-bot challenge page."""
    return {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": build_user_agent()}


def coerce_list(value: Any) -> List[str]:
    """Coerce a comma-separated string / list / tuple / set / scalar into a trimmed string list."""
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else value if isinstance(value, (list, tuple, set)) else [value]
    return [s for s in (str(item).strip() for item in items) if s]

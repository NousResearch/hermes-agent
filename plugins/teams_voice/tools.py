"""Agent-facing tools for the teams_voice bridge.

For the scaffold this is just a read-only status tool. The realtime call tools
(``look_at_screen``, ``show_to_caller``, ``post_meeting_minutes``) are surfaced
to the *realtime model* per-call by the dialogue handler, not registered here as
global agent tools — they only make sense inside an active call.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from .config import resolve_config

TEAMS_VOICE_STATUS_SCHEMA: Dict[str, Any] = {
    "name": "teams_voice_status",
    "description": (
        "Report the Microsoft Teams voice/video (CVI) bridge configuration and "
        "readiness: bind host/port, whether a shared secret is configured, and "
        "whether the aiohttp dependency is available. Does not reveal the secret."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def check_requirements() -> bool:
    """The bridge needs aiohttp (already a Hermes gateway dependency)."""
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return False
    return True


def handle_teams_voice_status(**_kwargs: Any) -> str:
    cfg = resolve_config()
    return json.dumps(
        {
            "ok": True,
            "configured": cfg.configured,  # bool — never the secret itself
            "host": cfg.host,
            "port": cfg.port,
            "path": cfg.path,
            "deps_available": check_requirements(),
        }
    )

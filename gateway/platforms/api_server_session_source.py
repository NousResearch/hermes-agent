"""Session ``source`` labels on the API server's ``/api/sessions`` routes.

A session's stored ``source`` names the surface that owns it. A REST client owns the API server's
own sessions and the first-party clients' (``gateway.session_api.API_CLIENT_SOURCES``); every other
label (a CLI, messaging, Desktop or dashboard conversation) belongs to another surface.
"""
from typing import Any

from gateway.session_api import API_CLIENT_SOURCES, FIRST_PARTY_SESSION_SOURCES

# Accepted as a creation label (the pre-existing client contract), but never a REST client's to own.
_FOREIGN_SOURCES = frozenset({"cli", "telegram", "discord", "slack", "desktop", "dashboard"})
_KNOWN_SOURCES = API_CLIENT_SOURCES | _FOREIGN_SOURCES
# Extension-internal tags (bot mode, /bg background tasks) are Browser Extension sessions.
_BROWSER_EXTENSION_TAGS = frozenset({"browser", "hermes_bot_mode", "hermes_browser_bg"})
# Stale clients that predate an explicit ``source`` still send first-party session ids.
_ID_PREFIX_SOURCES = (("hermes-web-", "hermes_web"), ("hermes-browser-", "hermes_browser"))


def normalize_session_source(value: Any, *, default: str = "api_server", session_id: str = "") -> str:
    """Accept the small public source taxonomy used by first-party clients.

    Known sources pass through; an explicit unknown source normalizes to an empty string so callers
    can reject it (or pass ``default`` to coerce). When ``value`` is None, first-party sources are
    inferred from the session-id prefix (``hermes-web-*`` / ``hermes-browser-*``) so stale clients
    that predate explicit ``source`` still group correctly; otherwise *default* is returned.
    """
    if value is None:
        normalized_id = str(session_id or "").strip().lower()
        for prefix, label in _ID_PREFIX_SOURCES:
            if normalized_id.startswith(prefix):
                return label
        return default
    text = str(value).strip().lower()
    if text in _BROWSER_EXTENSION_TAGS:
        return "hermes_browser"
    return text if text in _KNOWN_SOURCES else ""


def client_label(stored: Any) -> str:
    """The label a row the API server derives from another row carries: a first-party client's
    (``hermes_web`` ...) carries over, every other label becomes ``api_server``."""
    return stored if stored in FIRST_PARTY_SESSION_SOURCES else "api_server"

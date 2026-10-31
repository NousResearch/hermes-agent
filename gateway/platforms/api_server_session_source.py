"""Session ``source`` labels on the API server's ``/api/sessions`` routes.

A session's stored ``source`` names the surface that owns it. A REST client owns the API server's
own sessions and the first-party clients' (``gateway.session_api.API_CLIENT_SOURCES``); every other
label (a CLI, messaging, Desktop or dashboard conversation) belongs to another surface, so a REST
client may neither take it over by relabelling it nor relabel a session onto one.
"""
import asyncio
from functools import partial
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


_PATCH_FIELDS = frozenset({"title", "end_reason", "source", "pinned", "archived", "hidden", "unread"})
# pinned/archived/unread are durable desktop-sidebar flags.
_PATCH_FLAGS = ("pinned", "archived", "hidden", "unread")


class _SourceNotOwned(Exception):
    """The session's stored source is another surface's: the PATCH changes nothing."""


def patch_refusal(body: dict[str, Any]):
    """The error response for a PATCH body that is malformed or asks for a source a REST client may not
    set, else None. Pure: it reads no session, so nothing has been written when it answers."""
    from gateway.platforms.api_server import _error_response
    if unknown := sorted(set(body) - _PATCH_FIELDS):
        return _error_response(
            f"Unsupported session fields: {', '.join(unknown)}", 400, code="unsupported_session_field")
    for flag in _PATCH_FLAGS:
        if flag in body and not isinstance(body[flag], bool):
            return _error_response(f"'{flag}' must be a boolean", 400, code="invalid_session_field")
    if "source" in body:
        target = normalize_session_source(body["source"], default="")
        if not target:
            return _error_response("Invalid session source", 400, code="invalid_session_source")
        if target not in API_CLIENT_SOURCES:
            return _error_response(
                f"A REST client may set a session source only to one of: {', '.join(sorted(API_CLIENT_SOURCES))}",
                403, code="permission_denied")
    return None


def _write_title_and_source(db, session_id, body, conn):
    """The PATCH's two refusable writes in the caller's transaction, every check before the first
    write: a session another surface owns keeps its source, and a title the database refuses (too
    long, held by another session) leaves the source as it was."""
    row = conn.execute("SELECT source FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if "source" in body and row is not None and row["source"] not in API_CLIENT_SOURCES:
        raise _SourceNotOwned(row["source"])
    if "title" in body:
        db._set_session_title_in_transaction(
            conn, session_id, "" if body["title"] is None else str(body["title"]), source=db.TITLE_SOURCE_USER)
    if "source" in body:
        conn.execute("UPDATE sessions SET source = ? WHERE id = ?",
                     (normalize_session_source(body["source"], default=""), session_id))


async def apply_session_patch(db, session_id: str, body: dict[str, Any]):
    """Write a PATCH that ``patch_refusal`` passed: None, or the refusal response with nothing written.
    The title and the source commit together or not at all; the flags and ``end_reason`` follow, and
    nothing about them can be refused."""
    from gateway.platforms.api_server import _error_response
    if "title" in body or "source" in body:
        try:
            await asyncio.to_thread(db._execute_write, partial(_write_title_and_source, db, session_id, body))
        except ValueError as exc:
            return _error_response(str(exc), 400, code="invalid_title")
        except _SourceNotOwned as exc:
            return _error_response(
                f"Session source '{exc}' belongs to another surface; a REST client cannot relabel it",
                403, code="permission_denied")
    # Pinned last: set_session_pinned clears hidden, so a pin in the same request
    # wins over an explicit hidden (same order as the dashboard's _RENAME_FLAG_SETTERS).
    for flag, setter in (("archived", db.set_session_archived), ("hidden", db.set_session_hidden),
                         ("pinned", db.set_session_pinned)):
        if flag in body:
            await asyncio.to_thread(setter, session_id, body[flag])
    if "unread" in body:
        await asyncio.to_thread(db.set_session_read, session_id, read=not body["unread"])
    if body.get("end_reason"):
        await asyncio.to_thread(db.end_session, session_id, str(body["end_reason"]))
    return None

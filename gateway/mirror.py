"""Queue outbound context without mutating destination conversation history."""

import json
import logging
from pathlib import Path
from typing import Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"
_SESSIONS_INDEX_AT_IMPORT = _SESSIONS_INDEX


def _resolve_sessions_index() -> Path:
    """Active profile's ``sessions.json`` at call time: the patched ``_SESSIONS_INDEX`` when a test
    changed it, else live profile-scoped HERMES_HOME — under the multiplexed gateway one process
    serves every profile, so the import-time constant would resolve every profile's pre-migration
    session lookup against the launch profile's index."""
    return (_SESSIONS_INDEX if _SESSIONS_INDEX != _SESSIONS_INDEX_AT_IMPORT
            else get_hermes_home() / "sessions" / "sessions.json")


def _origin_user_id(entry: dict) -> str:
    return str((entry.get("origin") or {}).get("user_id") or "")


def mirror_to_session(
    platform: str, chat_id: str, message_text: str, source_label: str = "cli", thread_id: Optional[str] = None,
    user_id: Optional[str] = None, role: str = "assistant", session_id: Optional[str] = None,
) -> bool:
    """Queue confirmed delivery context for the destination's next turn.

    Never append to or rewrite its live transcript. Destination keys also cover
    chats that have not started a session yet; explicit session seeds retain IDs.
    """
    try:
        from agent.outbound_context import enqueue, destination_key
        target = session_id or destination_key(platform, chat_id, thread_id)
        enqueue(target, message_text, source_label, user_id=None if session_id else user_id)
        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True
    except Exception as e:
        # WARNING, not debug: a silent mirror drop is the cron continuation-amnesia bug.
        logger.warning("Mirror failed for %s:%s thread=%s user=%s session=%s: %s", platform, chat_id, thread_id, user_id, session_id, e)
        return False


def _find_session_id(platform: str, chat_id: str, thread_id: Optional[str] = None, user_id: Optional[str] = None) -> Optional[str]:
    """Active session_id for a platform + chat_id pair.

    state.db is primary; sessions.json is the pre-migration fallback.  DM keys
    don't embed the chat_id ("agent:main:telegram:dm"), so match on the persisted
    origin.  With *user_id*, exact sender matches win; several same-chat candidates
    with no user match → None rather than contaminate another participant's session.

    Queries state.db gateway session rows (primary source since #9006); falls back to scanning sessions.json
    for pre-migration databases.
    """
    try:
        from hermes_state_registry import acquire, release_or_close
        db = acquire()
        try:
            finder = getattr(db, "find_session_by_origin", None)
            session_id = finder(platform=platform, chat_id=chat_id, thread_id=thread_id, user_id=user_id) if callable(finder) else None
            if session_id:
                return str(session_id)
        finally:
            release_or_close(db)
    except Exception as e:
        logger.debug("Mirror state.db session lookup failed: %s", e)

    sessions_index = _resolve_sessions_index()
    if not sessions_index.exists():
        return None
    try:
        data = json.loads(sessions_index.read_text(encoding="utf-8-sig"))
    except Exception:
        return None

    def _matches(entry: dict) -> bool:
        origin = entry.get("origin") or {}
        return ((origin.get("platform") or entry.get("platform", "")).lower() == platform.lower()
                and str(origin.get("chat_id", "")) == str(chat_id)
                and (thread_id is None or str(origin.get("thread_id") or "") == str(thread_id)))

    # Keys starting with "_" (e.g. the gateway's "_README") are metadata sentinels.
    candidates = [e for k, e in data.items() if not str(k).startswith("_") and isinstance(e, dict) and _matches(e)]
    if not candidates:
        return None
    if user_id:
        exact_user_matches = [e for e in candidates if _origin_user_id(e) == str(user_id)]
        if exact_user_matches:
            candidates = exact_user_matches
        elif len(candidates) > 1:
            return None
    elif len(candidates) > 1 and len({u.strip() for u in map(_origin_user_id, candidates) if u.strip()}) > 1:
        return None
    return max(candidates, key=lambda entry: entry.get("updated_at", "")).get("session_id")

"""
Session mirroring for cross-platform message delivery.

When a message is sent to a platform (via send_message or cron delivery),
this module appends a "delivery-mirror" record to the target session's
transcript so the receiving-side agent has context about what was sent.

Standalone -- works from CLI, cron, and gateway contexts without needing
the full SessionStore machinery.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"


def mirror_to_session(
    platform: str,
    chat_id: str,
    message_text: str,
    source_label: str = "cli",
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    role: str = "assistant",
) -> bool:
    """
    Append a delivery-mirror message to the target session's transcript.

    Finds the gateway session that matches the given platform + chat_id,
    then writes a mirror entry to both the JSONL transcript and SQLite DB.

    ``role`` defaults to ``"assistant"`` — correct for the interactive
    ``send_message`` mirror, where the mirrored text is the agent's own
    outgoing reply (a genuine assistant turn). Callers mirroring text that is
    NOT the agent speaking — e.g. a cron brief delivered out-of-band — must
    pass ``role="user"``: the ``mirror``/``mirror_source`` metadata is dropped
    at the SQLite boundary (only role+content persist), so on replay an
    assistant-role mirror is indistinguishable from a real assistant turn and
    produces ``assistant → assistant`` pairs that break strict-alternation
    providers (issue #2221). A user-role mirror collapses safely via
    ``repair_message_sequence``'s consecutive-user merge on every provider.

    Returns True if mirrored successfully, False if no matching session or error.
    All errors are caught -- this is never fatal.
    """
    try:
        session_id = _find_session_id(
            platform,
            str(chat_id),
            thread_id=thread_id,
            user_id=user_id,
        )
        if not session_id:
            logger.debug(
                "Mirror: no session found for %s:%s:%s:%s",
                platform,
                chat_id,
                thread_id,
                user_id,
            )
            return False

        mirror_msg = {
            "role": role,
            "content": message_text,
            "timestamp": datetime.now().isoformat(),
            "mirror": True,
            "mirror_source": source_label,
        }

        _append_to_sqlite(session_id, mirror_msg)

        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True

    except Exception as e:
        logger.debug(
            "Mirror failed for %s:%s:%s:%s: %s",
            platform,
            chat_id,
            thread_id,
            user_id,
            e,
        )
        return False


def _find_session_id(
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """
    Find the active session_id for a platform + chat_id pair.

    Queries state.db gateway session rows (primary source since #9006);
    falls back to scanning sessions.json for pre-migration databases.
    DM session keys don't embed the chat_id (e.g. "agent:main:telegram:dm"),
    so we match on the persisted chat origin, not the key.

    When *user_id* is provided, prefer exact sender matches. If multiple
    same-chat candidates exist and none matches the user, return None instead
    of guessing and contaminating another participant's session.
    """
    # Primary: state.db
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            finder = getattr(db, "find_session_by_origin", None)
            if callable(finder):
                session_id = finder(
                    platform=platform,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    user_id=user_id,
                )
                if session_id:
                    return str(session_id)
        finally:
            db.close()
    except Exception as e:
        logger.debug("Mirror state.db session lookup failed: %s", e)

    # Fallback: sessions.json (pre-migration databases)
    if not _SESSIONS_INDEX.exists():
        return None

    try:
        with open(_SESSIONS_INDEX, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    platform_lower = platform.lower()
    candidates = []

    for _key, entry in data.items():
        # Skip documentation/metadata sentinels (keys starting with "_", e.g.
        # the gateway's "_README" note) — they are not session entries.
        if str(_key).startswith("_") or not isinstance(entry, dict):
            continue
        origin = entry.get("origin") or {}
        entry_platform = (origin.get("platform") or entry.get("platform", "")).lower()

        if entry_platform != platform_lower:
            continue

        origin_chat_id = str(origin.get("chat_id", ""))
        if origin_chat_id == str(chat_id):
            origin_thread_id = origin.get("thread_id")
            if thread_id is not None and str(origin_thread_id or "") != str(thread_id):
                continue
            candidates.append(entry)

    if not candidates:
        return None

    if user_id:
        exact_user_matches = [
            entry for entry in candidates
            if str((entry.get("origin") or {}).get("user_id") or "") == str(user_id)
        ]
        if exact_user_matches:
            candidates = exact_user_matches
        elif len(candidates) > 1:
            return None
    elif len(candidates) > 1:
        distinct_user_ids = {
            str((entry.get("origin") or {}).get("user_id") or "").strip()
            for entry in candidates
            if str((entry.get("origin") or {}).get("user_id") or "").strip()
        }
        if len(distinct_user_ids) > 1:
            return None

    best_entry = max(candidates, key=lambda entry: entry.get("updated_at", ""))
    return best_entry.get("session_id")



def _append_to_sqlite(session_id: str, message: dict) -> None:
    """Append a message to the SQLite session database."""
    db = None
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        db.append_message(
            session_id=session_id,
            role=message.get("role", "assistant"),
            content=message.get("content"),
        )
    except Exception as e:
        logger.debug("Mirror SQLite write failed: %s", e)
    finally:
        if db is not None:
            db.close()

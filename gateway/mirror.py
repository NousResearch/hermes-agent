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
from typing import Any, Dict, List, Optional

from hermes_cli.config import get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"
_PENDING_MIRRORS_PATH = _SESSIONS_DIR / "pending_mirrors.json"


def mirror_to_session(
    platform: str,
    chat_id: str,
    message_text: str,
    source_label: str = "cli",
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> bool:
    """
    Append a delivery-mirror message to the target session's transcript.

    Finds the gateway session that matches the given platform + chat_id,
    then writes a mirror entry to both the JSONL transcript and SQLite DB.

    Returns True if mirrored immediately, False if it was queued or failed.
    All errors are caught -- this is never fatal.
    """
    try:
        mirror_msg = {
            "role": "assistant",
            "content": message_text,
            "timestamp": datetime.now().isoformat(),
            "mirror": True,
            "mirror_source": source_label,
        }

        session_id = _find_session_id(
            platform,
            str(chat_id),
            thread_id=thread_id,
            user_id=user_id,
        )
        if not session_id:
            _queue_pending_mirror(
                platform,
                str(chat_id),
                mirror_msg,
                thread_id=thread_id,
            )
            logger.debug(
                "Mirror: queued pending entry for %s:%s:%s:%s",
                platform,
                chat_id,
                thread_id,
                user_id,
            )
            return False

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

    Scans sessions.json entries and matches where origin.chat_id == chat_id
    on the right platform.  DM session keys don't embed the chat_id
    (e.g. "agent:main:telegram:dm"), so we check the origin dict.

    When *user_id* is provided, prefer exact sender matches. If multiple
    same-chat candidates exist and none matches the user, return None instead
    of guessing and contaminating another participant's session.
    """
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


def drain_pending_mirrors(
    session_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> int:
    """Replay queued mirror messages into a newly started session."""
    key = _pending_mirror_key(platform, chat_id, thread_id=thread_id)
    pending = _load_pending_mirrors()
    messages = pending.pop(key, [])
    if not messages:
        return 0

    _save_pending_mirrors(pending)

    drained = 0
    for message in messages:
        try:
            _append_to_sqlite(session_id, message)
            drained += 1
        except Exception as e:
            logger.debug(
                "Mirror pending replay failed for %s into %s: %s",
                key,
                session_id,
                e,
            )

    return drained


def _pending_mirror_key(
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> str:
    return f"{platform.lower()}::{chat_id}::{thread_id or ''}"


def _load_pending_mirrors() -> Dict[str, List[Dict[str, Any]]]:
    if not _PENDING_MIRRORS_PATH.exists():
        return {}

    try:
        with open(_PENDING_MIRRORS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        logger.debug("Mirror pending load failed: %s", e)

    return {}


def _save_pending_mirrors(data: Dict[str, List[Dict[str, Any]]]) -> None:
    atomic_json_write(_PENDING_MIRRORS_PATH, data)


def _queue_pending_mirror(
    platform: str,
    chat_id: str,
    message: Dict[str, Any],
    thread_id: Optional[str] = None,
) -> None:
    key = _pending_mirror_key(platform, chat_id, thread_id=thread_id)
    pending = _load_pending_mirrors()
    pending.setdefault(key, []).append(message)
    _save_pending_mirrors(pending)



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

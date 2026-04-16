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
from gateway.session import (
    find_session_entry_by_origin,
    touch_session_updated_at,
)

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"


def mirror_to_session(
    platform: str,
    chat_id: str,
    message_text: str,
    source_label: str = "cli",
    thread_id: Optional[str] = None,
) -> bool:
    """
    Append a delivery-mirror message to the target session's transcript.

    Finds the gateway session that matches the given platform + chat_id,
    then writes a mirror entry to both the JSONL transcript and SQLite DB.

    Returns True if mirrored successfully, False if no matching session or error.
    All errors are caught -- this is never fatal.
    """
    try:
        timestamp = datetime.now().isoformat()
        session_key, session_entry = find_session_entry_by_origin(
            platform,
            str(chat_id),
            thread_id=thread_id,
        )
        if not session_entry:
            logger.debug("Mirror: no session found for %s:%s:%s", platform, chat_id, thread_id)
            return False
        session_id = session_entry.get("session_id")
        if not session_id:
            logger.debug("Mirror: matching session entry is missing session_id for %s:%s:%s", platform, chat_id, thread_id)
            return False

        mirror_msg = {
            "role": "assistant",
            "content": message_text,
            "timestamp": timestamp,
            "mirror": True,
            "mirror_source": source_label,
        }

        _append_to_jsonl(session_id, mirror_msg)
        _append_to_sqlite(session_id, mirror_msg)
        if session_key is not None:
            touch_session_updated_at(session_key, updated_at=timestamp)

        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True

    except Exception as e:
        logger.debug("Mirror failed for %s:%s:%s: %s", platform, chat_id, thread_id, e)
        return False


def _find_session_id(platform: str, chat_id: str, thread_id: Optional[str] = None) -> Optional[str]:
    """Find the active session_id for a platform + chat_id pair."""
    _session_key, session_entry = find_session_entry_by_origin(
        platform,
        chat_id,
        thread_id=thread_id,
    )
    if not session_entry:
        return None
    return session_entry.get("session_id")


def _append_to_jsonl(session_id: str, message: dict) -> None:
    """Append a message to the JSONL transcript file."""
    transcript_path = _SESSIONS_DIR / f"{session_id}.jsonl"
    try:
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Mirror JSONL write failed: %s", e)


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

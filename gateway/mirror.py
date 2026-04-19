"""Session mirroring for cross-platform message delivery.

When a message is sent to a platform (via send_message or cron delivery),
this module appends a "delivery-mirror" record to the target session's
transcript so the receiving-side agent has context about what was sent.

Standalone -- works from CLI, cron, and gateway contexts without needing
the full SessionStore machinery, but delegates transcript persistence to
the canonical helpers in ``gateway.session``.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from hermes_constants import get_hermes_home

from gateway.session import append_transcript_message, find_matching_session_id

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"


def mirror_to_session(
    platform: str,
    chat_id: str,
    message_text: str,
    source_label: str = "cli",
    thread_id: Optional[str] = None,
    db: Any = None,
) -> bool:
    """
    Append a delivery-mirror message to the target session's transcript.

    Finds the gateway session that matches the given platform + chat_id,
    then persists the mirror entry through the canonical gateway session
    storage path.

    Returns True if mirrored successfully, False if no matching session or error.
    All errors are caught -- this is never fatal.
    """
    try:
        session_id = _find_session_id(platform, str(chat_id), thread_id=thread_id)
        if not session_id:
            logger.debug("Mirror: no session found for %s:%s:%s", platform, chat_id, thread_id)
            return False

        mirror_msg = {
            "role": "assistant",
            "content": message_text,
            "timestamp": datetime.now().isoformat(),
            "mirror": True,
            "mirror_source": source_label,
        }

        _append_to_transcript(session_id, mirror_msg, db=db)

        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True

    except Exception as e:
        logger.debug("Mirror failed for %s:%s:%s: %s", platform, chat_id, thread_id, e)
        return False


def _find_session_id(platform: str, chat_id: str, thread_id: Optional[str] = None) -> Optional[str]:
    """Resolve a delivery target to the latest matching gateway session."""
    return find_matching_session_id(
        _SESSIONS_DIR,
        platform,
        str(chat_id),
        thread_id=thread_id,
    )


def _append_to_transcript(session_id: str, message: dict, db: Any = None) -> None:
    """Append a mirror message through the canonical gateway transcript path."""
    owns_db = db is None
    if owns_db:
        try:
            from hermes_state import SessionDB
            db = SessionDB()
        except Exception as e:
            logger.debug("Mirror SQLite unavailable: %s", e)

    try:
        append_transcript_message(_SESSIONS_DIR, db, session_id, message)
    except Exception as e:
        logger.debug("Mirror transcript write failed: %s", e)
    finally:
        if owns_db and db is not None:
            db.close()

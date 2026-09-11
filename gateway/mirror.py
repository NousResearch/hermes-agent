"""Session mirroring for cross-platform message delivery.

When a message is sent to a platform (send_message or cron delivery), append a
"delivery-mirror" record to the target session's transcript so the receiving-side
agent knows what was sent.  Standalone: works from CLI, cron and gateway contexts.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"


def _origin_user_id(entry: dict) -> str:
    return str((entry.get("origin") or {}).get("user_id") or "")


def mirror_to_session(
    platform: str, chat_id: str, message_text: str, source_label: str = "cli", thread_id: Optional[str] = None,
    user_id: Optional[str] = None, role: str = "assistant", session_id: Optional[str] = None,
) -> bool:
    """Append a delivery-mirror message to the target session's SQLite transcript.

    Pass ``session_id`` when the caller already holds the exact session (e.g. the
    cron in_channel seed) to skip the origin scan, which refuses to guess on a
    populated chat (flat + N thread sessions per chat_id) and would drop the mirror.
    Text that is NOT the agent speaking (e.g. a cron brief) must pass
    ``role="user"``: ``mirror`` metadata is dropped at the SQLite boundary, so an
    assistant-role mirror replays as a real turn and yields assistant→assistant
    pairs that break strict-alternation providers, while a user-role mirror
    collapses safely via the consecutive-user merge.
    Returns True if mirrored, False if no matching session or error. Never raises.

    ``role`` defaults to ``"assistant"`` — correct for the interactive ``send_message`` mirror, where the
    mirrored text is the agent's own outgoing reply (a genuine assistant turn). See #2221.
    """
    try:
        if not session_id:
            session_id = _find_session_id(platform, str(chat_id), thread_id=thread_id, user_id=user_id)
        if not session_id:
            logger.warning(
                "Mirror: no session found for %s:%s thread=%s user=%s (explicit_id=none, origin-scan bailed)",
                platform, chat_id, thread_id, user_id,
            )
            return False
        _append_to_sqlite(session_id, {
            "role": role, "content": message_text, "timestamp": datetime.now().isoformat(),
            "mirror": True, "mirror_source": source_label,
        })
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

    if not _SESSIONS_INDEX.exists():
        return None
    try:
        data = json.loads(_SESSIONS_INDEX.read_text(encoding="utf-8"))
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


def _append_to_sqlite(session_id: str, message: dict) -> None:
    """Append a message to the SQLite session database."""
    try:
        from hermes_state_registry import acquire, release_or_close

        db = acquire()
        try:
            db.append_message(session_id=session_id, role=message.get("role", "assistant"), content=message.get("content"))
        finally:
            release_or_close(db)
    except Exception as e:
        logger.debug("Mirror SQLite write failed: %s", e)

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


def ensure_outbound_session(
    platform: str,
    chat_id: str,
    thread_id: str,
    chat_type: Optional[str] = None,
) -> bool:
    """Pre-create a gateway session entry for a thread spawned by outbound send.

    When the agent sends a message that starts a new thread, the session does
    not exist yet — it is normally created on the first *inbound* message.
    Calling this immediately after send, with the thread root event ID as
    ``thread_id``, creates the sessions.json entry so the subsequent
    ``mirror_to_session`` call can find it and write the outbound content as
    the first assistant message.

    Uses the live gateway runner's SessionStore so the session key is
    computed by exactly the same rules as inbound processing.  Returns True
    if the session was created or already existed, False if the runner is
    not available (CLI / cron without gateway) — callers treat False as a
    non-fatal no-op.

    ``chat_type`` defaults to ``"group"`` when not provided.  When omitted,
    the function reads the current session's ``HERMES_SESSION_KEY`` context
    variable (set by the gateway inbound path) so the spawned session key
    matches the inbound routing exactly.
    """
    if not thread_id or not chat_id or not platform:
        return False
    try:
        from gateway.run import _gateway_runner_ref
        from gateway.session import SessionSource
        from gateway.config import Platform

        runner = _gateway_runner_ref()
        if runner is None:
            return False

        plat = Platform(platform)
        adapter = runner.adapters.get(plat)

        resolved_chat_type = chat_type
        if resolved_chat_type is None:
            try:
                from gateway.session_context import get_session_env
                session_key = get_session_env("HERMES_SESSION_KEY", "")
                parts = session_key.split(":")
                if len(parts) >= 4 and parts[3] in ("dm", "group", "room"):
                    resolved_chat_type = parts[3]
            except Exception:
                pass
            if resolved_chat_type is None:
                resolved_chat_type = "group"

        source = SessionSource(
            platform=plat,
            chat_id=str(chat_id),
            chat_type=resolved_chat_type,
            user_id=None,
            thread_id=str(thread_id),
        )
        runner.session_store.get_or_create_session(source)
        try:
            if adapter is not None and hasattr(adapter, "_threads"):
                adapter._threads.mark(str(thread_id))
                logger.debug(
                    "Mirror: seeded _threads for outbound thread %s on %s",
                    thread_id, platform,
                )
        except Exception as _threads_exc:
            logger.debug(
                "Mirror: _threads seeding failed (non-fatal): %s", _threads_exc
            )
        logger.debug(
            "Mirror: pre-created outbound thread session %s:%s:%s",
            platform, chat_id, thread_id,
        )
        return True
    except Exception as exc:
        logger.debug("Mirror: ensure_outbound_session failed: %s", exc)
        return False


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
            "role": "assistant",
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
            if thread_id is not None:
                if str(origin_thread_id or "") != str(thread_id):
                    continue
            else:
                if origin_thread_id:
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
            mirror=message.get("mirror", False),
            mirror_source=message.get("mirror_source"),
        )
    except Exception as e:
        logger.debug("Mirror SQLite write failed: %s", e)
    finally:
        if db is not None:
            db.close()

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
from typing import Any, Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"

DELIVERY_EVENT_ROLE = "delivery"
DELIVERY_EVENT_TYPE = "delivery_mirror"


def build_mirror_message(
    platform: str,
    chat_id: str,
    message_text: str,
    *,
    session_id: Optional[str] = None,
    source_label: str = "cli",
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    source_platform: Optional[str] = None,
    source_chat_id: Optional[str] = None,
    source_chat_name: Optional[str] = None,
    source_user_id: Optional[str] = None,
    source_user_name: Optional[str] = None,
    source_session_key: Optional[str] = None,
) -> dict[str, Any]:
    """Build the durable transcript record for a cross-chat delivery.

    The record is deliberately *not* an ``assistant`` message. It represents an
    external delivery event that happened in another chat/session, and keeping it
    typed as such prevents future turns from mistaking it for a local assistant
    reply to the recipient.
    """

    source = {
        "label": source_label or source_platform or "unknown",
        "platform": source_platform or source_label or "",
        "chat_id": str(source_chat_id or ""),
        "chat_name": str(source_chat_name or ""),
        "user_id": str(source_user_id or ""),
        "user_name": str(source_user_name or ""),
        "session_key": str(source_session_key or ""),
    }
    target = {
        "platform": str(platform or ""),
        "chat_id": str(chat_id or ""),
        "thread_id": str(thread_id or ""),
        "user_id": str(user_id or ""),
        "session_id": str(session_id or ""),
    }
    return {
        "role": DELIVERY_EVENT_ROLE,
        "content": message_text,
        "timestamp": datetime.now().isoformat(),
        "event_type": DELIVERY_EVENT_TYPE,
        "mirror": True,
        "mirror_source": source["label"],
        "delivery": {
            "source": source,
            "target": target,
        },
    }


def _compact_identity(*parts: Any) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return ", ".join(out)


def mirror_to_agent_history_entry(message: dict[str, Any]) -> dict[str, str]:
    """Convert a durable delivery event into safe model-facing context.

    Model providers do not understand the internal ``delivery`` transcript role,
    and treating a cross-chat delivery as ``assistant`` caused contamination: the
    receiving session appeared to have authored the delivered text. Replay it as
    a clearly-labelled system note instead.
    """

    content = str(message.get("content") or "")
    delivery = message.get("delivery") if isinstance(message.get("delivery"), dict) else {}
    source = delivery.get("source") if isinstance(delivery.get("source"), dict) else {}

    label = source.get("label") or message.get("mirror_source") or "another session"
    source_user = source.get("user_name") or source.get("user_id")
    source_chat = source.get("chat_name") or source.get("chat_id")
    source_platform = source.get("platform")
    origin = _compact_identity(source_user, source_chat, source_platform, label) or str(label)

    return {
        "role": "system",
        "content": (
            f"[External delivery from {origin}: this message was sent into "
            "this chat by the delivery/messaging tool. It is context only, "
            "not a reply authored by this session.]\n"
            f"{content}"
        ),
    }


def mirror_to_session(
    platform: str,
    chat_id: str,
    message_text: str,
    source_label: str = "cli",
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    source_platform: Optional[str] = None,
    source_chat_id: Optional[str] = None,
    source_chat_name: Optional[str] = None,
    source_user_id: Optional[str] = None,
    source_user_name: Optional[str] = None,
    source_session_key: Optional[str] = None,
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

        mirror_msg = build_mirror_message(
            platform,
            str(chat_id),
            message_text,
            session_id=session_id,
            source_label=source_label,
            thread_id=thread_id,
            user_id=user_id,
            source_platform=source_platform,
            source_chat_id=source_chat_id,
            source_chat_name=source_chat_name,
            source_user_id=source_user_id,
            source_user_name=source_user_name,
            source_session_key=source_session_key,
        )

        _append_to_jsonl(session_id, mirror_msg)
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
        metadata = {
            "mirror": message.get("mirror") is True,
            "mirror_source": message.get("mirror_source"),
            "delivery": message.get("delivery") or {},
        }
        db.append_message(
            session_id=session_id,
            role=message.get("role", DELIVERY_EVENT_ROLE),
            content=message.get("content"),
            event_type=message.get("event_type"),
            metadata=metadata,
        )
    except Exception as e:
        logger.debug("Mirror SQLite write failed: %s", e)
    finally:
        if db is not None:
            db.close()

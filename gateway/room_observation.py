"""Platform-neutral persistence helpers for ambient room observations."""

import dataclasses
import logging
from datetime import datetime, timedelta
from typing import Any

from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


logger = logging.getLogger(__name__)
OBSERVED_ROOM_CONTEXT_LIMIT = 25
OBSERVED_ROOM_CONTEXT_MAX_AGE = timedelta(hours=6)
ROOM_OBSERVATION_CHAT_TYPE = "room_observation"


def room_scoped_source(source: SessionSource) -> SessionSource:
    """Return a source in the dedicated room-observation session namespace."""

    return dataclasses.replace(
        source,
        chat_type=ROOM_OBSERVATION_CHAT_TYPE,
        user_id=None,
        user_id_alt=None,
        user_name=None,
    )


def observed_message_row(event: MessageEvent) -> dict[str, Any]:
    """Build the attributed transcript row for one observed message."""

    source = event.source
    stable_user_id = source.user_id_alt or source.user_id or "unknown"
    display_name = source.user_name or stable_user_id
    return {
        "role": "user",
        "content": f"[{display_name}|{stable_user_id}]\n{event.text}",
        "timestamp": event.timestamp.isoformat(),
        "observed": True,
        "message_id": event.message_id,
    }


def persist_room_observation(session_store: Any, event: MessageEvent) -> str:
    """Append one observed row and return its room-scoped session id."""

    source = room_scoped_source(event.source)
    session = session_store.get_or_create_session(source)
    session_store.append_to_transcript(
        session.session_id,
        observed_message_row(event),
    )
    return session.session_id


def load_observed_room_context(
    session_store: Any,
    source: SessionSource,
    *,
    now: datetime | None = None,
    limit: int = OBSERVED_ROOM_CONTEXT_LIMIT,
    max_age: timedelta = OBSERVED_ROOM_CONTEXT_MAX_AGE,
) -> str | None:
    """Return bounded context from the existing room session, if one exists."""

    if source.chat_type == "dm" or not source.chat_id:
        return None
    loader = getattr(session_store, "load_recent_observed_messages", None)
    if not callable(loader):
        return None
    try:
        rows = loader(
            room_scoped_source(source),
            now=now,
            limit=limit,
            max_age=max_age,
        )
    except Exception:
        logger.warning(
            "Failed to load observed room context: platform=%s chat=%s thread=%s",
            source.platform.value,
            source.chat_id,
            source.thread_id or "none",
            exc_info=True,
        )
        return None
    if not isinstance(rows, list):
        return None
    contents = [
        row["content"].strip()
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("content"), str)
        and row["content"].strip()
    ]
    return "\n".join(contents) or None

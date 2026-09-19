"""Detached completed-turn handoff for opt-in memory providers.

Only the last user-led completed text turn is shared, already bounded and
detached from the live transcript. Providers must verify committed rows
before treating the payload as durable evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Bounds for the shared turn: message count, per-message text, serialized size.
_MAX_TAIL_MESSAGES = 128
_MAX_CONTENT_CHARS = 64_000
_MAX_SERIALIZED_CHARS = 1_000_000
# Timestamps are small scalars; cap string length so one field cannot eat the budget.
_MAX_TIMESTAMP_CHARS = 256

# Allowlisted message fields. ``timestamp`` is handled separately so only a
# valid scalar event time crosses the boundary; everything else is dropped.
_SAFE_FIELDS = (
    "id",
    "_row_id",
    "_db_persisted",
    "role",
    "content",
    "tool_name",
    "tool_call_id",
    "tool_calls",
    "session_id",
    "scope",
    "source_id",
    "source_event_id",
)


@dataclass(frozen=True)
class CompletedTurnSnapshot:
    """Immutable completed-turn payload decoded via :meth:`messages`."""

    session_id: str
    hermes_home: str
    payload: str

    def messages(self) -> List[Dict[str, Any]]:
        return json.loads(self.payload)


def _valid_timestamp(value: Any) -> bool:
    """True when ``value`` is a plausible event time worth preserving."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        text = value.strip()
        return bool(text) and len(value) <= _MAX_TIMESTAMP_CHARS
    return False


def _project_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Copy allowlisted fields plus a valid timestamp; drop everything else."""
    projected = {key: message[key] for key in _SAFE_FIELDS if key in message}
    stamp = message.get("timestamp")
    if stamp is not None and _valid_timestamp(stamp):
        projected["timestamp"] = stamp
    return projected


def _encode_bounded(selected: List[Dict[str, Any]]) -> Optional[str]:
    """Serialize ``selected`` or return None when it cannot fit safely."""
    chunks: List[str] = []
    size = 0
    try:
        encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False)
        for chunk in encoder.iterencode(selected):
            size += len(chunk)
            if size > _MAX_SERIALIZED_CHARS:
                return None
            chunks.append(chunk)
    except (ValueError, TypeError, RecursionError):
        return None
    return "".join(chunks)


def snapshot_completed_turn(
    messages: Any,
    *,
    session_id: str,
    user_content: str,
    assistant_content: str,
) -> Optional[CompletedTurnSnapshot]:
    """Build a bounded snapshot of the last completed text turn, or None."""
    from hermes_constants import get_hermes_home

    if not isinstance(messages, list) or not messages:
        return None
    tail = messages[-_MAX_TAIL_MESSAGES:]
    if any(type(item) is not dict for item in tail):
        return None
    start = next(
        (index for index in range(len(tail) - 1, -1, -1) if tail[index].get("role") == "user"),
        None,
    )
    if start is None:
        return None
    turn = tail[start:]
    last = turn[-1]
    if (
        turn[0].get("content") != user_content
        or last.get("role") != "assistant"
        or last.get("content") != assistant_content
        or last.get("tool_calls")
    ):
        return None
    for item in turn:
        content = item.get("content")
        if content is not None and (
            not isinstance(content, str) or len(content) > _MAX_CONTENT_CHARS
        ):
            return None
    payload = _encode_bounded([_project_message(item) for item in turn])
    if payload is None:
        return None
    return CompletedTurnSnapshot(session_id, str(get_hermes_home().resolve()), payload)

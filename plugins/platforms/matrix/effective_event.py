"""Resolve the visible state of a Matrix message from a server event."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from plugins.platforms.matrix.relations import MatrixRelation


@dataclass(frozen=True)
class MatrixEffectiveEvent:
    content: dict[str, Any] | None
    original_content: dict[str, Any]
    edited: bool = False
    redacted: bool = False
    error: dict[str, str] | None = None


def event_content(event: Any) -> dict[str, Any]:
    content = event.get("content") if isinstance(event, dict) else getattr(event, "content", None)
    if isinstance(content, dict):
        return content
    serialize = getattr(content, "serialize", None)
    if callable(serialize):
        try:
            result = serialize()
        except Exception:
            return {}
        if isinstance(result, dict):
            return result
    return {}


def _replacement(raw: dict[str, Any]) -> dict[str, Any] | None:
    unsigned = raw.get("unsigned")
    relations = unsigned.get("m.relations") if isinstance(unsigned, dict) else None
    replacement = relations.get("m.replace") if isinstance(relations, dict) else None
    return replacement if isinstance(replacement, dict) else None


async def _decrypt(client: Any, raw: dict[str, Any]) -> tuple[Any | None, dict[str, str] | None]:
    event_id = raw.get("event_id")
    crypto = getattr(client, "crypto", None)
    if crypto is None:
        return None, {"event_id": event_id, "error": "missing decryption keys"}
    try:
        from mautrix.types import Event

        event = await asyncio.wait_for(
            crypto.decrypt_megolm_event(Event.deserialize(raw)), timeout=10.0,
        )
    except Exception as exc:
        error = "missing decryption keys" if type(exc).__name__ == "SessionNotFound" else "decryption failed"
        return None, {"event_id": event_id, "error": error}
    if event is None:
        return None, {"event_id": event_id, "error": "missing decryption keys"}
    return event, None


async def effective_event(client: Any, raw: dict[str, Any]) -> MatrixEffectiveEvent:
    original_content = event_content(raw)
    unsigned = raw.get("unsigned")
    if isinstance(unsigned, dict) and unsigned.get("redacted_because"):
        return MatrixEffectiveEvent({}, original_content, redacted=True)

    event: Any = raw
    if raw.get("type") == "m.room.encrypted":
        event, error = await _decrypt(client, raw)
        if error is not None:
            return MatrixEffectiveEvent(None, original_content, error=error)
    content = event_content(event)

    replacement = _replacement(raw)
    if replacement is None or MatrixRelation.from_content(original_content.get("m.relates_to")).is_edit:
        return MatrixEffectiveEvent(content, original_content)
    relation = event_content(replacement).get("m.relates_to")
    if (
        replacement.get("room_id", raw.get("room_id")) != raw.get("room_id")
        or replacement.get("sender") != raw.get("sender")
        or replacement.get("type") != raw.get("type")
        or "state_key" in replacement or "state_key" in raw
        or not isinstance(relation, dict)
        or relation.get("rel_type") != "m.replace"
        or relation.get("event_id") != raw.get("event_id")
        or (isinstance(replacement.get("unsigned"), dict)
            and replacement["unsigned"].get("redacted_because"))
    ):
        return MatrixEffectiveEvent(content, original_content)

    if replacement.get("type") == "m.room.encrypted":
        revised, error = await _decrypt(client, replacement)
        if error is not None:
            return MatrixEffectiveEvent(content, original_content, error=error)
        revised_content = event_content(revised).get("m.new_content")
    else:
        revised_content = event_content(replacement).get("m.new_content")
    if not isinstance(revised_content, dict):
        return MatrixEffectiveEvent(content, original_content)

    original_relation = content.get("m.relates_to")
    content = {key: value for key, value in revised_content.items() if key != "m.relates_to"}
    if original_relation is not None:
        content["m.relates_to"] = original_relation
    return MatrixEffectiveEvent(content, original_content, edited=True)

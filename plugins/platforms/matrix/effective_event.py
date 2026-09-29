"""Resolve the visible state of a Matrix message from a server event."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Any, Callable

from plugins.platforms.matrix.relations import MatrixRelation


@dataclass(frozen=True)
class MatrixEffectiveEvent:
    content: dict[str, Any] | None
    original_content: dict[str, Any]
    edited: bool = False
    redacted: bool = False
    error: dict[str, str] | None = None
    replacement_id: str | None = None
    event_type: str | None = None


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


async def _encrypted_replacement_content(client: Any, replacement: dict[str, Any]) -> dict[str, Any] | None:
    encrypted = replacement.get("content")
    if not isinstance(encrypted, dict):
        return None
    store = client.crypto.crypto_store
    session = await store.get_group_session(replacement["room_id"], encrypted["session_id"])
    plaintext, _ = session.decrypt(encrypted["ciphertext"])
    payload = json.loads(plaintext)
    if (not isinstance(payload, dict) or payload.get("room_id") != replacement["room_id"]
            or payload.get("type") != "m.room.message"):
        return None
    content = payload.get("content")
    if not isinstance(content, dict):
        return None
    revised = content.get("m.new_content")
    return revised if isinstance(revised, dict) else None


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


async def effective_event(
    client: Any, raw: dict[str, Any], *, is_redacted: Callable[[str | None], bool] | None = None,
) -> MatrixEffectiveEvent:
    original_content = event_content(raw)
    unsigned = raw.get("unsigned")
    if ((isinstance(unsigned, dict) and unsigned.get("redacted_because"))
            or (is_redacted is not None and is_redacted(raw.get("event_id")))):
        return MatrixEffectiveEvent({}, original_content, redacted=True)

    event: Any = raw
    if raw.get("type") == "m.room.encrypted":
        event, error = await _decrypt(client, raw)
        if is_redacted is not None and is_redacted(raw.get("event_id")):
            return MatrixEffectiveEvent({}, original_content, redacted=True)
        if error is not None:
            return MatrixEffectiveEvent(None, original_content, error=error)
    event_type = event.get("type") if isinstance(event, dict) else getattr(event, "type", None)
    state = await _current_state(client, raw, event_content(event), original_content, is_redacted)
    return replace(state, event_type=None if event_type is None else str(event_type))


async def _current_state(
    client: Any, raw: dict[str, Any], content: dict[str, Any], original_content: dict[str, Any],
    is_redacted: Callable[[str | None], bool] | None,
) -> MatrixEffectiveEvent:
    replacement = _replacement(raw)
    if replacement is None or MatrixRelation.from_content(original_content.get("m.relates_to")).is_edit:
        return MatrixEffectiveEvent(content, original_content)
    replacement_id = replacement.get("event_id")
    relation = event_content(replacement).get("m.relates_to")
    if (
        replacement.get("room_id", raw.get("room_id")) != raw.get("room_id")
        or replacement.get("sender") != raw.get("sender")
        or replacement.get("type") != raw.get("type")
        or not isinstance(replacement_id, str) or not replacement_id
        or "state_key" in replacement or "state_key" in raw
        or not isinstance(relation, dict)
        or relation.get("rel_type") != "m.replace"
        or relation.get("event_id") != raw.get("event_id")
    ):
        return MatrixEffectiveEvent(content, original_content)

    unavailable = MatrixEffectiveEvent(
        {"body": "[event content unavailable]"}, original_content,
        error={"event_id": str(raw.get("event_id") or ""), "error": "replacement was redacted"},
        replacement_id=replacement_id,
    )
    if is_redacted is not None and is_redacted(replacement_id):
        return unavailable
    if (isinstance(replacement.get("unsigned"), dict)
            and replacement["unsigned"].get("redacted_because")):
        return MatrixEffectiveEvent(content, original_content)

    if replacement.get("type") == "m.room.encrypted":
        _, error = await _decrypt(client, replacement)
        if is_redacted is not None and is_redacted(raw.get("event_id")):
            return MatrixEffectiveEvent({}, original_content, redacted=True)
        if is_redacted is not None and is_redacted(replacement_id):
            return unavailable
        if error is not None:
            return MatrixEffectiveEvent(content, original_content, error=error)
        try:
            # Mautrix's typed edit serializer synthesises m.new_content even when the payload omitted it.
            revised_content = await _encrypted_replacement_content(client, replacement)
        except Exception:
            if is_redacted is not None and is_redacted(raw.get("event_id")):
                return MatrixEffectiveEvent({}, original_content, redacted=True)
            if is_redacted is not None and is_redacted(replacement_id):
                return unavailable
            return MatrixEffectiveEvent(content, original_content, error={
                "event_id": replacement.get("event_id"),
                "error": "encrypted replacement could not be inspected",
            })
    else:
        revised_content = event_content(replacement).get("m.new_content")
    if is_redacted is not None and is_redacted(raw.get("event_id")):
        return MatrixEffectiveEvent({}, original_content, redacted=True)
    if is_redacted is not None and is_redacted(replacement_id):
        return unavailable
    if not isinstance(revised_content, dict):
        return MatrixEffectiveEvent(content, original_content)

    original_relation = content.get("m.relates_to")
    content = {key: value for key, value in revised_content.items() if key != "m.relates_to"}
    if original_relation is not None:
        content["m.relates_to"] = original_relation
    return MatrixEffectiveEvent(content, original_content, edited=True, replacement_id=replacement_id)

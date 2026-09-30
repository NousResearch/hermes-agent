"""Resolve the visible state of a Matrix message from a server event."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable

from plugins.platforms.matrix.client_events import UndecryptableEvent, decrypt_history_event
from plugins.platforms.matrix.relations import MatrixRelation

if TYPE_CHECKING:
    from plugins.platforms.matrix.reply_context import MatrixEventContext, MatrixEventContextCache


@dataclass(frozen=True)
class MatrixEffectiveEvent:
    content: dict[str, Any] | None
    original_content: dict[str, Any]
    edited: bool = False
    redacted: bool = False
    error: dict[str, str] | None = None
    replacement_id: str | None = None
    event_type: str | None = None
    _dependencies: tuple[MatrixEventContext, ...] = field(default=(), compare=False, repr=False)
    # Decrypting original_content gives this value, so it takes no part in comparisons.
    _decrypted_original: dict[str, Any] | None = field(default=None, compare=False, repr=False)

    @property
    def plain_original_content(self) -> dict[str, Any]:
        """The original event's content, decrypted when the event was encrypted."""
        return self._decrypted_original if self._decrypted_original is not None else self.original_content


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
    if "m.relates_to" in content:
        relation = content["m.relates_to"]
        outer_relation = encrypted["m.relates_to"]
        if (not isinstance(relation, dict) or relation.get("rel_type") != "m.replace"
                or relation.get("event_id") != outer_relation["event_id"]):
            return None
    revised = content.get("m.new_content")
    return revised if isinstance(revised, dict) else None


async def _decrypt(client: Any, raw: dict[str, Any]) -> tuple[Any | None, dict[str, str] | None]:
    try:
        return await decrypt_history_event(client, raw), None
    except UndecryptableEvent as exc:
        return None, {"event_id": raw.get("event_id"), "error": str(exc)}


async def effective_event(
    client: Any, raw: dict[str, Any], *, cache: MatrixEventContextCache | None = None,
    room_id: str | None = None,
) -> MatrixEffectiveEvent:
    room_id = room_id if room_id is not None else str(raw.get("room_id") or "")
    dependencies = cache.retain_events(room_id, [raw]) if cache is not None else {}
    state = await _effective_event(
        client, raw,
        is_redacted=(lambda target: cache.is_redacted(room_id, target)) if cache is not None else None,
    )
    return replace(state, _dependencies=tuple(dependencies.values()))


async def _effective_event(
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
    state = await _apply_replacement(client, raw, event_content(event), original_content, is_redacted)
    return replace(
        state, event_type=None if event_type is None else str(event_type),
        _decrypted_original=None if event is raw else event_content(event),
    )


async def _apply_replacement(
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
        decrypted_replacement, error = await _decrypt(client, replacement)
        if is_redacted is not None and is_redacted(raw.get("event_id")):
            return MatrixEffectiveEvent({}, original_content, redacted=True)
        if is_redacted is not None and is_redacted(replacement_id):
            return unavailable
        if error is not None:
            return MatrixEffectiveEvent(content, original_content, error=error)
        clear_relation = event_content(decrypted_replacement).get("m.relates_to")
        if (clear_relation is not None and (
                not isinstance(clear_relation, dict) or clear_relation.get("rel_type") != "m.replace"
                or clear_relation.get("event_id") != raw.get("event_id"))):
            return MatrixEffectiveEvent(content, original_content)
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

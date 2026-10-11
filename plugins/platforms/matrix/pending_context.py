"""Reconstruct pending Matrix context against the current transport cache."""

from __future__ import annotations

import re
from typing import Any

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, QuotedMediaDependency
from plugins.platforms.matrix.followup_context import (
    EXCERPT_PART_LIMIT, REPLY_EXCERPT_CHARS, LogicalReplyContext, ReplyExcerpt,
)
from plugins.platforms.matrix.rich_content import MatrixRichContentSnapshot, _MatrixAuthoredContent
from plugins.platforms.matrix.turn_context import MatrixTurnContext


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _turn_context(adapter: Any, event: MessageEvent, state: dict[str, Any]) -> MatrixTurnContext:
    if (not isinstance(state, dict) or state.get("room_id") != event.source.chat_id
            or state.get("reply_event_id") != event.reply_to_message_id):
        raise ValueError("pending Matrix context differs from its event")
    context = MatrixTurnContext.capture(adapter, event)
    logical = state.get("logical_reply")
    if logical is None:
        return context
    if not isinstance(logical, dict):
        raise ValueError("pending logical reply must be an object")
    excerpt, delivery = logical.get("excerpt"), logical.get("delivery")
    if (not isinstance(excerpt, str) or len(excerpt) > REPLY_EXCERPT_CHARS
            or not isinstance(delivery, dict) or not isinstance(delivery.get("complete"), bool)):
        raise ValueError("pending logical reply fields are invalid")
    parts = delivery.get("parts")
    if (not isinstance(parts, list) or not parts or len(parts) > EXCERPT_PART_LIMIT
            or any(not isinstance(part, list) or len(part) != 2
                   or not isinstance(part[0], str) or not part[0]
                   or not isinstance(part[1], str) or not _DIGEST.fullmatch(part[1]) for part in parts)
            or not isinstance(delivery.get("target_digest"), str)
            or delivery["target_digest"] and not _DIGEST.fullmatch(delivery["target_digest"])):
        raise ValueError("pending logical reply dependencies are invalid")
    context.logical_reply = LogicalReplyContext.capture(
        adapter, event.source.chat_id, excerpt, ReplyExcerpt.from_json(delivery),
    )
    return context


def restore_pending_context(adapter: Any, event: MessageEvent, state: dict[str, Any]) -> None:
    if event.source.platform != Platform.MATRIX or adapter.platform != Platform.MATRIX:
        raise ValueError("pending context requires its Matrix transport")
    if (not isinstance(state, dict) or not isinstance(state.get("quoted_media"), list)
            or not isinstance(state.get("snapshots"), list)):
        raise ValueError("pending Matrix context must contain dependency lists")
    quoted = []
    for value in state["quoted_media"]:
        if (not isinstance(value, dict) or value.get("room_id") != event.source.chat_id
                or not isinstance(value.get("event_id"), str) or not value["event_id"]
                or not isinstance(value.get("content_id"), str)
                or type(value.get("media_index")) is not int
                or not 0 <= value["media_index"] < len(event.media_urls)):
            raise ValueError("pending quoted attachment identity is invalid")
        quoted.append(QuotedMediaDependency(**value))
    event._quoted_media_dependencies = tuple(quoted)
    snapshots = []
    for value in state["snapshots"]:
        if not isinstance(value, dict):
            raise ValueError("pending Matrix snapshot must be an object")
        kind = value.get("kind")
        if kind is None:
            snapshots.append(_turn_context(adapter, event, value))
            continue
        if kind != "matrix_rich_content":
            raise ValueError("pending Matrix snapshot type is unsupported")
        context = _turn_context(adapter, event, value.get("context"))
        recorded = value.get("contributions")
        if not isinstance(recorded, list) or not recorded:
            raise ValueError("pending native Matrix snapshot requires contributions")
        contributions = []
        for item in recorded:
            if (not isinstance(item, dict)
                    or any(not isinstance(item.get(key), str) for key in (
                        "event_id", "sender", "original_text", "original_content_text", "original_media_identity"))
                    or item["event_id"] not in (event.message_id, *event.merged_message_ids)
                    or item["sender"] != event.source.user_id
                    or not isinstance(item.get("media_paths"), list)
                    or any(path not in event.media_urls for path in item["media_paths"])):
                raise ValueError("pending native Matrix contribution differs from its event")
            contributions.append(_MatrixAuthoredContent(
                authored=adapter._event_context_cache.retain(event.source.chat_id, item["event_id"]),
                original_text=item["original_text"], original_content_text=item["original_content_text"],
                original_media_identity=item["original_media_identity"], media_paths=tuple(item["media_paths"]),
            ))
        snapshots.append(MatrixRichContentSnapshot(context, tuple(contributions)))
    event._inbound_context_dependencies = tuple(snapshots)

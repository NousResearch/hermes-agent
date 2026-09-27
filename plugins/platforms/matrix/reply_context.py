"""Resolve Matrix events that are referenced by replies."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatrixEventContext:
    sender: str
    text: str


def _content_dict(event: Any) -> dict:
    content = getattr(event, "content", None)
    if content is None and isinstance(event, dict):
        content = event.get("content")
    if isinstance(content, dict):
        return content
    if content is None:
        return {}
    serialise = getattr(content, "serialize", None)
    if callable(serialise):
        try:
            result = serialise()
        except Exception:
            return {}
        if isinstance(result, dict):
            return result
    return {}


def _effective_content(event: Any) -> tuple[dict, bool]:
    content = _content_dict(event)
    unsigned = getattr(event, "unsigned", None)
    if unsigned is None and isinstance(event, dict):
        unsigned = event.get("unsigned")
    if not isinstance(unsigned, dict) and unsigned is not None:
        serialise = getattr(unsigned, "serialize", None)
        if callable(serialise):
            try:
                unsigned = serialise()
            except Exception:
                unsigned = None
    relations = unsigned.get("m.relations") if isinstance(unsigned, dict) else None
    replacement = relations.get("m.replace") if isinstance(relations, dict) else None
    replacement_content = _content_dict(replacement) if replacement is not None else {}
    candidate = replacement_content or content
    new_content = candidate.get("m.new_content")
    if isinstance(new_content, dict):
        return {**content, **new_content}, True
    if replacement_content:
        return {**content, **replacement_content}, True
    return content, False


def _own_text(body: str) -> str:
    if not body.startswith("> "):
        return body
    lines = body.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("> ") or line == ">":
            continue
        if line == "":
            return "\n".join(lines[index + 1:]).strip()
        return "\n".join(lines[index:]).strip()
    return ""


def _label_body(msgtype: str, body: str) -> str:
    labels = {
        "m.image": "image", "m.audio": "audio", "m.video": "video",
        "m.file": "file", "m.notice": "notice", "m.location": "location",
    }
    label = labels.get(msgtype)
    if label is None:
        return body
    if msgtype == "m.image" and body.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
        body = ""
    return f"[{label}: {body}]" if body else f"[{label}]"


class MatrixEventContextCache:
    def __init__(self, max_entries: int = 500, timeout_seconds: float = 10.0) -> None:
        self.max_entries = max_entries
        self.timeout_seconds = timeout_seconds
        self._entries: OrderedDict[tuple[str, str], MatrixEventContext] = OrderedDict()

    def store(self, room_id: str, event_id: str, entry: MatrixEventContext) -> None:
        if not event_id:
            return
        key = room_id, event_id
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def apply_edit(self, room_id: str, sender: str, content: dict) -> None:
        relation = content.get("m.relates_to")
        target = relation.get("event_id") if isinstance(relation, dict) else None
        replacement = content.get("m.new_content")
        if not isinstance(target, str) or not isinstance(replacement, dict):
            return
        body = replacement.get("body")
        if not isinstance(body, str) or not body.strip():
            return
        prior = self._entries.get((room_id, target))
        if prior is not None and prior.sender and prior.sender != sender:
            return
        self.store(room_id, target, MatrixEventContext(sender, _own_text(body.strip())))

    def redact(self, room_id: str, event_id: str) -> None:
        prior = self._entries.get((room_id, event_id))
        sender = prior.sender if prior is not None else ""
        self.store(room_id, event_id, MatrixEventContext(sender, ""))

    async def resolve(self, client: Any, room_id: str, event_id: str) -> MatrixEventContext | None:
        key = room_id, event_id
        if key in self._entries:
            self._entries.move_to_end(key)
            entry = self._entries[key]
            return entry if entry.text else None
        if client is None:
            return None

        try:
            event = await asyncio.wait_for(client.get_event(room_id, event_id), self.timeout_seconds)
            if str(getattr(event, "type", "")) == "m.room.encrypted":
                crypto = getattr(client, "crypto", None)
                if crypto is None:
                    return None
                event = await asyncio.wait_for(crypto.decrypt_megolm_event(event), self.timeout_seconds)
        except Exception as exc:
            logger.debug("Matrix: could not resolve reply target %s in %s: %s", event_id, room_id, exc)
            return None

        sender = str(getattr(event, "sender", "") or "")
        if not sender and isinstance(event, dict):
            sender = str(event.get("sender", "") or "")
        content, edited = _effective_content(event)
        body = content.get("body")
        if not isinstance(body, str):
            body = getattr(getattr(event, "content", None), "body", "")
        body = body.strip() if isinstance(body, str) else ""
        if edited and body.startswith("* "):
            body = body[2:].strip()
        body = _own_text(body)
        text = _label_body(str(content.get("msgtype") or ""), body)
        entry = MatrixEventContext(sender=sender, text=text)
        self.store(room_id, event_id, entry)
        return entry if text else None

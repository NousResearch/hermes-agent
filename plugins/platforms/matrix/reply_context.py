"""Resolve Matrix events that are referenced by replies."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Awaitable, Callable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatrixEventContext:
    sender: str
    text: str
    media_path: str | None = None
    media_type: str | None = None


@dataclass(frozen=True)
class MatrixReplyContext:
    body: str
    event_id: str | None
    text: str | None
    author_id: str | None
    author_name: str | None
    is_own_message: bool
    author_authorized: bool | None
    media_path: str | None = None
    media_type: str | None = None


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


class _MxReplyQuoteExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._depth = 0
        self._done = False
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "mx-reply" and not self._done:
            self._depth += 1
        elif tag == "br" and self._depth and not self._done:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "mx-reply" and self._depth:
            self._depth -= 1
            if self._depth == 0:
                self._done = True

    def handle_data(self, data: str) -> None:
        if self._depth and not self._done:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def extract_mx_reply_quote(formatted_body: Any) -> str | None:
    if not isinstance(formatted_body, str) or not formatted_body.lstrip().startswith("<mx-reply"):
        return None
    parser = _MxReplyQuoteExtractor()
    try:
        parser.feed(formatted_body)
        parser.close()
    except Exception:
        return None
    text = parser.text().strip()
    first, separator, rest = text.partition("\n")
    if separator and first.strip().lower().startswith("in reply to"):
        text = rest.strip()
    return text or None


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

    async def resolve(
        self, client: Any, room_id: str, event_id: str,
        image_loader: Callable[[dict, str], Awaitable[tuple[str, str] | None]] | None = None,
    ) -> MatrixEventContext | None:
        key = room_id, event_id
        if key in self._entries:
            self._entries.move_to_end(key)
            entry = self._entries[key]
            if entry.media_path and not Path(entry.media_path).is_file():
                self._entries.pop(key)
            else:
                return entry if entry.text or entry.media_path else None
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
        msgtype = str(content.get("msgtype") or "")
        text = _label_body(msgtype, body)
        media = None
        if msgtype == "m.image" and image_loader is not None:
            try:
                media = await asyncio.wait_for(
                    image_loader(content, event_id), self.timeout_seconds
                )
            except Exception as exc:
                logger.debug("Matrix: could not cache quoted image %s: %s", event_id, exc)
        entry = MatrixEventContext(
            sender=sender, text=text,
            media_path=media[0] if media else None,
            media_type=media[1] if media else None,
        )
        self.store(room_id, event_id, entry)
        return entry if text else None

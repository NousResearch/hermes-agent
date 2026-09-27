"""Resolve Matrix events that are referenced by replies."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, replace
from enum import Enum
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote

from plugins.platforms.matrix.effective_event import effective_event
from plugins.platforms.matrix.reaction_context import MatrixReaction


logger = logging.getLogger(__name__)

try:
    from mautrix.api import Method
except ImportError:
    class Method(str, Enum):
        GET = "GET"


@dataclass(frozen=True)
class MatrixEventContext:
    sender: str
    text: str
    media_path: str | None = None
    media_type: str | None = None
    is_image: bool = False
    redacted: bool = False
    reactions: tuple[MatrixReaction, ...] = ()
    reactions_truncated: bool = False
    reaction_keys_missing: bool = False
    reactions_unavailable: bool = False
    state_error: str | None = None


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

    def history_entry(self, room_id: str, event_id: str) -> MatrixEventContext | None:
        return self._entries.get((room_id, event_id))

    def store(self, room_id: str, event_id: str, entry: MatrixEventContext) -> MatrixEventContext | None:
        if not event_id:
            return None
        key = room_id, event_id
        prior = self._entries.get(key)
        if prior is not None and prior.redacted and not entry.redacted:
            if not prior.sender and entry.sender:
                prior = replace(prior, sender=entry.sender)
                self._entries[key] = prior
            return prior
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry if entry.redacted or entry.text or entry.media_path else None

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
        if prior is not None and prior.redacted:
            return
        if prior is not None and prior.sender and prior.sender != sender:
            return
        self.store(room_id, target, MatrixEventContext(
            sender, _own_text(body.strip()),
            media_path=prior.media_path if prior else None,
            media_type=prior.media_type if prior else None,
            is_image=prior.is_image if prior else False,
        ))

    def redact(self, room_id: str, event_id: str) -> None:
        prior = self._entries.get((room_id, event_id))
        sender = prior.sender if prior is not None else ""
        self.store(room_id, event_id, MatrixEventContext(sender, "", redacted=True))

    async def resolve(
        self, client: Any, room_id: str, event_id: str,
        image_loader: Callable[[dict, str], Awaitable[tuple[str, str] | None]] | None = None,
    ) -> MatrixEventContext | None:
        key = room_id, event_id
        cached = None
        if key in self._entries:
            self._entries.move_to_end(key)
            entry = self._entries[key]
            if entry.redacted:
                return None
            if entry.media_path and not Path(entry.media_path).is_file():
                self._entries.pop(key)
            else:
                cached = entry
                if not entry.state_error and (not entry.is_image or entry.media_path or image_loader is None):
                    return entry if entry.text or entry.media_path else None
        if client is None:
            return cached

        try:
            path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
            raw = await asyncio.wait_for(client.api.request(Method.GET, path), self.timeout_seconds)
            if not isinstance(raw, dict) or raw.get("event_id") != event_id or raw.get("room_id", room_id) != room_id:
                return cached
            state = await effective_event(client, raw)
            if state.redacted:
                self.redact(room_id, event_id)
                return None
            if state.content is None:
                return cached
            content = state.content
        except Exception as exc:
            logger.debug("Matrix: could not resolve reply target %s in %s: %s", event_id, room_id, exc)
            current = self._entries.get(key)
            return current if current is not None and not current.redacted else None

        sender = str(raw.get("sender") or "")
        body = content.get("body")
        body = body.strip() if isinstance(body, str) else ""
        if state.edited and body.startswith("* "):
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
            is_image=msgtype == "m.image",
            state_error=state.error["error"] if state.error else None,
        )
        stored = self.store(room_id, event_id, entry)
        return stored if stored is not None and not stored.redacted else None

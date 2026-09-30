"""Resolve Matrix events that are referenced by replies."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote
from weakref import WeakSet

from plugins.platforms.matrix.client_events import Method
from plugins.platforms.matrix.effective_event import _replacement, effective_event
from plugins.platforms.matrix.reaction_context import MatrixReaction


logger = logging.getLogger(__name__)


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
    reactions_undecryptable: bool = False
    reactions_unavailable: bool = False
    state_error: str | None = None
    event_id: str | None = None
    replacement_id: str | None = None
    media_content: str | None = None
    _state: _MatrixEventState | None = field(default=None, compare=False, repr=False)
    _reaction_states: tuple[_MatrixEventState, ...] = field(default=(), compare=False, repr=False)
    _attachment: MatrixEventContext | None = field(default=None, compare=False, repr=False)

    @staticmethod
    def image_content(content: dict) -> str | None:
        if content.get("msgtype") != "m.image":
            return None
        return json.dumps(content, sort_keys=True, separators=(",", ":"))

    @property
    def attachment_identity(self) -> str:
        return json.dumps([self.sender, self.text, self.replacement_id, self.media_content])


@dataclass(eq=False)
class _MatrixEventState:
    room_id: str
    event_id: str
    current: MatrixEventContext
    redacted_replacement: _MatrixEventState | None = None


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
    media_content_id: str | None = None
    parent: MatrixEventContext | None = field(default=None, compare=False, repr=False)


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
        self._active_states: WeakSet[_MatrixEventState] = WeakSet()

    def history_entry(self, room_id: str, event_id: str) -> MatrixEventContext | None:
        entry = self._entries.get((room_id, event_id))
        if entry is not None:
            return entry
        return next(
            (state.current for state in self._active_states
             if state.room_id == room_id and state.event_id == event_id),
            None,
        )

    def hold(self, room_id: str, event_id: str) -> MatrixEventContext:
        """Track an event for the caller without adding it to the bounded table.

        The caller's reference keeps the event's state alive, so an edit or
        redaction stored while the caller waits updates that state.
        """
        entry = self.history_entry(room_id, event_id)
        if entry is not None:
            return entry
        state = _MatrixEventState(room_id, event_id, MatrixEventContext("", "", event_id=event_id))
        state.current = replace(state.current, _state=state)
        self._active_states.add(state)
        return state.current

    def retain(self, room_id: str, event_id: str) -> MatrixEventContext:
        entry = self.history_entry(room_id, event_id)
        if entry is None:
            self.store(room_id, event_id, MatrixEventContext("", ""))
            entry = self.history_entry(room_id, event_id)
        if entry is None:
            raise ValueError("Matrix event dependency requires an event ID")
        return entry

    def snapshot(self, room_id: str) -> dict[str, MatrixEventContext]:
        return {event_id: entry for (room, event_id), entry in self._entries.items() if room == room_id}

    def retain_events(self, room_id: str, events: list[dict]) -> dict[str, MatrixEventContext]:
        dependencies = {}
        for raw in events:
            if raw.get("room_id", room_id) != room_id:
                continue
            for event in (raw, _replacement(raw)):
                if event is None or event.get("room_id", room_id) != room_id:
                    continue
                event_id = event.get("event_id")
                if isinstance(event_id, str) and event_id:
                    dependencies[event_id] = self.retain(room_id, event_id)
        return dependencies

    def store(self, room_id: str, event_id: str, entry: MatrixEventContext) -> MatrixEventContext | None:
        if not event_id:
            return None
        key = room_id, event_id
        prior = self.history_entry(room_id, event_id)
        if prior is not None and prior.redacted:
            if not prior.sender and entry.sender:
                prior = replace(prior, sender=entry.sender)
                if prior._state is not None:
                    prior._state.current = prior
                if key in self._entries:
                    self._entries[key] = prior
            return prior
        entry = replace(entry, event_id=event_id)
        entry = self._check_dependencies(room_id, entry)
        state = prior._state if prior is not None else None
        if state is None:
            state = _MatrixEventState(room_id, event_id, entry)
            self._active_states.add(state)
        entry = replace(entry, _state=state)
        state.current = entry
        if (
            state.redacted_replacement is not None
            and entry.replacement_id != state.redacted_replacement.event_id
        ):
            state.redacted_replacement = None
        replacement = self.history_entry(room_id, entry.replacement_id) if entry.replacement_id else None
        if replacement is not None and replacement.redacted:
            state.redacted_replacement = replacement._state
        self._entries[key] = entry
        self._entries.move_to_end(key)
        if entry.redacted:
            for dependent_state in list(self._active_states):
                dependent = dependent_state.current
                if dependent_state.room_id == room_id and dependent.replacement_id == event_id:
                    dependent_state.redacted_replacement = state
                    self.store(
                        room_id, dependent_state.event_id,
                        self._unavailable(dependent, "replacement was redacted"),
                    )
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry if entry.redacted or entry.text or entry.media_path else None

    def is_redacted(self, room_id: str, event_id: str | None) -> bool:
        if event_id is None:
            return False
        entry = self.history_entry(room_id, event_id)
        return entry is not None and entry.redacted

    def store_resolved(
        self, room_id: str, event_id: str, entry: MatrixEventContext, before: MatrixEventContext | None,
    ) -> MatrixEventContext | None:
        current = self.history_entry(room_id, event_id)
        if current is not None and current.state_error and entry.state_error and not entry.redacted:
            return current
        if current is before or entry.redacted:
            attachment = (current._attachment or current) if current is not None else None
            if (
                attachment is not None and attachment.media_path and not entry.media_path
                and not entry.redacted and not entry.state_error and not attachment.state_error
                and entry.is_image and attachment.media_content is not None
                and attachment.attachment_identity == entry.attachment_identity
                and Path(attachment.media_path).is_file()
            ):
                entry = replace(entry, media_path=attachment.media_path, media_type=attachment.media_type)
            return self.store(room_id, event_id, entry)
        if current is not None and not current.sender and entry.sender:
            return self.store(room_id, event_id, replace(current, sender=entry.sender))
        return current

    def invalidate(self, room_id: str, event_id: str) -> None:
        entry = self.history_entry(room_id, event_id)
        if entry is not None and entry.redacted:
            return
        self.store(room_id, event_id, self._unavailable(
            entry or MatrixEventContext("", ""), entry.state_error if entry and entry.state_error else "event content changed",
        ))

    @staticmethod
    def _unavailable(entry: MatrixEventContext, error: str) -> MatrixEventContext:
        return MatrixEventContext(
            entry.sender, "[event content unavailable]", event_id=entry.event_id,
            state_error=error, replacement_id=entry.replacement_id,
            _state=entry._state,
            _attachment=entry._attachment or (entry if entry.media_path else None),
        )

    def recheck(self, room_id: str, entry: MatrixEventContext) -> MatrixEventContext:
        current = self.history_entry(room_id, entry.event_id) if entry.event_id else None
        if current is not None and current.redacted:
            return current
        if current is not None:
            entry = replace(
                current, reactions=entry.reactions, reactions_truncated=entry.reactions_truncated,
                reactions_undecryptable=entry.reactions_undecryptable, reactions_unavailable=entry.reactions_unavailable,
                _reaction_states=entry._reaction_states,
            )
        return self._check_dependencies(room_id, entry)

    def _check_dependencies(self, room_id: str, entry: MatrixEventContext) -> MatrixEventContext:
        if entry.replacement_id and self.is_redacted(room_id, entry.replacement_id):
            return self._unavailable(entry, "replacement was redacted")
        dependencies = tuple(
            retained._state for reaction in entry.reactions
            if (retained := self.retain(room_id, reaction.event_id))._state is not None
        )
        return replace(entry, reactions=tuple(
            reaction for reaction in entry.reactions if not self.is_redacted(room_id, reaction.event_id)
        ), _reaction_states=dependencies)

    def apply_edit(
        self, room_id: str, sender: str, content: dict, *, replacement_id: str | None = None,
    ) -> None:
        relation = content.get("m.relates_to")
        target = relation.get("event_id") if isinstance(relation, dict) else None
        replacement = content.get("m.new_content")
        if not isinstance(target, str) or not isinstance(replacement, dict):
            return
        body = replacement.get("body")
        if not isinstance(body, str) or not body.strip():
            return
        prior = self.history_entry(room_id, target)
        # Without the original, the editor cannot be checked against its sender. A later
        # resolve fetches the event, and the server bundles only same-sender replacements.
        if prior is None or prior.redacted:
            return
        if not prior.sender:
            self.invalidate(room_id, target)
            return
        if prior.sender != sender:
            return
        self.store(room_id, target, MatrixEventContext(
            sender, _own_text(body.strip()),
            is_image=replacement.get("msgtype") == "m.image",
            media_content=MatrixEventContext.image_content(replacement),
            replacement_id=replacement_id,
        ))

    def redact(self, room_id: str, event_id: str) -> None:
        prior = self.history_entry(room_id, event_id)
        sender = prior.sender if prior is not None else ""
        self.store(room_id, event_id, MatrixEventContext(sender, "", redacted=True))

    async def refresh(
        self, client: Any, room_id: str, entry: MatrixEventContext,
    ) -> MatrixEventContext:
        current = self.recheck(room_id, entry)
        if current.event_id and not current.redacted and (current.state_error or not current.text):
            await self.resolve(client, room_id, current.event_id)
        return self.recheck(room_id, entry)

    async def resolve(
        self, client: Any, room_id: str, event_id: str,
        image_loader: Callable[[dict, str], Awaitable[tuple[str, str] | None]] | None = None,
    ) -> MatrixEventContext | None:
        key = room_id, event_id
        before = self.retain(room_id, event_id)
        cached = None
        if before is not None:
            if key in self._entries:
                self._entries.move_to_end(key)
            entry = before
            if entry.redacted:
                return None
            if entry.media_path and not Path(entry.media_path).is_file():
                entry = replace(entry, media_path=None, media_type=None)
                self.store(room_id, event_id, entry)
                entry = self.history_entry(room_id, event_id)
                before = entry
                cached = entry
            else:
                cached = entry
                if (not entry.state_error and (entry.text or entry.media_path)
                        and (not entry.is_image or entry.media_path or image_loader is None)):
                    return entry if entry.text or entry.media_path else None
        if client is None:
            return cached if cached is not None and (cached.text or cached.media_path or cached.state_error) else None

        def current_cached() -> MatrixEventContext | None:
            current = self.history_entry(room_id, event_id)
            return current if current is not None and not current.redacted and (current.text or current.media_path or current.state_error) else None

        try:
            path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}/event/{quote(event_id, safe='')}"
            raw = await asyncio.wait_for(client.api.request(Method.GET, path), self.timeout_seconds)
            if not isinstance(raw, dict) or raw.get("event_id") != event_id or raw.get("room_id", room_id) != room_id:
                return current_cached()
            state = await effective_event(client, raw, cache=self, room_id=room_id)
            if state.redacted:
                self.redact(room_id, event_id)
                return None
            if state.content is None:
                return current_cached()
            if state.error and cached is not None and cached.state_error:
                return current_cached()
            content = state.content
        except Exception as exc:
            logger.debug("Matrix: could not resolve reply target %s in %s: %s", event_id, room_id, exc)
            return current_cached()

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
            media_content=MatrixEventContext.image_content(content),
            state_error=state.error["error"] if state.error else None,
            replacement_id=state.replacement_id,
        )
        stored = self.store_resolved(room_id, event_id, entry, before)
        return stored if stored is not None and not stored.redacted else None

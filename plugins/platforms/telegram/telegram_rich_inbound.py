"""Bounded, text-only projection of inbound Telegram ``rich_message`` blocks.

Only documented display fields are read, from mappings or typed Telegram objects.
A local policy view separates visible trigger text from code and URL destinations;
authorization remains in the adapter. No serialization, media lookup or commands.
Depth counts schema edges (root at zero), and every visited collection, block, item,
cell and inline value shares one node budget. Any limit aborts all remaining branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from inspect import getattr_static
import re
from types import SimpleNamespace
from typing import Any


MAX_DEPTH = 16
MAX_NODES = 2048
MAX_BLOCKS = 256
MAX_CELLS = 512
MAX_OUTPUT_CHARS = 32768

UNSUPPORTED = "[Unsupported Telegram rich content]"
INVALID = "[Invalid Telegram rich content]"
EMPTY = "[Empty Telegram rich message]"
TRUNCATED = "[Telegram rich message truncated]"

_MISSING = object()
_INLINE_TEXT_TYPES = frozenset({
    "bold", "italic", "underline", "strikethrough", "spoiler", "code", "url",
    "mention", "text_mention", "bot_command",
    "date_time", "subscript", "superscript", "marked", "email_address",
    "phone_number", "bank_card_number", "hashtag", "cashtag", "anchor_link",
    "reference", "reference_link",
})
_BLOCK_HANDLERS = {
    "paragraph": "_text", "heading": "_text", "footer": "_text", "pre": "_pre",
    "blockquote": "_quote", "pullquote": "_pullquote", "details": "_details",
    "table": "_table", "list": "_list",
    "expandable_blockquote": "_pullquote",
    "divider": "_divider", "mathematical_expression": "_math",
    **dict.fromkeys(("photo", "video", "animation", "audio", "voice_note", "document",
                     "map", "collage", "slideshow"), "_media"),
}
_MAX_TYPE_LENGTH = max(map(len, (*_BLOCK_HANDLERS, *_INLINE_TEXT_TYPES, "custom_emoji")))


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _rich_payload(message: Any) -> Any:
    # Dynamic proxies must not invent a carrier on an otherwise ordinary message.
    declared = isinstance(message, Mapping) or getattr_static(message, "rich_message", _MISSING) is not _MISSING
    typed = _field(message, "rich_message") if declared else _MISSING
    # A typed optional None means absent; an explicit raw key (even null) does not.
    if typed is not _MISSING and typed is not None:
        return typed
    extra = _field(message, "api_kwargs")
    if isinstance(extra, Mapping) and "rich_message" in extra:
        return extra["rich_message"]
    return _MISSING


def has_rich_message(message: Any) -> bool:
    """Cheap presence check, including malformed/empty explicit API payloads."""
    return _rich_payload(message) is not _MISSING


def rich_message_text(message: Any) -> str | None:
    """Return None only when absent; otherwise project text or a fixed fallback.

    Typed non-null ``rich_message`` wins over ``api_kwargs['rich_message']``.
    Ordinary text/caption precedence is deliberately the caller's responsibility.
    """
    payload = _rich_payload(message)
    return None if payload is _MISSING else project_rich_message(payload)


class RichMessageView:
    """Local policy view; never mutate PTB or replace the event's raw message."""

    def __init__(self, message: Any) -> None:
        self.original = message
        self.display_text, _, self.text, mention_text = _project_rich_message(_rich_payload(message))
        self.caption = None
        self.caption_entities = ()
        entities = []
        previous = offset = 0
        for match in re.finditer(r"(?i)(?<![A-Za-z0-9_`/])@[A-Za-z0-9_]{2,32}\b", mention_text):
            offset += len(self.text[previous:match.start()].encode("utf-16-le")) // 2
            entities.append(SimpleNamespace(type="mention", offset=offset, length=len(match.group()), user=None))
            previous = match.start()
        # This view has a complete mention inventory, even if empty. The sentinel
        # prevents legacy entity-less fallback from reinterpreting link labels.
        self.entities = tuple(entities) or (SimpleNamespace(type="rich_text"),)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


def rich_reply_text(message: Any) -> str | None:
    """Leave the sent-index fallback available when the echo has no display content.

    Status is kept separately: a user's literal marker-looking text is still content.
    """
    payload = _rich_payload(message)
    if payload is _MISSING:
        return None
    text, has_content, _, _ = _project_rich_message(payload)
    return text if has_content else None


class _LimitReached(Exception):
    """Unwind immediately, without inspecting any remaining siblings."""


class _InvalidContent(Exception):
    """A malformed supported shape must not produce a misleading partial message."""


def _kind(value: Any) -> str:
    kind = _field(value, "type")
    if not isinstance(kind, str):
        raise _InvalidContent
    # Unknown arbitrarily long discriminators must not incur whole-string hashing.
    return kind if len(kind) <= _MAX_TYPE_LENGTH else ""


class _Output:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.size = 0
        self.pending_boundary = False
        self.last = ""
        self.capture_triggers = False
        self.capture_mentions = False
        # Display cost, semantic text, explicit-mention eligibility. An escaped
        # cell character costs two display characters but remains one semantic unit.
        self.trigger_parts: list[tuple[int, str, bool]] = []

    def trigger_boundary(self) -> None:
        # An empty excluded node still separates adjacent visible inline fragments.
        self.trigger_parts.append((0, "\n", True))

    def finish_triggers(self, *, truncated: bool = False, mentions_only: bool = False) -> str:
        remaining = min(self.size, MAX_OUTPUT_CHARS - len(TRUNCATED) - 1) if truncated else self.size
        parts = []
        for size, text, mention in self.trigger_parts:
            if remaining <= 0:
                break
            if mentions_only and not mention:
                text = " " * len(text)
            if size > remaining and len(text) < size:
                text = ""  # Never expose a partially retained cell escape.
            parts.append(text[:remaining] if size else text)
            remaining -= min(size, remaining)
        text = "".join(parts)
        clipped = len(text) > MAX_OUTPUT_CHARS
        text = text[:MAX_OUTPUT_CHARS]
        # Never turn a cut longer handle/word into an exact mention/wake word.
        if truncated or clipped:
            text = re.sub(r"\S+$", "", text)
        return text if text.strip() else ""

    def boundary(self) -> None:
        self.pending_boundary = True

    def _append(self, text: str, *, cell_character: str | None = None) -> None:
        available = MAX_OUTPUT_CHARS - self.size
        part = text[:available]
        if part:
            self.parts.append(part)
            semantic = part if cell_character is None else (cell_character if len(part) == len(text) else "")
            self.trigger_parts.append((len(part), semantic if self.capture_triggers else " " * len(semantic), self.capture_mentions))
            self.size += len(part)
            self.last = part[-1]
        if len(text) > available:
            raise _LimitReached

    def write(self, text: str, *, cell: bool = False) -> None:
        if not text:
            return
        if self.pending_boundary and self.size and self.last != "\n":
            self._append("\n")
        self.pending_boundary = False
        # Slice BEFORE escaping: a single huge string must not bypass the work bound.
        piece = text[:MAX_OUTPUT_CHARS - self.size + 1]
        if cell:
            escapes = {"\\": "\\\\", "|": "\\|", "\r": "\\r", "\n": "\\n"}
            for fragment in re.split(r"([\\|\r\n])", piece):
                if fragment in escapes:
                    self._append(escapes[fragment], cell_character=fragment)
                else:
                    self._append(fragment)
            return
        self._append(piece)

    def finish(self, *, truncated: bool = False) -> str:
        text = "".join(self.parts)  # Only already-bounded output, never input collections.
        if truncated:
            if not text:
                return TRUNCATED
            return text[:MAX_OUTPUT_CHARS - len(TRUNCATED) - 1] + "\n" + TRUNCATED
        return text if text.strip() else EMPTY


class _Projection:
    def __init__(self) -> None:
        self.output = _Output()
        self.nodes = 0
        self.blocks = 0
        self.cells = 0
        self.has_content = False
        self.suppress_triggers = False
        self.suppress_mentions = False

    def _check(self, depth: int, kind: str = "node") -> None:
        if depth > MAX_DEPTH or self.nodes >= MAX_NODES:
            raise _LimitReached
        if self.output.size >= MAX_OUTPUT_CHARS:
            raise _LimitReached
        if kind == "block" and self.blocks >= MAX_BLOCKS:
            raise _LimitReached
        if kind == "cell" and self.cells >= MAX_CELLS:
            raise _LimitReached

    def _visit(self, depth: int, kind: str = "node") -> None:
        self._check(depth, kind)
        self.nodes += 1
        self.blocks += kind == "block"
        self.cells += kind == "cell"

    def _child(self, value: Any, name: str, depth: int, default: Any = _MISSING) -> Any:
        # Check BEFORE getattr: even retrieving the next branch must stay bounded.
        self._check(depth)
        return _field(value, name, default)

    def _sequence(self, value: Any, depth: int) -> list | tuple:
        self._visit(depth)
        if not isinstance(value, (list, tuple)):
            raise _InvalidContent
        return value

    def _inline(self, value: Any, depth: int, *, cell: bool = False) -> None:
        self._visit(depth)
        if isinstance(value, str):
            self.has_content |= bool(value[:MAX_OUTPUT_CHARS].strip())
            self.output.capture_triggers = not self.suppress_triggers
            self.output.capture_mentions = not (self.suppress_triggers or self.suppress_mentions)
            try:
                self.output.write(value, cell=cell)
            finally:
                self.output.capture_triggers = False
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                self._inline(child, depth + 1, cell=cell)
            return
        kind = _kind(value)
        if kind in _INLINE_TEXT_TYPES:
            suppressed = self.suppress_triggers
            mentions_suppressed = self.suppress_mentions
            if kind in {"code", "url"}:
                self.output.trigger_boundary()
            self.suppress_triggers |= kind == "code"
            # Link labels remain visible to wake patterns, not explicit mentions.
            self.suppress_mentions |= kind == "url"
            try:
                self._inline(self._child(value, "text", depth + 1), depth + 1, cell=cell)
            finally:
                self.suppress_triggers = suppressed
                self.suppress_mentions = mentions_suppressed
            if kind == "url":
                self._url(value, depth, cell=cell)
            if kind in {"code", "url"}:
                self.output.trigger_boundary()
        elif kind == "custom_emoji":
            self._inline(self._child(value, "alternative_text", depth + 1), depth + 1, cell=cell)
        elif kind == "mathematical_expression":
            self.output.write("$", cell=cell)
            self._inline(self._child(value, "expression", depth + 1), depth + 1, cell=cell)
            self.output.write("$", cell=cell)
        else:
            self.output.write(UNSUPPORTED, cell=cell)

    def _blocks(self, value: Any, depth: int, *, continuation: bool = False) -> None:
        for block in self._sequence(value, depth):
            size = self.output.size
            pending_boundary = self.output.pending_boundary
            self._block(block, depth + 1, continuation=continuation)
            if self.output.size > size:
                continuation = False
            elif continuation:
                self.output.pending_boundary = pending_boundary

    def _block(self, block: Any, depth: int, *, continuation: bool = False) -> None:
        self._visit(depth, "block")
        if not continuation:
            self.output.boundary()
        kind = _kind(block)
        handler = _BLOCK_HANDLERS.get(kind)
        if handler is None:
            self.output.write(UNSUPPORTED)
        else:
            getattr(self, handler)(block, depth)

    def _text(self, block: Any, depth: int) -> None:
        self._inline(self._child(block, "text", depth + 1), depth + 1)

    def _url(self, value: Any, depth: int, *, cell: bool) -> None:
        url = self._child(value, "url", depth + 1, None)
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            self.has_content = True
            self.output.write(" (", cell=cell)
            self.output.write(url, cell=cell)
            self.output.write(")", cell=cell)

    def _pre(self, block: Any, depth: int) -> None:
        self.output.write("[Code]\n")
        suppressed = self.suppress_triggers
        self.suppress_triggers = True
        try:
            self._text(block, depth)
        finally:
            self.suppress_triggers = suppressed
        self.output.boundary()
        self.output.write("[/Code]")

    def _quote(self, block: Any, depth: int) -> None:
        self.output.write("[Quote]")
        self._blocks(self._child(block, "blocks", depth + 1), depth + 1)
        credit = self._child(block, "credit", depth + 1, None)
        if credit is not None:
            self.output.boundary()
            self._inline(credit, depth + 1)
        self.output.boundary()
        self.output.write("[/Quote]")

    def _pullquote(self, block: Any, depth: int) -> None:
        self.output.write("[Quote]\n")
        self._text(block, depth)
        credit = self._child(block, "credit", depth + 1, None)
        if credit is not None:
            self.output.boundary()
            self._inline(credit, depth + 1)
        self.output.boundary()
        self.output.write("[/Quote]")

    def _details(self, block: Any, depth: int) -> None:
        self.output.write("[Details]\n")
        self._inline(self._child(block, "summary", depth + 1), depth + 1)
        self._blocks(self._child(block, "blocks", depth + 1), depth + 1)
        self.output.boundary()
        self.output.write("[/Details]")

    def _list(self, block: Any, depth: int) -> None:
        for item in self._sequence(self._child(block, "items", depth + 1), depth + 1):
            self._visit(depth + 2)
            self.output.boundary()
            label = self._child(item, "label", depth + 3, "")
            if not isinstance(label, str):
                raise _InvalidContent
            if label:
                self._inline(label, depth + 3)
                self.output.write(" ")
            if self._child(item, "has_checkbox", depth + 3, False) is True:
                checked = self._child(item, "is_checked", depth + 3, False) is True
                self.output.write("[x] " if checked else "[ ] ")
            self._blocks(self._child(item, "blocks", depth + 3), depth + 3, continuation=True)
        # Retain the display fields supported by the original nodeXbit/toffguy77 path.
        text = self._child(block, "text", depth + 1, None)
        if text is not None:
            self.output.boundary()
            self._inline(text, depth + 1)
        children = self._child(block, "blocks", depth + 1, None)
        if children is not None:
            self._blocks(children, depth + 1)

    def _divider(self, block: Any, depth: int) -> None:
        self.has_content = True
        self.output.write("---")

    def _math(self, block: Any, depth: int) -> None:
        self.output.write("$$")
        self._inline(self._child(block, "expression", depth + 1), depth + 1)
        self.output.write("$$")

    def _media(self, block: Any, depth: int) -> None:
        # Display only: never inspect file IDs, resolve media, or fetch a URL.
        self.output.write("[" + _kind(block).replace("_", " "))
        caption = self._child(block, "caption", depth + 1, None)
        if caption is not None:
            self._visit(depth + 1)
            text = self._child(caption, "text", depth + 2, None)
            if text is not None:
                self.output.write(": ")
                self._inline(text, depth + 2)
            credit = self._child(caption, "credit", depth + 2, None)
            if credit is not None:
                self.output.write(" — ")
                self._inline(credit, depth + 2)
        self.output.write("]")
        if _kind(block) in ("collage", "slideshow"):
            children = self._child(block, "blocks", depth + 1, None)
            if children is not None:
                self._blocks(children, depth + 1)

    def _span(self, cell: Any, name: str) -> None:
        if self.output.size >= MAX_OUTPUT_CHARS:
            raise _LimitReached
        span = _field(cell, name, None)
        if span is None:
            return
        if type(span) is not int or span < 1:
            raise _InvalidContent
        if span > 1:
            # Beyond the entire cell budget, retain the marker, not an unbounded integer.
            count = f"={span}" if span <= MAX_CELLS else ""
            self.output.write(f" [{name}{count}]")

    def _cell(self, cell: Any, depth: int) -> None:
        self._visit(depth, "cell")
        if cell is None or isinstance(cell, (str, int, float, bool, list, tuple)):
            raise _InvalidContent
        self.output.write(" ")
        text = self._child(cell, "text", depth + 1, "")
        self._inline("" if text is None else text, depth + 1, cell=True)
        self._span(cell, "colspan")
        self._span(cell, "rowspan")
        self.output.write(" |")

    def _table(self, block: Any, depth: int) -> None:
        caption = self._child(block, "caption", depth + 1, None)
        if caption is not None:
            self._inline(caption, depth + 1)
        rows = self._sequence(self._child(block, "cells", depth + 1), depth + 1)
        for row in rows:
            cells = self._sequence(row, depth + 2)
            self.output.boundary()
            self.output.write("|")
            for cell in cells:
                self._cell(cell, depth + 3)


def project_rich_message(payload: Any) -> str:
    """Project a mapping/typed rich-message payload under shared hard limits.

    Malformed supported shapes return INVALID (never a raw dump or partial parse).
    Unknown types get UNSUPPORTED markers without reading their payloads. Truncation
    preserves a bounded prefix and always includes the full TRUNCATED marker.
    """
    return _project_rich_message(payload)[0]


def _project_rich_message(payload: Any) -> tuple[str, bool, str, str]:
    projection = _Projection()
    truncated = False
    try:
        projection._visit(0)
        projection._blocks(projection._child(payload, "blocks", 1), 1)
    except _InvalidContent:
        return INVALID, False, "", ""
    except _LimitReached:
        truncated = True
    return (projection.output.finish(truncated=truncated), projection.has_content,
            projection.output.finish_triggers(truncated=truncated),
            projection.output.finish_triggers(truncated=truncated, mentions_only=True))

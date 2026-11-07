"""Bounded, text-only projection of inbound Telegram ``rich_message`` blocks.

Only documented display fields are read, from mappings or typed Telegram objects.
No serialization, media lookup, command canonicalization, or adapter policy lives here.
Depth counts schema edges (root at zero), and every visited collection, block, item,
cell and inline value shares one node budget. Any limit aborts all remaining branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from inspect import getattr_static
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
})
_BLOCK_HANDLERS = {
    "paragraph": "_text", "heading": "_text", "footer": "_text", "pre": "_pre",
    "blockquote": "_quote", "pullquote": "_pullquote", "details": "_details",
    "table": "_table", "list": "_list",
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

    def boundary(self) -> None:
        self.pending_boundary = True

    def _append(self, text: str) -> None:
        available = MAX_OUTPUT_CHARS - self.size
        part = text[:available]
        if part:
            self.parts.append(part)
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
            piece = (piece.replace("\\", "\\\\").replace("|", "\\|")
                     .replace("\r", "\\r").replace("\n", "\\n"))
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
            self.output.write(value, cell=cell)
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                self._inline(child, depth + 1, cell=cell)
            return
        kind = _kind(value)
        if kind in _INLINE_TEXT_TYPES:
            self._inline(self._child(value, "text", depth + 1), depth + 1, cell=cell)
            if kind == "url":
                self._url(value, depth, cell=cell)
        elif kind == "custom_emoji":
            self._inline(self._child(value, "alternative_text", depth + 1), depth + 1, cell=cell)
        else:
            self.output.write(UNSUPPORTED, cell=cell)

    def _blocks(self, value: Any, depth: int) -> None:
        for block in self._sequence(value, depth):
            self._block(block, depth + 1)

    def _block(self, block: Any, depth: int) -> None:
        self._visit(depth, "block")
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
            self.output.write(" (", cell=cell)
            self.output.write(url, cell=cell)
            self.output.write(")", cell=cell)

    def _pre(self, block: Any, depth: int) -> None:
        self.output.write("[Code]\n")
        self._text(block, depth)
        self.output.boundary()
        self.output.write("[/Code]")

    def _quote(self, block: Any, depth: int) -> None:
        self.output.write("[Quote]")
        self._blocks(self._child(block, "blocks", depth + 1), depth + 1)
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
            self._blocks(self._child(item, "blocks", depth + 3), depth + 3)

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
        self.output.write(" ")
        self._inline(self._child(cell, "text", depth + 1), depth + 1, cell=True)
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
    projection = _Projection()
    try:
        projection._visit(0)
        projection._blocks(projection._child(payload, "blocks", 1), 1)
    except _InvalidContent:
        return INVALID
    except _LimitReached:
        return projection.output.finish(truncated=True)
    return projection.output.finish()

"""Telegram split delivery: ``(N/M)`` chunk markers and the split cap.

The cap (#152012): a degenerate 415,742-char final once went out as 402 back-to-back
``sendMessage`` calls and held the chat in multi-minute flood control. Like Discord (#86581),
a reply that splits past ``MAX_SPLIT_MESSAGES`` keeps its first N-1 chunks and replaces the
rest with one visible notice; the full text stays in the session history. The cap runs before
delivery, so a partial send's resumable remainder never includes the dropped tail.
"""

from __future__ import annotations

import re
from typing import Any

from gateway.platforms.base import utf16_len

_CHUNK_INDICATOR_RE = re.compile(r" \((\d+)/(\d+)\)$")
_CHUNK_INDICATOR_ON_FENCE_RE = re.compile(r"(?m)^``` (?P<indicator>(?:\\)?\(\d+/\d+(?:\\)?\))$")
_MDV2_ESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)


def separate_chunk_indicator_from_fence(text: str) -> str:
    """Move a ``(N/M)`` chunk marker that ``truncate_message()`` appended to a synthesized closing
    fence onto its own line — Telegram rejects ````` \\(1/2\\)`` as a fence."""
    return _CHUNK_INDICATOR_ON_FENCE_RE.sub(r"```\n\g<indicator>", text)


def cap_split_chunks(adapter: Any, chunks: list[str], *, formatted: bool) -> list[str]:
    """Keep the first ``MAX_SPLIT_MESSAGES - 1`` chunks plus the truncation notice (when warnings
    are shown), renumbering the raw ``(N/M)`` markers to the capped count. ``formatted`` chunks are
    MarkdownV2 already, so the notice is formatted the same way and escapes are not counted as
    dropped characters; raw chunks get their formatting at send time."""
    cap = adapter.MAX_SPLIT_MESSAGES
    if len(chunks) <= cap:
        return chunks
    kept, dropped = chunks[: cap - 1], chunks[cap - 1:]
    dropped_text = (_CHUNK_INDICATOR_RE.sub("", chunk) for chunk in dropped)
    if formatted:
        dropped_text = (_MDV2_ESCAPE_RE.sub(r"\1", chunk) for chunk in dropped_text)
    notice = adapter._split_cap_notice(sum(len(chunk) for chunk in dropped_text))
    tail = [adapter.format_message(notice) if formatted else notice] if notice else []
    total = str(len(kept) + len(tail))
    renumbered = [_CHUNK_INDICATOR_RE.sub(lambda m: f" ({m.group(1)}/{total})", chunk) for chunk in kept]
    return renumbered + tail


def split_send_chunks(adapter: Any, content: str) -> list[str]:
    """``send()``'s MarkdownV2 chunks: formatted first, sized in UTF-16 units, capped, then the raw
    `` (1/2)`` suffix ``truncate_message`` appends gets its MarkdownV2-special parentheses escaped."""
    chunks = adapter.truncate_message(adapter.format_message(content), adapter.MAX_MESSAGE_LENGTH, len_fn=utf16_len)
    if len(chunks) <= 1:
        return chunks
    return [
        separate_chunk_indicator_from_fence(_CHUNK_INDICATOR_RE.sub(r" \\(\1/\2\\)", chunk))
        for chunk in cap_split_chunks(adapter, chunks, formatted=True)
    ]

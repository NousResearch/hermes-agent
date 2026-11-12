"""Telegram split-delivery cap, and the MarkdownV2 strip a split chunk's plain-text fallback uses.

The cap (#152012): a degenerate 415,742-char final once went out as 402 back-to-back
``sendMessage`` calls and held the chat in multi-minute flood control. Like Discord (#86581),
a reply that splits past ``MAX_SPLIT_MESSAGES`` keeps its first N-1 chunks and replaces the
rest with one visible notice; the full text stays in the session history. The cap runs before
delivery, so a partial send's resumable remainder never includes the dropped tail.
"""

from __future__ import annotations

import re
from typing import Any

# The ``(N/M)`` marker ``truncate_message`` appends: raw on edit-overflow chunks, MarkdownV2-escaped
# (``\(N/M\)``, possibly moved onto its own line after a closing fence) on send chunks.
_RAW_INDICATOR_RE = re.compile(r" \((\d+)/(\d+)\)$")
_MDV2_INDICATOR_RE = re.compile(r"\\\((\d+)/(\d+)\\\)$")
_MDV2_ESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)
# A dropped chunk's marker together with the separator before it (a space, or the newline that moves it
# off a closing fence): neither is reply text, so neither counts as a dropped character.
_DROPPED_MARKER_RE = {False: _RAW_INDICATOR_RE, True: re.compile(r"\s*" + _MDV2_INDICATOR_RE.pattern)}


class CappedChunks(list):
    """``cap_split_chunks`` output that the cap cut; ``dropped`` is the reply text it dropped."""
    dropped = 0


def split_budget(adapter: Any, metadata: Any) -> int:
    """The ceiling for one delivery: ``MAX_SPLIT_MESSAGES``, lowered by the stream consumer's
    ``_split_budget`` (the messages this call may still open inside the reply's ceiling)."""
    budget = metadata.get("_split_budget") if isinstance(metadata, dict) else None
    cap = adapter.MAX_SPLIT_MESSAGES
    return max(1, min(cap, budget)) if type(budget) is int else cap


def capped_raw(chunks: Any, raw: Any) -> Any:
    """``raw_response`` for a delivery of ``chunks``: a capped one says so (``split_capped``)."""
    if not isinstance(chunks, CappedChunks):
        return raw
    return {**(raw if isinstance(raw, dict) else {}), "split_capped": True, "dropped_chars": chunks.dropped}


def cap_split_chunks(adapter: Any, chunks: list[str], *, formatted: bool, metadata: Any = None) -> list[str]:
    """Keep the first ``cap - 1`` chunks plus the truncation notice (when warnings are shown),
    renumbering the ``(N/M)`` markers to the capped count; ``cap`` is :func:`split_budget`.
    ``formatted`` chunks are MarkdownV2 already, so the notice is formatted the same way and escapes
    are not counted as dropped characters; raw chunks get their formatting at send time."""
    cap = split_budget(adapter, metadata)
    if len(chunks) <= cap:
        return chunks
    marker = _MDV2_INDICATOR_RE if formatted else _RAW_INDICATOR_RE
    kept, dropped = chunks[: cap - 1], chunks[cap - 1:]
    dropped_text = (_DROPPED_MARKER_RE[formatted].sub("", chunk) for chunk in dropped)
    if formatted:
        dropped_text = (_MDV2_ESCAPE_RE.sub(r"\1", chunk) for chunk in dropped_text)
    dropped_chars = sum(len(chunk) for chunk in dropped_text)
    notice = adapter._split_cap_notice(dropped_chars)
    tail = [adapter.format_message(notice) if formatted else notice] if notice else []
    total = len(kept) + len(tail)
    if formatted:
        out = CappedChunks(marker.sub(lambda m: f"\\({m.group(1)}/{total}\\)", chunk) for chunk in kept)
    else:
        out = CappedChunks(marker.sub(lambda m: f" ({m.group(1)}/{total})", chunk) for chunk in kept)
    out += tail
    out.dropped = dropped_chars
    return out


def _strip_mdv2(text: str) -> str:
    """Strip MarkdownV2 escapes and formatting markers for the plain-text fallback."""
    cleaned = re.sub(r'\\([_*\[\]()~`>#\+\-=|{}.!\\])', r'\1', text)  # escape backslashes
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)  # **bold** BEFORE MarkdownV2 *bold*
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)
    cleaned = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', cleaned)  # italic; word-bounded so snake_case survives
    cleaned = re.sub(r'~([^~]+)~', r'\1', cleaned)  # strikethrough
    cleaned = re.sub(r'\|\|([^|]+)\|\|', r'\1', cleaned)  # spoiler
    return cleaned

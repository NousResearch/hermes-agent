"""Telegram split-delivery cap.

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


def cap_split_chunks(adapter: Any, chunks: list[str], *, formatted: bool) -> list[str]:
    """Keep the first ``MAX_SPLIT_MESSAGES - 1`` chunks plus the truncation notice (when warnings
    are shown), renumbering the ``(N/M)`` markers to the capped count. ``formatted`` chunks are
    MarkdownV2 already, so the notice is formatted the same way and escapes are not counted as
    dropped characters; raw chunks get their formatting at send time."""
    cap = adapter.MAX_SPLIT_MESSAGES
    if len(chunks) <= cap:
        return chunks
    marker = _MDV2_INDICATOR_RE if formatted else _RAW_INDICATOR_RE
    kept, dropped = chunks[: cap - 1], chunks[cap - 1:]
    dropped_text = (_DROPPED_MARKER_RE[formatted].sub("", chunk) for chunk in dropped)
    if formatted:
        dropped_text = (_MDV2_ESCAPE_RE.sub(r"\1", chunk) for chunk in dropped_text)
    notice = adapter._split_cap_notice(sum(len(chunk) for chunk in dropped_text))
    tail = [adapter.format_message(notice) if formatted else notice] if notice else []
    total = len(kept) + len(tail)
    if formatted:
        return [marker.sub(lambda m: f"\\({m.group(1)}/{total}\\)", chunk) for chunk in kept] + tail
    return [marker.sub(lambda m: f" ({m.group(1)}/{total})", chunk) for chunk in kept] + tail

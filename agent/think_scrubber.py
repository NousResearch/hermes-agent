"""Stateful scrubber for reasoning/thinking blocks in streamed assistant text.

The regex ``_strip_think_blocks`` is correct for a complete string but, run per-delta, erases an
opening ``<think>`` that arrives alone, so downstream state machines leak reasoning. This class
holds partial tags at delta boundaries until resolved; ``flush()`` releases held-back prose that
was not a tag; ``reset()`` at the top of each turn. An open tag only starts a block at a block
boundary (stream start / after a newline / whitespace-only line so far), so prose that *mentions*
``<think>`` is not suppressed; closed pairs are always suppressed (intentional).
"""

from __future__ import annotations

import re
from typing import Tuple

__all__ = ["StreamingThinkScrubber"]


class StreamingThinkScrubber:
    """Stateful scrubber for streaming reasoning/thinking blocks."""

    _OPEN_TAG_NAMES: Tuple[str, ...] = (
        "think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD",
    )
    _OPEN_TAGS: Tuple[str, ...] = tuple(f"<{name.lower()}>" for name in _OPEN_TAG_NAMES)
    _CLOSE_TAGS: Tuple[str, ...] = tuple(f"</{name.lower()}>" for name in _OPEN_TAG_NAMES)
    _ALL_TAGS: Tuple[str, ...] = _OPEN_TAGS + _CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(tag) for tag in _ALL_TAGS)
    _ORPHAN_CLOSE_RE = re.compile(
        "(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")[ \t\n\r]*", re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state. Call at the top of every new turn."""
        self._in_block = False
        self._buf = ""
        self._last_stripped_reasoning = ""
        self._last_emitted_ended_newline = True

    def _emit(self, out: list[str], text: str) -> None:
        """Append visible prose to *out* and track its newline boundary."""
        text = self._strip_orphan_close_tags(text)
        if text:
            out.append(text)
            self._last_emitted_ended_newline = text.endswith("\n")

    def pop_stripped_reasoning(self) -> str:
        """Return and clear reasoning text stripped by the most recent feed."""
        text = self._last_stripped_reasoning
        self._last_stripped_reasoning = ""
        return text

    def feed(self, text: str) -> str:
        """Feed one delta; return its visible portion and retain stripped reasoning."""
        self._last_stripped_reasoning = ""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_block:
                close_idx, close_len = self._find_first_tag(buf, self._CLOSE_TAGS)
                if close_idx == -1:
                    visible = self._hold_partial(buf, self._CLOSE_TAGS)
                    if visible:
                        self._last_stripped_reasoning += visible
                    break
                if close_idx:
                    self._last_stripped_reasoning += buf[:close_idx]
                buf = buf[close_idx + close_len:]
                self._in_block = False
                continue

            pair = self._find_earliest_closed_pair(buf)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                self._last_stripped_reasoning += self._closed_pair_inner_text(buf[pair[0]:pair[1]])
                buf = buf[pair[1]:]
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                buf = buf[open_idx + open_len:]
                continue

            self._emit(out, self._hold_partial(buf, self._ALL_TAGS))
            break

        return "".join(out)

    def _hold_partial(self, buf: str, tags: Tuple[str, ...]) -> str:
        """Move a trailing partial-tag prefix into ``_buf``; return the remainder."""
        held = self._max_partial_suffix(buf, tags)
        self._buf = buf[-held:] if held else ""
        return buf[:-held] if held else buf

    def flush(self) -> str:
        """Flush a benign partial tag tail and reset the stream-boundary state."""
        if self._in_block:
            self._buf = ""
            self._in_block = False
            self._last_stripped_reasoning = ""
            self._last_emitted_ended_newline = True
            return ""
        tail = self._buf
        self._buf = ""
        self._in_block = False
        self._last_emitted_ended_newline = True
        return self._strip_orphan_close_tags(tail) if tail else ""

    @staticmethod
    def _find_first_tag(buf: str, tags: Tuple[str, ...]) -> Tuple[int, int]:
        """Return the earliest case-insensitive tag location, or ``(-1, 0)``."""
        buf_lower = buf.lower()
        hits = [(idx, len(tag)) for tag in tags if (idx := buf_lower.find(tag)) != -1]
        return min(hits) if hits else (-1, 0)

    def _find_earliest_closed_pair(self, buf: str) -> tuple[int, int] | None:
        """Return the earliest non-greedy ``<tag>...</tag>`` range, if any."""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = buf_lower.find(open_tag)
            close_idx = buf_lower.find(close_tag, open_idx + len(open_tag)) if open_idx != -1 else -1
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        return min(pairs) if pairs else None

    def _closed_pair_inner_text(self, segment: str) -> str:
        """Return the inner text of a closed reasoning tag segment."""
        segment_lower = segment.lower()
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = segment_lower.find(open_tag)
            if open_idx == -1:
                continue
            start = open_idx + len(open_tag)
            close_idx = segment_lower.find(close_tag, start)
            if close_idx != -1:
                return segment[start:close_idx]
        return ""

    def _find_open_at_boundary(self, buf: str, already_emitted: list[str]) -> Tuple[int, int]:
        """Return the earliest block-boundary open tag, or ``(-1, 0)``."""
        buf_lower = buf.lower()
        hits = []
        for tag in self._OPEN_TAGS:
            idx = buf_lower.find(tag)
            while idx != -1 and not self._is_block_boundary(buf, idx, already_emitted):
                idx = buf_lower.find(tag, idx + 1)
            if idx != -1:
                hits.append((idx, len(tag)))
        return min(hits) if hits else (-1, 0)

    def _is_block_boundary(self, buf: str, idx: int, already_emitted: list[str]) -> bool:
        prior_newline = already_emitted[-1].endswith("\n") if already_emitted else self._last_emitted_ended_newline
        if idx == 0:
            return prior_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        return (prior_newline if last_nl == -1 else True) and preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(cls, buf: str, tags: Tuple[str, ...]) -> int:
        """Return the longest suffix that is a strict prefix of a tag."""
        buf_lower = buf.lower()
        for i in range(min(len(buf_lower), cls._MAX_TAG_LEN - 1), 0, -1):
            suffix = buf_lower[-i:]
            if any(len(tag) > i and tag.startswith(suffix) for tag in tags):
                return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags with no matching open plus trailing whitespace."""
        return cls._ORPHAN_CLOSE_RE.sub("", text) if "</" in text else text

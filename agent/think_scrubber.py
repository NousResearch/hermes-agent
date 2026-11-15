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

__all__ = [
    "CODE_FENCE_OR_SPAN",
    "THINK_CLOSE_TAGS",
    "THINK_OPEN_TAGS",
    "THINK_TAG_NAMES",
    "StreamingThinkScrubber",
]

# Left-to-right regex alternation branch matching fenced code blocks or single-line inline code spans
# so post-stream regex strippers/extractors never treat backtick-quoted tag mentions in visible prose
# as real reasoning delimiters while still stripping code blocks nested INSIDE outer reasoning blocks.
CODE_FENCE_OR_SPAN: str = r"(```[\s\S]*?```|`[^`\n]+`)"

# The one list of model reasoning tag names. Every surface that hides reasoning (this scrubber,
# the CLI stream filter, the gateway stream filter, the final-response regex stripper) binds to
# these; a tag added here is covered everywhere. Consumers match case-insensitively, so the
# literal tags are lowercase. The CJK names cover models (MiniMax-M3) that emit Chinese reasoning
# tags: 思考 (think), 反思 (reflect), 推理 (reason), 推敲 (deliberate). MiniMax-M3 also emits the
# namespaced ``<mm:think>``; streaming surfaces match literal tags, so a namespaced name is listed whole.
THINK_TAG_NAMES: tuple[str, ...] = (
    "think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD",
    "思考", "反思", "推理", "推敲", "mm:think",
)
THINK_OPEN_TAGS: tuple[str, ...] = tuple(f"<{name.lower()}>" for name in THINK_TAG_NAMES)
THINK_CLOSE_TAGS: tuple[str, ...] = tuple(f"</{name.lower()}>" for name in THINK_TAG_NAMES)


class StreamingThinkScrubber:
    """Stateful scrubber for streaming reasoning/thinking blocks.

    State: ``_in_block`` (inside an open block; text discarded), ``_buf`` (held-back partial-tag
    tail), ``_last_emitted_ended_newline`` (True iff the last emission ended with ``\\n`` or nothing
    was emitted yet — decides whether an open tag at buffer position 0 sits at a block boundary).
    """

    # Literal tags so the hot path does string ops, not regex per feed().
    _OPEN_TAGS: tuple[str, ...] = THINK_OPEN_TAGS
    _CLOSE_TAGS: tuple[str, ...] = THINK_CLOSE_TAGS
    _ALL_TAGS: tuple[str, ...] = _OPEN_TAGS + _CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(tag) for tag in _ALL_TAGS)
    # Orphan close tag plus trailing whitespace (matches _strip_think_blocks case 3).
    _ORPHAN_CLOSE_RE = re.compile(
        rf"{CODE_FENCE_OR_SPAN}|(?<!`)(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")(?!`)[ \t\n\r]*",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True
        self._last_emitted_ended_backtick: bool = False
        self._last_hidden_ended_backtick: bool = False
        self._block_accum: str = ""
        self._emitted_accum: str = ""
        # Reasoning text the most recent feed() stripped from inside think blocks (tags excluded).
        self.last_hidden: str = ""

    def _emit(self, out: list[str], text: str) -> None:
        """Append visible prose to *out* (orphan close tags stripped) and track the newline flag."""
        text = self._strip_orphan_close_tags(text, self._last_emitted_ended_backtick, self._emitted_accum)
        if text:
            out.append(text)
            self._emitted_accum += text
            self._last_emitted_ended_newline = text.endswith("\n")
            self._last_emitted_ended_backtick = text.endswith("`")

    def _hide(self, hidden: list[str], text: str) -> None:
        """Append hidden reasoning text and track whether it ended with a backtick."""
        if text:
            hidden.append(text)
            self._last_hidden_ended_backtick = text.endswith("`")
            self._block_accum += text

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion ("" when it is all reasoning or held back)."""
        self.last_hidden = ""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []
        hidden: list[str] = []

        while buf:
            if self._in_block:
                prior_bt = hidden[-1].endswith("`") if hidden else self._last_hidden_ended_backtick
                close_idx, close_len = self._find_first_tag(
                    buf, self._CLOSE_TAGS, prior_bt, self._block_accum
                )
                if close_idx == -1:
                    # No close yet: hold back a possible partial close-tag prefix; the rest is reasoning.
                    self._hide(hidden, self._hold_partial(buf, self._CLOSE_TAGS))
                    break
                self._hide(hidden, buf[:close_idx])
                buf = buf[close_idx + close_len:]
                self._in_block = False
                self._last_hidden_ended_backtick = False
                self._block_accum = ""
                continue

            # Priority 1: closed <tag>X</tag> pair anywhere (even inline pairs are almost
            # certainly leaked reasoning). Priority 2: unterminated open tag at a block
            # boundary (gated so prose mentioning '<think>' isn't over-stripped). Earliest wins.
            prior_bt = out[-1].endswith("`") if out else self._last_emitted_ended_backtick
            pair = self._find_earliest_closed_pair(buf, prior_bt)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                # Pair tags are exact ``<name>``/``</name>``: inner text sits between them.
                self._hide(hidden, buf[buf.index(">", pair[0]) + 1:buf.rindex("<", pair[0], pair[1])])
                buf = buf[pair[1]:]
                self._last_hidden_ended_backtick = False
                self._block_accum = ""
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                self._last_hidden_ended_backtick = False
                self._block_accum = ""
                buf = buf[open_idx + open_len:]
                continue

            # No resolvable tag: hold back any partial-tag prefix at the tail
            # so a tag split across deltas isn't missed, then emit the rest.
            self._emit(out, self._hold_partial(buf, self._ALL_TAGS))
            break

        self.last_hidden = "".join(hidden)
        return "".join(out)

    def _hold_partial(self, buf: str, tags: tuple[str, ...]) -> str:
        """Move a trailing partial-tag prefix of *buf* into ``_buf``; return the remainder."""
        held = self._max_partial_suffix(buf, tags)
        self._buf = buf[-held:] if held else ""
        return buf[:-held] if held else buf

    def flush(self) -> str:
        """End-of-stream flush: inside an unterminated block the held-back content is discarded (leaking
        partial reasoning is worse than a truncated answer), otherwise the tail is emitted verbatim.
        Always resets the boundary flag — intra-turn retries flush then stream again without ``reset()``,
        and a stale False flag made the new stream's opening ``<think>`` look mid-line."""
        tail = "" if self._in_block else self._buf
        prior_bt = self._last_emitted_ended_backtick
        emitted_prefix = self._emitted_accum
        self._buf = ""
        self._in_block = False
        self._last_emitted_ended_newline = True
        self._last_emitted_ended_backtick = False
        self._last_hidden_ended_backtick = False
        self._block_accum = ""
        self._emitted_accum = ""
        return self._strip_orphan_close_tags(tail, prior_bt, emitted_prefix) if tail else ""

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _is_backtick_quoted(
        buf: str,
        idx: int,
        tag_len: int,
        prior_ended_backtick: bool = False,
        fence_prefix: str = "",
    ) -> bool:
        """True iff the tag at ``buf[idx:idx+tag_len]`` is adjacent to a backtick or inside an open ``` fence."""
        before_bt = prior_ended_backtick if idx == 0 else (buf[idx - 1] == "`")
        after_bt = (idx + tag_len < len(buf)) and (buf[idx + tag_len] == "`")
        if before_bt or after_bt:
            return True
        if (fence_prefix + buf[:idx]).count("```") % 2 == 1:
            return True
        return False

    @classmethod
    def _find_unquoted_tag(
        cls,
        buf_lower: str,
        buf: str,
        tag: str,
        start: int = 0,
        prior_ended_backtick: bool = False,
        fence_prefix: str = "",
        fence_start: int = 0,
    ) -> int:
        """Find earliest occurrence of *tag* at or after *start* that is not adjacent to a backtick or in a ``` fence."""
        idx = buf_lower.find(tag, start)
        tag_len = len(tag)
        while idx != -1 and cls._is_backtick_quoted(
            buf, idx, tag_len, prior_ended_backtick, fence_prefix if fence_start == 0 else ""
        ) or (idx != -1 and fence_start > 0 and buf.count("```", fence_start, idx) % 2 == 1):
            idx = buf_lower.find(tag, idx + 1)
        return idx

    @classmethod
    def _find_first_tag(
        cls,
        buf: str,
        tags: tuple[str, ...],
        prior_ended_backtick: bool = False,
        fence_prefix: str = "",
    ) -> tuple[int, int]:
        """Return (earliest_index, tag_length) over *tags* (case-insensitive), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = [
            (idx, len(tag))
            for tag in tags
            if (idx := cls._find_unquoted_tag(buf_lower, buf, tag, 0, prior_ended_backtick, fence_prefix)) != -1
        ]
        return min(hits) if hits else (-1, 0)

    def _find_earliest_closed_pair(self, buf: str, prior_ended_backtick: bool = False):
        """(start_idx, end_idx) of the earliest ``<tag>...</tag>`` pair (non-greedy, case-insensitive), else None."""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = self._find_unquoted_tag(buf_lower, buf, open_tag, 0, prior_ended_backtick)
            close_idx = (
                self._find_unquoted_tag(
                    buf_lower,
                    buf,
                    close_tag,
                    open_idx + len(open_tag),
                    False,
                    fence_start=open_idx + len(open_tag),
                )
                if open_idx != -1
                else -1
            )
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        return min(pairs) if pairs else None

    def _find_open_at_boundary(self, buf: str, already_emitted: list[str]) -> tuple[int, int]:
        """Return the earliest block-boundary open-tag (idx, len), or (-1, 0)."""
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
        """True iff *idx* is a block boundary: position 0 after a newline-terminated (or no) prior emission,
        or any position whose preceding text on the current line is whitespace-only (when no newline
        precedes it in *buf*, the prior emission must also have ended with a newline)."""
        prior_newline = already_emitted[-1].endswith("\n") if already_emitted else self._last_emitted_ended_newline
        if idx == 0:
            return prior_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        return (prior_newline if last_nl == -1 else True) and preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(cls, buf: str, tags: tuple[str, ...]) -> int:
        """Longest buf-suffix that is a strict prefix of any tag (full matches are real tags, handled elsewhere)."""
        buf_lower = buf.lower()
        for i in range(min(len(buf_lower), cls._MAX_TAG_LEN - 1), 0, -1):
            suffix = buf_lower[-i:]
            if any(len(tag) > i and tag.startswith(suffix) for tag in tags):
                return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(
        cls, text: str, prior_ended_backtick: bool = False, emitted_prefix: str = ""
    ) -> str:
        """Remove close tags with no matching open (always noise) plus trailing whitespace."""
        if "</" not in text:
            return text
        probe = f"`{text}" if prior_ended_backtick else text
        offset = 1 if prior_ended_backtick else 0

        def _replace(m: re.Match[str]) -> str:
            if m.group(1) is not None:
                return m.group(1)
            raw_start = max(0, m.start() - offset)
            full_before = emitted_prefix + text[:raw_start]
            if full_before.count("```") % 2 == 1:
                return m.group(0)
            if full_before.rsplit("\n", 1)[-1].count("`") % 2 == 1:
                return m.group(0)
            return ""

        cleaned = cls._ORPHAN_CLOSE_RE.sub(_replace, probe)
        return cleaned[1:] if prior_ended_backtick else cleaned

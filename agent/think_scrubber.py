"""Stateful scrubber for reasoning/thinking blocks in streamed assistant text.

The regex ``_strip_think_blocks`` is correct for a complete string but, run per-delta, erases an
opening ``<think>`` that arrives alone, so downstream state machines leak reasoning. This class
holds partial tags at delta boundaries until resolved; ``flush()`` releases held-back prose that
was not a tag; ``reset()`` at the top of each turn. An open tag only starts a block at a block
boundary (stream start / after a newline / whitespace-only line so far), so prose that *mentions*
``<think>`` is not suppressed; closed pairs are always suppressed (intentional).

The scrubber also strips the text-channel tool-call tag family (#137775): models whose relay
returned no native ``tool_calls`` serialize them as ``<tool_call>`` markup, which must not reach
stream consumers either. Open tags may carry attributes and an optional namespace prefix.
"""

from __future__ import annotations

import re
from typing import Tuple

__all__ = [
    "THINK_CLOSE_TAGS",
    "THINK_OPEN_TAGS",
    "THINK_TAG_NAMES",
    "TOOL_CALL_TAG_NAMES",
    "StreamingThinkScrubber",
]

# The one list of model reasoning tag names. Every surface that hides reasoning (this scrubber,
# the CLI stream filter, the gateway stream filter, the final-response regex stripper) binds to
# these; a tag added here is covered everywhere. Consumers match case-insensitively, so the
# literal tags are lowercase. The CJK names cover models (MiniMax-M3) that emit Chinese reasoning
# tags: 思考 (think), 反思 (reflect), 推理 (reason), 推敲 (deliberate).
THINK_TAG_NAMES: tuple[str, ...] = (
    "think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD",
    "思考", "反思", "推理", "推敲",
)
THINK_OPEN_TAGS: tuple[str, ...] = tuple(f"<{name.lower()}>" for name in THINK_TAG_NAMES)
THINK_CLOSE_TAGS: tuple[str, ...] = tuple(f"</{name.lower()}>" for name in THINK_TAG_NAMES)

# The text-channel tool-call tag family — models that serialize tool calls as XML because the
# provider/relay returned no native ``tool_calls`` channel. Kept beside THINK_TAG_NAMES so the
# streaming scrubber and the whole-string stripper (``agent_runtime_helpers``) bind to one list;
# a tag added here is covered everywhere. Open tags may carry attributes (``<tool_call id=…>``)
# and an optional XML namespace prefix (``<ns:function_calls>``); closers are bare. Claude-style
# payloads nest ``<invoke>``/``<parameter>`` inside the block; only their orphan closers ever
# appear outside one, so they need no open handling (#137775).
TOOL_CALL_TAG_NAMES: tuple[str, ...] = (
    "tool_call",
    "tool_calls",
    "tool_result",
    "function_call",
    "function_calls",
)
_NS_PREFIX = r"(?:[\w.-]+:)?"
TOOL_CALL_OPEN_TAG_RE = re.compile(
    rf"<{_NS_PREFIX}(?:{'|'.join(TOOL_CALL_TAG_NAMES)})\b[^>]*>", re.IGNORECASE
)
TOOL_CALL_CLOSE_TAG_RE = re.compile(
    rf"</{_NS_PREFIX}(?:{'|'.join(TOOL_CALL_TAG_NAMES)})>", re.IGNORECASE
)
TOOL_CALL_BLOCK_RE = re.compile(
    rf"<{_NS_PREFIX}(?:{'|'.join(TOOL_CALL_TAG_NAMES)})\b[^>]*>"
    rf".*?</{_NS_PREFIX}(?:{'|'.join(TOOL_CALL_TAG_NAMES)})>",
    re.DOTALL | re.IGNORECASE,
)
# An open tag whose ``>`` has not arrived yet: either the name has fully arrived and attributes
# may still be streaming in, or the name itself (with its optional namespace) is still streaming.
# A delta tail matching either must be held back exactly like the literal partial-tag suffixes;
# ``flush()``/the next delta release it when it turns out to be prose.
_TOOL_CALL_OPEN_PARTIAL_RE = re.compile(
    rf"<{_NS_PREFIX}(?:{'|'.join(TOOL_CALL_TAG_NAMES)})\b[^>]*$"
    r"|<[^\s>]*$",
    re.IGNORECASE,
)


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
    # Closers whose *strict prefixes* are held back at a delta tail. Tool-call closers are
    # literal (namespace-prefixed ones are rare enough to leave to the whole-string strip);
    # tool-call open tags are regex-matched (they carry attributes), so their partial form is
    # detected by ``_TOOL_CALL_OPEN_PARTIAL_RE`` in ``_hold_partial`` instead — the bare
    # ``<tool_call>`` literals below still cover the name-while-streaming prefix case.
    _TC_HOLD_CLOSE_TAGS: tuple[str, ...] = tuple(
        f"</{name}>" for name in TOOL_CALL_TAG_NAMES
    ) + (
        "</invoke>",
        "</parameter>",
    )
    _HOLD_TAGS: tuple[str, ...] = (
        _ALL_TAGS
        + _TC_HOLD_CLOSE_TAGS
        + tuple(f"<{name}>" for name in TOOL_CALL_TAG_NAMES)
    )
    _HOLD_CLOSE_TAGS: tuple[str, ...] = _CLOSE_TAGS + _TC_HOLD_CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(tag) for tag in _HOLD_TAGS)
    # Orphan close tag plus trailing whitespace (matches _strip_think_blocks case 3).
    # Tool-call closers (namespaced or not) and stray invoke/parameter closers from a
    # degenerate tail are noise the same way (#137775).
    _ORPHAN_CLOSE_RE = re.compile(
        "(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS)
        + f"|</{_NS_PREFIX}(?:{'|'.join(TOOL_CALL_TAG_NAMES)}|invoke|parameter)>"
        + r")[ \t\n\r]*",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True
        # Reasoning text the most recent feed() stripped from inside think blocks (tags excluded).
        self.last_hidden: str = ""

    def _emit(self, out: list[str], text: str) -> None:
        """Append visible prose to *out* (orphan close tags stripped) and track the newline flag."""
        text = self._strip_orphan_close_tags(text)
        if text:
            out.append(text)
            self._last_emitted_ended_newline = text.endswith("\n")

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
                close_idx, close_len = self._find_close_tag(buf)
                if close_idx == -1:
                    # No close yet: hold back a possible partial close-tag prefix; the rest is reasoning.
                    hidden.append(self._hold_partial(buf, self._HOLD_CLOSE_TAGS))
                    break
                hidden.append(buf[:close_idx])
                buf = buf[close_idx + close_len:]
                self._in_block = False
                continue

            # Priority 1: closed <tag>X</tag> pair anywhere (even inline pairs are almost
            # certainly leaked reasoning). Priority 2: unterminated open tag at a block
            # boundary (gated so prose mentioning '<think>' isn't over-stripped). Earliest wins.
            # Both scans cover the tool-call family too (#137775).
            pair = self._find_earliest_closed_pair(buf)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                # Pair tags are exact ``<name>``/``</name>``: inner text sits between them.
                hidden.append(buf[buf.index(">", pair[0]) + 1:buf.rindex("<", pair[0], pair[1])])
                buf = buf[pair[1]:]
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                buf = buf[open_idx + open_len:]
                continue

            # No resolvable tag: hold back any partial-tag prefix at the tail
            # so a tag split across deltas isn't missed, then emit the rest.
            self._emit(out, self._hold_partial(buf, self._HOLD_TAGS, open_partial=True))
            break

        self.last_hidden = "".join(hidden)
        return "".join(out)

    def _hold_partial(self, buf: str, tags: tuple[str, ...], *, open_partial: bool = False) -> str:
        """Move a trailing partial-tag prefix of *buf* into ``_buf``; return the remainder."""
        held = self._max_partial_suffix(buf, tags)
        if open_partial and held < len(buf):
            m = _TOOL_CALL_OPEN_PARTIAL_RE.search(buf)
            if m is not None:
                held = max(held, len(buf) - m.start())
        self._buf = buf[-held:] if held else ""
        return buf[:-held] if held else buf

    def flush(self) -> str:
        """End-of-stream flush: inside an unterminated block the held-back content is discarded (leaking
        partial reasoning is worse than a truncated answer), otherwise the tail is emitted verbatim.
        Always resets the boundary flag — intra-turn retries flush then stream again without ``reset()``,
        and a stale False flag made the new stream's opening ``<think>`` look mid-line."""
        tail = "" if self._in_block else self._buf
        self._buf = ""
        self._in_block = False
        self._last_emitted_ended_newline = True
        return self._strip_orphan_close_tags(tail) if tail else ""

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _find_first_tag(buf: str, tags: tuple[str, ...]) -> tuple[int, int]:
        """Return (earliest_index, tag_length) over *tags* (case-insensitive), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = [(idx, len(tag)) for tag in tags if (idx := buf_lower.find(tag)) != -1]
        return min(hits) if hits else (-1, 0)

    def _find_close_tag(self, buf: str) -> tuple[int, int]:
        """Earliest reasoning close tag or tool-call close tag (namespaced ok), or (-1, 0)."""
        idx, length = self._find_first_tag(buf, self._CLOSE_TAGS)
        m = TOOL_CALL_CLOSE_TAG_RE.search(buf)
        if m is not None and (idx == -1 or m.start() < idx):
            idx, length = m.start(), m.end() - m.start()
        return idx, length

    def _find_earliest_closed_pair(self, buf: str):
        """(start_idx, end_idx) of the earliest ``<tag>...</tag>`` pair (non-greedy, case-insensitive), else None."""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = buf_lower.find(open_tag)
            close_idx = buf_lower.find(close_tag, open_idx + len(open_tag)) if open_idx != -1 else -1
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        m = TOOL_CALL_BLOCK_RE.search(buf)
        if m is not None:
            pairs.append((m.start(), m.end()))
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
        m = TOOL_CALL_OPEN_TAG_RE.search(buf)
        while m is not None and not self._is_block_boundary(buf, m.start(), already_emitted):
            m = TOOL_CALL_OPEN_TAG_RE.search(buf, m.end())
        if m is not None:
            hits.append((m.start(), m.end() - m.start()))
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
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags with no matching open (always noise) plus trailing whitespace."""
        return cls._ORPHAN_CLOSE_RE.sub("", text) if "</" in text else text

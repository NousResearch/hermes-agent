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
    "StreamingThinkScrubber", "THINK_TAG_NAMES", "THINK_OPEN_TAGS", "THINK_CLOSE_TAGS",
    "DSML_LINE_RE", "DSML_BLOCK_OPEN_RE", "DSML_BLOCK_CLOSE_RE",
]

# The one list of model reasoning tag names. Every surface that hides reasoning (this scrubber,
# the CLI stream filter, the gateway stream filter, the final-response regex stripper) binds to
# these; a tag added here is covered everywhere. Consumers match case-insensitively, so the
# literal tags are lowercase.
THINK_TAG_NAMES: Tuple[str, ...] = ("think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD")
THINK_OPEN_TAGS: Tuple[str, ...] = tuple(f"<{name.lower()}>" for name in THINK_TAG_NAMES)
THINK_CLOSE_TAGS: Tuple[str, ...] = tuple(f"</{name.lower()}>" for name in THINK_TAG_NAMES)

# The one set of DeepSeek DSML tool-call markup recognizers (#115475). DSML is line-based
# (``DSML | tool_calls`` … ``DSML | /tool_calls``, optionally angle-bracketed ``<DSML | …>``)
# and leaks into visible text on deepseek models served through OpenAI-compatible gateways.
# Every surface that strips it (this scrubber, the final-response regex stripper
# ``strip_dsml_blocks``, the gateway sanitizer, the CLI display stripper) binds to these.
DSML_LINE_RE = re.compile(r"^\s*<?DSML\s*\|\s*", re.IGNORECASE)
DSML_BLOCK_OPEN_RE = re.compile(
    r"^\s*<?DSML\s*\|\s*(?:tool_calls|function_results)\b", re.IGNORECASE,
)
DSML_BLOCK_CLOSE_RE = re.compile(
    r"^\s*<?DSML\s*\|\s*/(?:tool_calls|function_results)\b", re.IGNORECASE,
)


class StreamingThinkScrubber:
    """Stateful scrubber for streaming reasoning/thinking blocks.

    State: ``_in_block`` (inside an open block; text discarded), ``_buf`` (held-back partial-tag
    tail), ``_last_emitted_ended_newline`` (True iff the last emission ended with ``\\n`` or nothing
    was emitted yet — decides whether an open tag at buffer position 0 sits at a block boundary).
    """

    # Literal tags so the hot path does string ops, not regex per feed().
    _OPEN_TAGS: Tuple[str, ...] = THINK_OPEN_TAGS
    _CLOSE_TAGS: Tuple[str, ...] = THINK_CLOSE_TAGS
    _ALL_TAGS: Tuple[str, ...] = _OPEN_TAGS + _CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(tag) for tag in _ALL_TAGS)
    # Orphan close tag plus trailing whitespace (matches _strip_think_blocks case 3).
    _ORPHAN_CLOSE_RE = re.compile(
        "(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")[ \t\n\r]*", re.IGNORECASE
    )
    # DSML recognition binds to the module-level one-set recognizers (#115475).
    _DSML_LINE_RE = DSML_LINE_RE
    _DSML_BLOCK_OPEN_RE = DSML_BLOCK_OPEN_RE
    _DSML_BLOCK_CLOSE_RE = DSML_BLOCK_CLOSE_RE

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True
        self._dsml_line: str = ""
        self._in_dsml_block: bool = False

    def _emit(self, out: list[str], text: str) -> None:
        """Append visible prose to *out* (orphan close tags stripped) and track the newline flag."""
        text = self._strip_orphan_close_tags(text)
        if text:
            out.append(text)
            self._last_emitted_ended_newline = text.endswith("\n")

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion ("" when it is all reasoning or held back)."""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        buf = self._prefilter_dsml(buf)
        if not buf:
            return ""
        out: list[str] = []

        while buf:
            if self._in_block:
                close_idx, close_len = self._find_first_tag(buf, self._CLOSE_TAGS)
                if close_idx == -1:
                    # No close yet: hold back a possible partial close-tag prefix, drop the rest.
                    self._hold_partial(buf, self._CLOSE_TAGS)
                    break
                buf = buf[close_idx + close_len:]
                self._in_block = False
                continue

            # Priority 1: closed <tag>X</tag> pair anywhere (even inline pairs are almost
            # certainly leaked reasoning). Priority 2: unterminated open tag at a block
            # boundary (gated so prose mentioning '<think>' isn't over-stripped). Earliest wins.
            pair = self._find_earliest_closed_pair(buf)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                buf = buf[pair[1]:]
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                buf = buf[open_idx + open_len:]
                continue

            # No resolvable tag: hold back any partial-tag prefix at the tail
            # so a tag split across deltas isn't missed, then emit the rest.
            self._emit(out, self._hold_partial(buf, self._ALL_TAGS))
            break

        return "".join(out)

    def _hold_partial(self, buf: str, tags: Tuple[str, ...]) -> str:
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
        self._buf = ""
        self._in_block = False
        # A held DSML partial line that never completed into a directive is prose; inside an
        # unterminated DSML block the tail is markup and stays dropped (#115475).
        dsml_tail = "" if self._in_dsml_block else self._dsml_line
        self._dsml_line = ""
        self._in_dsml_block = False
        self._last_emitted_ended_newline = True
        if dsml_tail:
            tail = tail + dsml_tail
        return self._strip_orphan_close_tags(tail) if tail else ""

    # ── DSML line prefilter (#115475) ───────────────────────────────────

    def _prefilter_dsml(self, buf: str) -> str:
        """Strip DeepSeek DSML markup lines from *buf*, holding the trailing partial line.

        Runs before the think-tag state machine on the same buffer. Every line whose
        stripped form starts with the ``DSML |`` marker (``<DSML |`` included) is markup:
        block openers (``tool_calls`` / ``function_results``) start a discard state that
        swallows plain value lines until the matching closer; stray directives outside a
        block are dropped outright. A trailing line without a newline is held back when it
        could still grow into a DSML directive, so a line split across deltas is handled;
        ``flush()`` releases it as prose if it never completed.
        """
        if not self._in_dsml_block and not self._dsml_line and "dsml" not in buf.lower():
            return buf
        if self._dsml_line:
            buf = self._dsml_line + buf
            self._dsml_line = ""
        if "\n" not in buf:
            s = buf.strip()
            if self._DSML_BLOCK_CLOSE_RE.match(s):
                self._in_dsml_block = False
                return ""
            if self._in_dsml_block:
                return ""
            if self._DSML_BLOCK_OPEN_RE.match(s):
                self._in_dsml_block = True
                return ""
            if self._could_be_dsml_line(buf):
                self._dsml_line = buf
                return ""
            if self._DSML_LINE_RE.match(s):
                return ""
            return buf
        out: list[str] = []
        lines = buf.split("\n")
        held = False
        for i, line in enumerate(lines):
            last = i == len(lines) - 1
            s = line.strip()
            if last and line:
                if self._DSML_BLOCK_CLOSE_RE.match(s):
                    self._in_dsml_block = False
                    continue
                if self._in_dsml_block:
                    continue
                if self._DSML_BLOCK_OPEN_RE.match(s):
                    self._in_dsml_block = True
                    continue
                if self._could_be_dsml_line(line):
                    self._dsml_line = line
                    held = True
                    continue
                if self._DSML_LINE_RE.match(s):
                    continue
                out.append(line)
                continue
            if self._in_dsml_block:
                if self._DSML_BLOCK_CLOSE_RE.match(s):
                    self._in_dsml_block = False
                continue
            if self._DSML_BLOCK_OPEN_RE.match(s):
                self._in_dsml_block = True
                continue
            if self._DSML_LINE_RE.match(s):
                continue
            out.append(line)
        # Holding the tail must not swallow the newline that terminated the last kept
        # line — downstream boundary detection (think tags, markdown) depends on it.
        return "\n".join(out) + ("\n" if held and out else "")

    @classmethod
    def _could_be_dsml_line(cls, line: str) -> bool:
        """True when *line* is a strict prefix of a DSML directive and could still complete.

        A line that IS a directive (matches ``_DSML_LINE_RE`` with a complete command, or is
        a full block opener/closer) returns False — callers process those immediately. Only
        partial forms (``DSML``, ``<DSML |``, ``DSML | tool``, ``DSML | /tool``, …) are held
        so a block split across deltas isn't missed.
        """
        s = line.strip().lower()
        if not s:
            return False
        core = s[1:] if s.startswith("<") else s
        if core.startswith("dsml"):
            rest = core[4:].lstrip()
            if not rest or rest == "|":
                return True
            if rest.startswith("|"):
                cmd = rest[1:].strip()
                if cmd.startswith("/"):
                    cmd = cmd[1:].lstrip()
                if not cmd:
                    return True
                return any(
                    keyword.startswith(cmd) and len(keyword) > len(cmd)
                    for keyword in ("tool_calls", "function_results")
                )
        return False

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _find_first_tag(buf: str, tags: Tuple[str, ...]) -> Tuple[int, int]:
        """Return (earliest_index, tag_length) over *tags* (case-insensitive), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = [(idx, len(tag)) for tag in tags if (idx := buf_lower.find(tag)) != -1]
        return min(hits) if hits else (-1, 0)

    def _find_earliest_closed_pair(self, buf: str):
        """(start_idx, end_idx) of the earliest ``<tag>...</tag>`` pair (non-greedy, case-insensitive), else None."""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = buf_lower.find(open_tag)
            close_idx = buf_lower.find(close_tag, open_idx + len(open_tag)) if open_idx != -1 else -1
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        return min(pairs) if pairs else None

    def _find_open_at_boundary(self, buf: str, already_emitted: list[str]) -> Tuple[int, int]:
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
    def _max_partial_suffix(cls, buf: str, tags: Tuple[str, ...]) -> int:
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

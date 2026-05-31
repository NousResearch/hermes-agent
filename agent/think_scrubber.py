"""Stateful scrubber for reasoning/thinking blocks in streamed assistant text.

``run_agent._strip_think_blocks`` is regex-based and correct for a complete
string, but when it runs *per-delta* in ``_fire_stream_delta`` it destroys
the state that downstream consumers (CLI ``_stream_delta``, gateway
``GatewayStreamConsumer._filter_and_accumulate``) rely on.

Concretely, when MiniMax-M2.7 streams

    delta1 = "<think>"
    delta2 = "Let me check their config"
    delta3 = "</think>"

the per-delta regex erases delta1 entirely (case 2: unterminated-open at
boundary matches ``^<think>...``), so the downstream state machine never
sees the open tag, treats delta2 as regular content, and leaks reasoning
to the user.  Consumers that don't run their own state machine (ACP,
api_server, TTS) never had any defence at all — they just emitted
whatever survived the upstream regex.

This module centralises the tag-suppression state machine at the
upstream layer so every stream_delta_callback sees text that has
already had reasoning blocks removed.  Partial tags at delta
boundaries are held back until the next delta resolves them, and
end-of-stream flushing surfaces any held-back prose that turned out
not to be a real tag.

Usage::

    scrubber = StreamingThinkScrubber()
    for delta in stream:
        visible = scrubber.feed(delta)
        if visible:
            emit(visible)
    tail = scrubber.flush()  # at end of stream
    if tail:
        emit(tail)

The scrubber is re-entrant per agent instance.  Call ``reset()`` at
the top of each new turn so a hung block from an interrupted prior
stream cannot taint the next turn's output.

Tag variants handled (case-insensitive):
  ``<think>``, ``<thinking>``, ``<reasoning>``, ``<thought>``,
  ``<REASONING_SCRATCHPAD>``.

Block-boundary rule for opens: an opening tag is only treated as a
reasoning-block opener when it appears at the start of the stream,
after a newline (optionally followed by whitespace), or when only
whitespace has been emitted on the current line.  This prevents prose
that *mentions* the tag name (e.g. ``"use <think> tags here"``) from
being incorrectly suppressed.  Closed pairs (``<think>X</think>``) are
always suppressed regardless of boundary; a closed pair is an
intentional, bounded construct.
"""

from __future__ import annotations

from typing import Tuple

__all__ = ["StreamingThinkScrubber"]


class StreamingThinkScrubber:
    """Stateful scrubber for streaming reasoning/thinking blocks.

    State machine:
      - ``_in_block``: True while inside an opened block, waiting for
        a close tag.  All text inside is discarded.
      - ``_buf``: held-back partial-tag tail.  Emitted / discarded on
        the next ``feed()`` call or by ``flush()``.
      - ``_last_emitted_ended_newline``: True iff the most recent
        emission to the consumer ended with ``\\n``, or nothing has
        been emitted yet (start-of-stream counts as a boundary).  Used
        to decide whether an open tag at buffer position 0 is at a
        block boundary.
    """

    _OPEN_TAG_NAMES: Tuple[str, ...] = (
        "think",
        "thinking",
        "reasoning",
        "thought",
        "REASONING_SCRATCHPAD",
    )

    # Materialise literal tag strings so the hot path does string
    # operations, not regex compilation per feed().
    _OPEN_TAGS: Tuple[str, ...] = tuple(f"<{name}>" for name in _OPEN_TAG_NAMES)
    _CLOSE_TAGS: Tuple[str, ...] = tuple(f"</{name}>" for name in _OPEN_TAG_NAMES)

    # Pre-compute the longest tag (for partial-tag hold-back bound).
    _MAX_TAG_LEN: int = max(len(tag) for tag in _OPEN_TAGS + _CLOSE_TAGS)

    def __init__(self) -> None:
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._in_block = False
        self._buf = ""
        self._last_emitted_ended_newline = True

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion.

        May return an empty string when the entire delta is reasoning
        content or is being held back pending resolution of a partial
        tag at the boundary.
        """
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_block:
                # Hunt for the earliest close tag.
                close_idx, close_len = self._find_first_tag(
                    buf, self._CLOSE_TAGS,
                )
                if close_idx == -1:
                    # No close yet — hold back a potential partial
                    # close-tag prefix; discard everything else.
                    held = self._max_partial_suffix(buf, self._CLOSE_TAGS)
                    self._buf = buf[-held:] if held else ""
                    return "".join(out)
                # Found close: discard block content + tag, continue.
                buf = buf[close_idx + close_len:]
                self._in_block = False
            else:
                # Priority 1 — closed <tag>X</tag> pair anywhere in
                # buf.  Closed pairs are always an intentional,
                # bounded construct (even mid-line prose containing
                # an open/close pair is almost certainly a model
                # leaking reasoning inline), so no boundary gating.
                pair = self._find_earliest_closed_pair(buf)
                # Priority 2 — unterminated open tag at a block
                # boundary.  Boundary-gated so prose that mentions
                # '<think>' isn't over-stripped.
                open_idx, open_len = self._find_open_at_boundary(
                    buf, out,
                )

                # Pick whichever match comes earliest in the buffer.
                if pair is not None and (
                    open_idx == -1 or pair[0] <= open_idx
                ):
                    start_idx, end_idx = pair
                    preceding = buf[:start_idx]
                    if preceding:
                        preceding = self._strip_orphan_close_tags(preceding)
                        if preceding:
                            out.append(preceding)
                            self._last_emitted_ended_newline = (
                                preceding.endswith("\n")
                            )
                    buf = buf[end_idx:]
                    continue

                if open_idx != -1:
                    # Unterminated open at boundary — emit preceding,
                    # enter block, continue loop with remainder.
                    preceding = buf[:open_idx]
                    if preceding:
                        preceding = self._strip_orphan_close_tags(preceding)
                        if preceding:
                            out.append(preceding)
                            self._last_emitted_ended_newline = (
                                preceding.endswith("\n")
                            )
                    self._in_block = True
                    buf = buf[open_idx + open_len:]
                    continue

                # No resolvable tag structure in buf.  Hold back any
                # partial-tag prefix at the tail so a split tag
                # across deltas isn't missed, then emit the rest.
                held = self._max_partial_suffix(buf, self._OPEN_TAGS)
                held_close = self._max_partial_suffix(
                    buf, self._CLOSE_TAGS,
                )
                held = max(held, held_close)
                if held:
                    emit_text = buf[:-held]
                    self._buf = buf[-held:]
                else:
                    emit_text = buf
                    self._buf = ""
                if emit_text:
                    emit_text = self._strip_orphan_close_tags(emit_text)
                    if emit_text:
                        out.append(emit_text)
                        self._last_emitted_ended_newline = (
                            emit_text.endswith("\n")
                        )
                return "".join(out)

        return "".join(out)

    def flush(self) -> str:
        """End-of-stream flush.

        If still inside an unterminated block, held-back content is
        discarded — leaking partial reasoning is worse than a
        truncated answer.  Otherwise the held-back partial-tag tail is
        emitted verbatim (it turned out not to be a real tag prefix).
        """
        if self._in_block:
            self._buf = ""
            self._in_block = False
            return ""
        tail = self._buf
        self._buf = ""
        if not tail:
            return ""
        tail = self._strip_orphan_close_tags(tail)
        if tail:
            self._last_emitted_ended_newline = tail.endswith("\n")
        return tail

    # ── internal helpers ───────────────────────────────────────────────

    @staticmethod
    def _find_first_tag(
        buf: str, tags: Tuple[str, ...],
    ) -> Tuple[int, int]:
        """Return (earliest_index, tag_length) over *tags*, or (-1, 0).

        Case-insensitive match.
        """
        buf_lower = buf.lower()
        best_idx = -1
        best_len = 0
        for tag in tags:
            idx = buf_lower.find(tag.lower())
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_len = len(tag)
        return best_idx, best_len

    def _find_earliest_closed_pair(self, buf: str):
        """Return (start_idx, end_idx) of the earliest closed pair, else None.

        A closed pair is ``<tag>...</tag>`` of any variant.  Matches are
        case-insensitive and non-greedy (the closest close tag after
        an open tag wins), matching the regex ``<tag>.*?</tag>``
        semantics of ``_strip_think_blocks`` case 1.  When two tag
        variants could both match, the one whose open tag appears
        earlier wins.
        """
        buf_lower = buf.lower()
        best: "tuple[int, int] | None" = None
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_lower = open_tag.lower()
            close_lower = close_tag.lower()
            open_idx = buf_lower.find(open_lower)
            if open_idx == -1:
                continue
            close_idx = buf_lower.find(
                close_lower, open_idx + len(open_lower),
            )
            if close_idx == -1:
                continue
            end_idx = close_idx + len(close_lower)
            if best is None or open_idx < best[0]:
                best = (open_idx, end_idx)
        return best

    def _find_open_at_boundary(
        self, buf: str, already_emitted: list[str],
    ) -> Tuple[int, int]:
        """Return the earliest block-boundary open-tag (idx, len).

        Returns (-1, 0) if no boundary-legal opener is present.
        """
        buf_lower = buf.lower()
        best_idx = -1
        best_len = 0
        for tag in self._OPEN_TAGS:
            tag_lower = tag.lower()
            search_start = 0
            while True:
                idx = buf_lower.find(tag_lower, search_start)
                if idx == -1:
                    break
                if self._is_block_boundary(buf, idx, already_emitted):
                    if best_idx == -1 or idx < best_idx:
                        best_idx = idx
                        best_len = len(tag)
                    break  # first boundary hit for this tag is enough
                search_start = idx + 1
        return best_idx, best_len

    def _is_block_boundary(
        self, buf: str, idx: int, already_emitted: list[str],
    ) -> bool:
        """True iff position *idx* in *buf* is a block boundary.

        A block boundary is:
          - buf position 0 AND the most recent emission ended with
            a newline (or nothing has been emitted yet)
          - any position whose preceding text on the current line
            (since the last newline in buf) is whitespace-only, AND
            if there is no newline in the preceding buf portion, the
            most recent prior emission ended with a newline
        """
        if idx == 0:
            # Check whether the last already-emitted chunk in THIS
            # feed() call ended with a newline, otherwise fall back
            # to the cross-feed flag.
            if already_emitted:
                return already_emitted[-1].endswith("\n")
            return self._last_emitted_ended_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        if last_nl == -1:
            # No newline in buf before the tag — boundary only if the
            # prior emission ended with a newline AND everything since
            # is whitespace.
            if already_emitted:
                prior_newline = already_emitted[-1].endswith("\n")
            else:
                prior_newline = self._last_emitted_ended_newline
            return prior_newline and preceding.strip() == ""
        # Newline present — text between it and the tag must be
        # whitespace-only.
        return preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(
        cls, buf: str, tags: Tuple[str, ...],
    ) -> int:
        """Return the longest buf-suffix that is a prefix of any tag.

        Only prefixes strictly shorter than the tag itself count
        (full-length suffixes are the tag and are handled as matches,
        not held-back partials).  Case-insensitive.
        """
        if not buf:
            return 0
        buf_lower = buf.lower()
        max_check = min(len(buf_lower), cls._MAX_TAG_LEN - 1)
        for i in range(max_check, 0, -1):
            suffix = buf_lower[-i:]
            for tag in tags:
                tag_lower = tag.lower()
                if len(tag_lower) > i and tag_lower.startswith(suffix):
                    return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove any close tags from *text* (orphan-close handling).

        An orphan close tag has no matching open in the current
        scrubber state; it's always noise, stripped with any trailing
        whitespace so the surrounding prose flows naturally.
        """
        if "</" not in text:
            return text
        text_lower = text.lower()
        out: list[str] = []
        i = 0
        while i < len(text):
            matched = False
            if text_lower[i:i + 2] == "</":
                for tag in cls._CLOSE_TAGS:
                    tag_lower = tag.lower()
                    tag_len = len(tag_lower)
                    if text_lower[i:i + tag_len] == tag_lower:
                        # Skip the tag and any trailing whitespace,
                        # matching _strip_think_blocks case 3.
                        j = i + tag_len
                        while j < len(text) and text[j] in " \t\n\r":
                            j += 1
                        i = j
                        matched = True
                        break
            if not matched:
                out.append(text[i])
                i += 1
        return "".join(out)

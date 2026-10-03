"""Think-block filtering for GatewayStreamConsumer.

Some models emit inline <think>...</think> blocks in content.  The agent strips
them from the final response, but intermediate edits go out before that, so this
mirrors the CLI's _stream_delta state machine; tag primitives are shared with
``agent/think_scrubber.py`` so the progressive display matches the post-stream scrubber."""

from __future__ import annotations

import logging

from agent.think_scrubber import (
    _CLOSE_TAG_RE,
    _OPEN_TAG_RE,
    StreamingThinkScrubber as _Scrubber,
)

logger = logging.getLogger("gateway.stream_consumer")


class StreamThinkFilterMixin:
    """Progressive <think>-tag suppression over streamed deltas."""

    def _at_block_boundary(self, buf: str, idx: int) -> bool:
        """Tag at ``idx`` starts a block: start of text, or newline + optional whitespace.

        Prose that merely *mentions* a tag must not trigger.
        """
        acc_boundary = not self._accumulated or self._accumulated.endswith("\n")
        if idx == 0:
            return acc_boundary
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        if last_nl == -1:
            return acc_boundary and preceding.strip() == ""
        return preceding[last_nl + 1:].strip() == ""

    def _earliest_open_tag(self, buf: str) -> "tuple[int, int]":
        """(index, length) of the earliest block-boundary opening tag, or (-1, 0)."""
        for m in _OPEN_TAG_RE.finditer(buf):
            if self._at_block_boundary(buf, m.start()):
                return m.start(), m.end() - m.start()
        return -1, 0

    def _filter_and_accumulate(self, text: str) -> None:
        """Append a delta to the buffer, discarding think blocks.

        Partial tags at buffer boundaries are held in ``_think_buffer`` until
        enough characters arrive to decide.
        """
        buf = self._think_buffer + text
        self._think_buffer = ""

        while buf:
            if self._in_think_block:
                close_m = _CLOSE_TAG_RE.search(buf)
                if close_m is not None:
                    self._in_think_block = False
                    buf = buf[close_m.end():]
                else:
                    # Hold a tail that could be a partial close tag; discard the rest.
                    self._think_buffer = _Scrubber._partial_tag_tail(buf, closing=True)
                    return
            else:
                # Priority 1: a closed pair anywhere (matching the scrubber — inline pairs are
                # almost certainly leaked reasoning). Priority 2: an open tag at a block boundary.
                pair = _Scrubber._find_earliest_closed_pair(buf)
                best_idx, best_len = self._earliest_open_tag(buf)
                if pair is not None and (best_idx == -1 or pair[0] <= best_idx):
                    self._append_accumulated(buf[:pair[0]])
                    buf = buf[pair[1]:]
                    continue
                if best_len:
                    self._append_accumulated(buf[:best_idx])
                    self._in_think_block = True
                    buf = buf[best_idx + best_len:]
                else:
                    # Hold back a partial open tag at the tail.
                    held_back = _Scrubber._partial_tag_tail(buf, closing=False)
                    if held_back:
                        self._append_accumulated(buf[:-len(held_back)])
                        self._think_buffer = held_back
                    else:
                        # An orphan </think> (thinking-mode toggle dropped the open, or
                        # incomplete upstream stripping) is noise.
                        self._append_accumulated(self._strip_orphan_close_tags(buf))
                    return

    @staticmethod
    def _strip_orphan_close_tags(text: str) -> str:
        """Remove close tags (plus trailing whitespace) that have no matching open."""
        return _Scrubber._strip_orphan_close_tags(text)

    def _flush_think_buffer(self) -> None:
        """On stream end, flush text held back waiting for a possible open tag."""
        if self._think_buffer and not self._in_think_block:
            self._append_accumulated(self._strip_orphan_close_tags(self._think_buffer))
            self._think_buffer = ""

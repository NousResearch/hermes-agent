"""Think-block filtering for GatewayStreamConsumer.

Some models emit inline <think>...</think> blocks in content.  The agent strips
them from the final response, but intermediate edits go out before that, so this
mirrors the CLI's _stream_delta state machine; tag primitives are shared with
``agent/think_scrubber.py`` so the progressive display matches the post-stream scrubber."""

from __future__ import annotations

import logging

from agent.think_scrubber import PENDING_RETAIN_CAP, THINK_CLOSE_TAGS, THINK_OPEN_TAGS
from agent.think_scrubber import StreamingThinkScrubber as _Scrubber

logger = logging.getLogger("gateway.stream_consumer")


class StreamThinkFilterMixin:
    """Progressive <think>-tag suppression over streamed deltas."""

    _OPEN_THINK_TAGS = THINK_OPEN_TAGS
    _CLOSE_THINK_TAGS = THINK_CLOSE_TAGS
    # Pending latch (#128294): a mid-line open held until its close completes.
    _pending_think_name = ""
    _pending_think_text = ""
    _pending_think_buf = ""

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

    def _earliest_open_tag(self, buf: str, lower_buf: str) -> "tuple[int, int]":
        """(index, length) of the earliest block-boundary opening tag, or (-1, 0)."""
        best_idx, best_len = -1, 0
        for tag in self._OPEN_THINK_TAGS:
            tag_lower = tag.lower()
            search_start = 0
            while (idx := lower_buf.find(tag_lower, search_start)) != -1:
                if self._at_block_boundary(buf, idx):
                    if best_idx == -1 or idx < best_idx:
                        best_idx, best_len = idx, len(tag)
                    break  # first boundary hit for this tag is enough
                search_start = idx + 1
        return best_idx, best_len

    def _filter_and_accumulate(self, text: str) -> None:
        """Append a delta to the buffer, discarding think blocks.

        Partial tags at buffer boundaries are held in ``_think_buffer`` until
        enough characters arrive to decide.
        """
        buf = self._think_buffer + text
        self._think_buffer = ""

        while buf:
            # Case-insensitive: models emit <Think>, <THINKING>, …
            lower_buf = buf.lower()
            if self._in_think_block:
                best_idx, best_len = _Scrubber._find_first_tag(buf, self._CLOSE_THINK_TAGS)
                if best_len:
                    self._in_think_block = False
                    buf = buf[best_idx + best_len:]
                else:
                    # Hold a tail that could be a partial close tag; discard the rest.
                    max_tag = max(len(t) for t in self._CLOSE_THINK_TAGS)
                    self._think_buffer = buf[-max_tag:] if len(buf) > max_tag else buf
                    return
            elif self._pending_think_name:
                # Pending latch (#128294): hold everything after the mid-line open until a
                # completing close hides it as a pair, the cap gives up, or flush releases.
                close_tag = f"</{self._pending_think_name}>"
                joined = self._pending_think_buf + buf
                idx = joined.lower().find(close_tag)
                if idx != -1:
                    consumed_from_buf = max(0, idx + len(close_tag) - len(self._pending_think_buf))
                    self._pending_think_name = self._pending_think_text = self._pending_think_buf = ""
                    buf = buf[consumed_from_buf:]
                    continue
                self._pending_think_buf = joined
                buf = ""
                if len(self._pending_think_buf) > PENDING_RETAIN_CAP:
                    # Give up: pass through verbatim (never worse than the old leak-through).
                    self._append_accumulated(self._strip_orphan_close_tags(
                        self._pending_think_text + self._pending_think_buf))
                    self._pending_think_name = self._pending_think_text = self._pending_think_buf = ""
                return
            else:
                best_idx, best_len = self._earliest_open_tag(buf, lower_buf)
                if best_len:
                    self._append_accumulated(buf[:best_idx])
                    self._in_think_block = True
                    buf = buf[best_idx + best_len:]
                else:
                    # A complete open tag mid-line cannot latch the hard block (that gate IS
                    # the prose-mention guard); hold the tail in the pending latch (#128294)
                    # instead of streaming the reasoning out when the close splits.
                    mid_idx, mid_len = _Scrubber._find_first_tag(buf, self._OPEN_THINK_TAGS)
                    if mid_len:
                        self._append_accumulated(buf[:mid_idx])
                        self._pending_think_text = buf[mid_idx:mid_idx + mid_len]
                        self._pending_think_name = self._pending_think_text[1:-1].lower()
                        self._pending_think_buf = ""
                        buf = buf[mid_idx + mid_len:]
                        continue
                    # Hold back a partial open tag at the tail.
                    held_back = _Scrubber._max_partial_suffix(buf, self._OPEN_THINK_TAGS)
                    if held_back:
                        self._append_accumulated(buf[:-held_back])
                        self._think_buffer = buf[-held_back:]
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
        if self._pending_think_name:
            # The pending latch (#128294) never saw its close: release the tag literal and
            # the held text verbatim -- the mention it may have been.
            self._append_accumulated(self._strip_orphan_close_tags(
                self._pending_think_text + self._pending_think_buf + self._think_buffer))
            self._pending_think_name = self._pending_think_text = self._pending_think_buf = ""
            self._think_buffer = ""
            return
        if self._think_buffer and not self._in_think_block:
            self._append_accumulated(self._strip_orphan_close_tags(self._think_buffer))
            self._think_buffer = ""

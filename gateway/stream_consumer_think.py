"""Think-block filtering for GatewayStreamConsumer.

Some models emit inline <think>...</think> blocks in content.  The agent strips
them from the final response, but intermediate edits go out before that, so this
mirrors the CLI's _stream_delta state machine; tag primitives are shared with
``agent/think_scrubber.py`` so the progressive display matches the post-stream scrubber."""

from __future__ import annotations

import logging

from agent.think_scrubber import StreamingThinkScrubber as _Scrubber

logger = logging.getLogger("gateway.stream_consumer")


class StreamThinkFilterMixin:
    """Progressive <think>-tag suppression over streamed deltas."""

    def _think_scrubber(self) -> _Scrubber:
        scrubber = getattr(self, "_streaming_think_scrubber", None)
        if scrubber is None:
            scrubber = _Scrubber()
            self._streaming_think_scrubber = scrubber
        return scrubber

    def _filter_and_accumulate(self, text: str) -> None:
        """Append a delta to the buffer, discarding think blocks."""
        visible = self._think_scrubber().feed(text)
        if visible:
            self._append_accumulated(visible)

    @staticmethod
    def _strip_orphan_close_tags(text: str) -> str:
        """Remove close tags (plus trailing whitespace) that have no matching open."""
        return _Scrubber._strip_orphan_close_tags(text)

    def _flush_think_buffer(self) -> None:
        """On stream end, flush text held back waiting for a possible open tag."""
        tail = self._think_scrubber().flush()
        if tail:
            self._append_accumulated(tail)

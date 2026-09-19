"""Shared per-chat send/edit budget + word-boundary fallback continuation (#116312).

Part 1 — pacing: Telegram counts ``editMessageText`` against the same per-chat
allowance as ``sendMessage``, but streaming previews (``DEFAULT_STREAMING_EDIT_INTERVAL``)
and sends drew on independent throttles.  Both now draw on one shared egress
budget: a send waits for its slot, an interim edit is skipped, the final edit
is never gated.

Part 2 — fallback cut: ``_continuation_text`` sliced at the last visible prefix
even mid-word.  The cut now backs up to the last space/newline (re-sending the
broken word's head); with no boundary the original cut stands.
"""

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig, _Tick


def _consumer(**kwargs):
    adapter = MagicMock()
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    consumer = GatewayStreamConsumer(adapter=adapter, chat_id="chat_1", **kwargs)
    return consumer, adapter


def _interim_tick():
    tick = _Tick()
    assert tick.is_interim
    return tick


class TestSharedEgressBudget:
    """Sends and interim edits draw on one per-chat budget."""

    @pytest.mark.asyncio
    async def test_interim_edit_skipped_while_send_budget_spent(self):
        """A successful send consumes the budget: the next interim tick skips its
        edit instead of firing into the same per-chat allowance."""
        consumer, _ = _consumer()
        await consumer._first_send("hello", finalize=False)
        consumer._accumulated = "hello world"  # under buffer_threshold (24)

        assert consumer._should_edit(_interim_tick()) is False

    @pytest.mark.asyncio
    async def test_first_send_waits_for_budget_slot(self):
        """A send immediately after an egress waits for the slot instead of
        firing back-to-back into the same chat."""
        consumer, adapter = _consumer(
            config=StreamConsumerConfig(edit_interval=0.05))
        consumer._last_edit_time = time.monotonic()  # pretend an edit just landed

        started = time.monotonic()
        assert await consumer._first_send("hello again", finalize=False) is True
        waited = time.monotonic() - started

        adapter.send.assert_awaited_once()
        assert waited >= 0.04

    @pytest.mark.asyncio
    async def test_send_consumes_budget_for_next_edit(self):
        """After a send lands, the shared clock moved: an interim edit that would
        otherwise fire on elapsed time is skipped."""
        consumer, _ = _consumer(
            config=StreamConsumerConfig(edit_interval=0.05))
        consumer._last_edit_time = 0.0  # an edit would fire on elapsed time alone
        consumer._accumulated = "hi"

        assert consumer._should_edit(_interim_tick()) is True
        await consumer._first_send("hi", finalize=False)
        assert consumer._should_edit(_interim_tick()) is False

    def test_final_edit_never_gated_by_budget(self):
        """The answer itself is never withheld: non-interim ticks always edit,
        even with a freshly spent budget."""
        consumer, _ = _consumer()
        consumer._last_edit_time = time.monotonic()  # budget just spent
        consumer._accumulated = "final answer"

        tick = _Tick(got_done=True)
        assert not tick.is_interim
        assert consumer._should_edit(tick) is True


class TestContinuationWordBoundary:
    """The fallback continuation must not resume mid-word."""

    def _consumer_with_visible(self, visible):
        consumer, _ = _consumer()
        consumer._fallback_prefix = ""
        consumer._last_sent_text = visible
        return consumer

    def test_continuation_backs_up_to_word_start(self):
        """Prefix ends inside a word: re-send the broken word's head."""
        consumer = self._consumer_with_visible("fixing the strea")
        continuation = consumer._continuation_text("fixing the streaming edits")
        assert continuation == "streaming edits"

    def test_continuation_without_boundary_keeps_original_cut(self):
        """One very long token: no boundary to back up to, original cut stands."""
        consumer = self._consumer_with_visible("abcdefghij")
        assert consumer._continuation_text("abcdefghijklmnop") == "klmnop"

    def test_continuation_at_word_boundary_does_not_resend(self):
        """Prefix ends exactly at a word end: no duplication."""
        consumer = self._consumer_with_visible("fixing the")
        assert consumer._continuation_text("fixing the streams") == "streams"

    def test_continuation_backs_up_across_newline(self):
        """Newline counts as a boundary too."""
        consumer = self._consumer_with_visible("line one\nline t")
        assert consumer._continuation_text("line one\nline two here") == "two here"

    def test_continuation_fully_seen_yields_empty(self):
        """Prefix is the whole final (e.g. only a cursor differs): nothing unseen,
        so no word is re-sent — the caller takes the cursor-strip path."""
        consumer = self._consumer_with_visible("Hello world")
        assert consumer._continuation_text("Hello world") == ""

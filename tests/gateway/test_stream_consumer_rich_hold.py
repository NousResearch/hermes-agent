"""Tests for streaming consistency fixes in GatewayStreamConsumer.

Covers three related fixes:

1. ``finalize=True`` in ``_send_fallback_final`` edit-in-place — the stale
   partial message is replaced with the complete answer using the finalize
   path so rich formatting is applied, not raw markdown.

2. Buffer-threshold 1-second floor — the buffer_threshold early-fire branch
   now requires ``elapsed >= min(edit_interval, 1.0)`` so fast LLMs cannot
   rapid-fire edits faster than platform rate limits.

3. Rich-hold mechanism — the consumer can suppress all streaming edits until
   ``got_done`` when the adapter signals it via
   ``should_hold_streaming_for_rich``.  Only activates before the first
   message is sent; resets on segment break.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ── shared adapter factory ────────────────────────────────────────────────


def _make_adapter(*, hold_fn=None):
    """Build a minimal BasePlatformAdapter subclass for stream-consumer tests.

    Uses the runtime-subclass pattern so ``isinstance(adapter,
    BasePlatformAdapter)`` returns True and the consumer's capability gates
    fire correctly, without pulling in any heavy platform state.

    ``hold_fn``, when provided, is installed as
    ``should_hold_streaming_for_rich`` on the adapter.
    """
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    _Adapter = type("_TestAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4096})
    _Adapter.__abstractmethods__ = frozenset()
    adapter = _Adapter.__new__(_Adapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None

    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="msg_1"),
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="msg_1"),
    )
    if hold_fn is not None:
        adapter.should_hold_streaming_for_rich = hold_fn
    return adapter


# ── 1. finalize=True in _send_fallback_final edit-in-place ───────────────


class TestFallbackFinalFinalize:
    """_send_fallback_final must use finalize=True when editing in-place."""

    @pytest.mark.asyncio
    async def test_edit_in_place_uses_finalize_true(self):
        """When a stale partial exists and the text fits, the edit call
        receives finalize=True so the adapter applies rich formatting."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(adapter, "chat1")

        # Simulate state after flood-control fallback was entered
        consumer._message_id = "msg_stale"
        consumer._last_sent_text = "partial text ▉"
        consumer._fallback_final_send = True
        consumer._already_sent = True

        final_text = "complete answer"
        await consumer._send_fallback_final(final_text)

        assert adapter.edit_message.called
        call_kwargs = adapter.edit_message.call_args.kwargs
        assert call_kwargs.get("finalize") is True, (
            "_send_fallback_final must pass finalize=True to preserve rich formatting"
        )

    @pytest.mark.asyncio
    async def test_edit_in_place_delivers_full_text(self):
        """The edit-in-place call receives the complete final text,
        not just the unseen tail."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(adapter, "chat1")

        consumer._message_id = "msg_stale"
        consumer._last_sent_text = "hello"
        consumer._fallback_final_send = True
        consumer._already_sent = True

        final_text = "hello world — the complete answer"
        await consumer._send_fallback_final(final_text)

        call_kwargs = adapter.edit_message.call_args.kwargs
        assert call_kwargs.get("content") == final_text

    @pytest.mark.asyncio
    async def test_edit_failure_falls_through_to_send(self):
        """If the edit-in-place fails, _send_fallback_final falls through and
        sends a new message instead of silently dropping content."""
        adapter = _make_adapter()
        adapter.edit_message = AsyncMock(
            return_value=SimpleNamespace(success=False, error="flood"),
        )
        consumer = GatewayStreamConsumer(adapter, "chat1")

        consumer._message_id = "msg_stale"
        consumer._last_sent_text = "partial"
        consumer._fallback_final_send = True
        consumer._already_sent = True
        consumer._fallback_prefix = "partial"

        await consumer._send_fallback_final("partial\ncomplete answer")

        # send() should have been called for the tail
        assert adapter.send.called


# ── 2. Buffer-threshold 1-second floor ───────────────────────────────────


class TestBufferThresholdFloor:
    """buffer_threshold early-fire must not trigger before 1 second."""

    @pytest.mark.asyncio
    async def test_buffer_waits_1s_before_firing(self):
        """Even with buffer_threshold=5 and a full buffer, no edit is sent
        until at least 1 second has elapsed (with a long edit_interval that
        would not independently fire in time)."""
        adapter = _make_adapter()
        # edit_interval long enough that it won't fire by itself in <1.2s
        cfg = StreamConsumerConfig(
            buffer_threshold=5,
            edit_interval=10.0,
            cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "chat1", cfg)

        task = asyncio.create_task(consumer.run())

        # Flood the buffer well above threshold immediately
        consumer.on_delta("A" * 200)

        # After 0.6 s the buffer is full but the 1-second floor has not expired
        await asyncio.sleep(0.6)
        assert not adapter.send.called, (
            "Edit must not fire before 1 second elapsed (buffer_threshold floor)"
        )

        # After another 0.6 s (total ~1.2 s) the floor has expired
        await asyncio.sleep(0.6)
        assert adapter.send.called, (
            "Edit should fire once 1 second has elapsed with a full buffer"
        )

        consumer.finish()
        await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_interval_still_fires_independently(self):
        """The regular edit_interval fires independently of buffer_threshold,
        even when the buffer is below the threshold."""
        adapter = _make_adapter()
        cfg = StreamConsumerConfig(
            buffer_threshold=5000,  # very high — threshold won't fire
            edit_interval=0.3,
            cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "chat1", cfg)

        task = asyncio.create_task(consumer.run())
        consumer.on_delta("hello")

        await asyncio.sleep(0.5)  # enough for the 0.3s interval to fire
        assert adapter.send.called, "Interval-based edit should still fire"

        consumer.finish()
        await asyncio.wait_for(task, timeout=2.0)


# ── 3. Rich-hold mechanism ────────────────────────────────────────────────


class TestRichHold:
    """should_hold_streaming_for_rich suppresses streaming before first send."""

    @pytest.mark.asyncio
    async def test_hold_suppresses_streaming_before_first_send(self):
        """When the adapter signals hold, no streaming edits are sent and
        the full answer is delivered as a single first send at got_done."""
        def hold_always(content: str) -> bool:
            return True

        adapter = _make_adapter(hold_fn=hold_always)
        cfg = StreamConsumerConfig(
            buffer_threshold=5,
            edit_interval=0.2,
            cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "chat1", cfg)

        task = asyncio.create_task(consumer.run())
        consumer.on_delta("A" * 50)
        await asyncio.sleep(0.5)  # interval would normally have fired

        # No message should have been sent — hold suppressed streaming
        assert not adapter.send.called, "Hold should have suppressed streaming"

        consumer.on_delta(" final part")
        consumer.finish()
        await asyncio.wait_for(task, timeout=2.0)

        # At got_done the complete answer is delivered as the first send
        assert adapter.send.called
        sent_content = adapter.send.call_args.kwargs.get("content") or \
                       adapter.send.call_args.args[0] if adapter.send.call_args.args else ""
        assert "final part" in sent_content

    @pytest.mark.asyncio
    async def test_hold_does_not_fire_after_first_send(self):
        """Once a message is on screen (_message_id set), the hold check
        is skipped — edits continue even for content that would trigger hold."""
        first_send_done = False

        async def patched_send(*, chat_id, content, reply_to=None, metadata=None):
            nonlocal first_send_done
            first_send_done = True
            return SimpleNamespace(success=True, message_id="msg_1")

        def hold_for_table(content: str) -> bool:
            # Hold would normally activate when a table is present
            return "|---|" in content

        adapter = _make_adapter(hold_fn=hold_for_table)
        adapter.send = patched_send
        cfg = StreamConsumerConfig(
            buffer_threshold=5,
            edit_interval=0.2,
            cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "chat1", cfg)

        task = asyncio.create_task(consumer.run())

        # First delta — no table yet, hold won't fire, first send happens
        consumer.on_delta("Plain text with no table")
        await asyncio.sleep(0.4)
        assert first_send_done, "First send should have happened before table"

        # Now send content that would trigger hold (but message_id is set)
        consumer.on_delta("\n| col | val |\n|---|---|")
        await asyncio.sleep(0.4)

        # edit_message should have been called (hold didn't suppress post-first-send)
        assert adapter.edit_message.called, (
            "Edits should continue after first send regardless of hold signal"
        )

        consumer.finish()
        await asyncio.wait_for(task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_hold_resets_on_segment_break(self):
        """After a segment break (tool boundary), the hold is cleared so
        the next segment can stream normally."""
        call_count = [0]

        def hold_once(content: str) -> bool:
            # Return True only on the first call
            call_count[0] += 1
            return call_count[0] == 1

        adapter = _make_adapter(hold_fn=hold_once)
        cfg = StreamConsumerConfig(
            buffer_threshold=5,
            edit_interval=0.2,
            cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "chat1", cfg)

        task = asyncio.create_task(consumer.run())

        # First segment — hold fires
        consumer.on_delta("A" * 50)
        await asyncio.sleep(0.4)
        assert not adapter.send.called, "Hold should suppress first segment"

        # Segment break — resets hold
        consumer.on_segment_break()
        await asyncio.sleep(0.05)

        # Second segment — hold_once returns False now (count > 1)
        consumer.on_delta("B" * 50)
        await asyncio.sleep(0.4)
        assert adapter.send.called, "Second segment should stream after hold reset"

        consumer.finish()
        await asyncio.wait_for(task, timeout=2.0)

    def test_rich_hold_initialised_false(self):
        """_rich_hold starts False so the hold path is opt-in."""
        adapter = _make_adapter()
        consumer = GatewayStreamConsumer(adapter, "chat1")
        assert consumer._rich_hold is False

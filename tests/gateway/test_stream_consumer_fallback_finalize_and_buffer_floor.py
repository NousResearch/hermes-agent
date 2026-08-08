"""Tests for streaming consistency fixes in GatewayStreamConsumer.

Covers two related fixes:

1. ``finalize=True`` in ``_send_fallback_final`` edit-in-place — the stale
   partial message is replaced with the complete answer using the finalize
   path so rich formatting is applied, not raw markdown.

2. Buffer-threshold 1-second floor — the buffer_threshold early-fire branch
   requires ``elapsed >= 1.0`` (a fixed floor, independent of
   ``edit_interval``) so fast LLMs cannot rapid-fire edits faster than
   platform rate limits, while a long configured ``edit_interval`` can still
   flush early at the 1-second mark instead of waiting the full interval.

A third mechanism — an adapter-side ``should_hold_streaming_for_rich`` hook
letting the consumer suppress all streaming until ``got_done`` — was removed
(hermes-sweeper review): no adapter enabled it (Telegram always returned
``False``, by design — see its former docstring), so it was dead extension
surface with dedicated tests but no live production behavior. Reintroduce it
if a concrete adapter actually needs it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ── shared adapter factory ────────────────────────────────────────────────


def _make_adapter():
    """Build a minimal BasePlatformAdapter subclass for stream-consumer tests.

    Uses the runtime-subclass pattern so ``isinstance(adapter,
    BasePlatformAdapter)`` returns True and the consumer's capability gates
    fire correctly, without pulling in any heavy platform state.
    """
    from gateway.platforms.base import BasePlatformAdapter

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
        """After an initial flush establishes a real _last_edit_time, a
        subsequent buffer_threshold-exceeding burst must still wait a full
        second before firing again — even with a long edit_interval that
        would not independently fire in time.

        The very first ever flush is deliberately excluded from this check:
        ``_last_edit_time`` starts at 0.0, so "elapsed since last edit" is
        enormous on the first-ever tick and that flush fires immediately
        regardless of the floor — correct UX (show something as soon as
        possible), but not something this test is about."""
        adapter = _make_adapter()
        # edit_interval long enough that it won't fire by itself in <1.2s
        cfg = StreamConsumerConfig(
            buffer_threshold=5,
            edit_interval=10.0,
            cursor="",
        )
        consumer = GatewayStreamConsumer(adapter, "chat1", cfg)

        task = asyncio.create_task(consumer.run())

        consumer.on_delta("initial")
        await asyncio.sleep(0.1)
        assert adapter.send.called, "Initial content should flush immediately"
        adapter.send.reset_mock()

        # Now _last_edit_time reflects a real, recent edit. Flood the buffer
        # well above threshold and confirm the 1-second floor applies to
        # this (realistic) second flush.
        consumer.on_delta("A" * 200)

        # After 0.6 s the buffer is full but the 1-second floor has not expired
        await asyncio.sleep(0.6)
        assert not adapter.edit_message.called, (
            "Edit must not fire before 1 second elapsed (buffer_threshold floor)"
        )

        # After another 0.6 s (total ~1.2 s) the floor has expired
        await asyncio.sleep(0.6)
        assert adapter.edit_message.called, (
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

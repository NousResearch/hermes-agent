"""Tests for native streaming in GatewayStreamConsumer (WeCom-style transport).

Native streaming is the consumer's transport for adapters that:
  * cannot edit messages (``SUPPORTS_MESSAGE_EDITING = False``); but
  * expose a stream protocol where every frame is a cumulative content
    update plus a ``finish: true`` final frame (e.g. WeCom's
    ``msgtype: "stream"`` via ``aibot_respond_msg``).

These tests use a runtime subclass of ``BasePlatformAdapter`` so the
consumer's ``isinstance(BasePlatformAdapter)`` gate is satisfied. They
verify the full lifecycle (seed → mid-stream updates → finalize), the
throttling that keeps frames under WeCom's 30/min rate ceiling, and the
fallback path when ``send_stream_frame`` returns False.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
    StreamConsumerConfig,
)


def _make_native_streaming_adapter(
    *,
    supports_native: bool = True,
    seed_succeeds: bool = True,
    frames_succeed: bool = True,
    finalize_succeeds: bool = True,
):
    """Build a BasePlatformAdapter subclass that supports native streaming.

    Records every ``send_stream_frame`` call on ``adapter.frames`` for assertions.
    """
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    NativeStreamingAdapter = type(
        "NativeStreamingAdapter",
        (BasePlatformAdapter,),
        {
            "MAX_MESSAGE_LENGTH": 4096,
            "SUPPORTS_MESSAGE_EDITING": False,
            "SUPPORTS_NATIVE_STREAMING": True,
        },
    )
    NativeStreamingAdapter.__abstractmethods__ = frozenset()
    adapter = NativeStreamingAdapter.__new__(NativeStreamingAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None

    adapter.frames = []  # list of (text, finalize)

    def _supports(chat_type=None, metadata=None):
        return bool(supports_native)
    adapter.supports_native_streaming = _supports

    async def _send_stream_frame(
        text, *, finalize=False, chat_id=None, reply_to=None,
    ):
        adapter.frames.append({
            "text": text,
            "finalize": finalize,
            "chat_id": chat_id,
            "reply_to": reply_to,
        })
        if finalize:
            return finalize_succeeds
        # First frame is the seed (empty content).
        if text == "" and len(adapter.frames) == 1:
            return seed_succeeds
        return frames_succeed
    adapter.send_stream_frame = _send_stream_frame

    # send / edit_message: count fallback usage so we can assert native
    # ran without ever touching them.
    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="fallback_msg"),
    )
    adapter.edit_message = AsyncMock(
        return_value=SimpleNamespace(success=True),
    )
    return adapter


# === RESOLVER ===


class TestNativeStreamingResolver:
    """``_resolve_native_streaming`` gating logic."""

    def test_capable_adapter_resolves_to_native(self):
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(chat_type="dm", cursor="")
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)
        assert consumer._resolve_native_streaming() is True

    def test_class_attribute_required(self):
        """Adapter without SUPPORTS_NATIVE_STREAMING class attr returns False."""
        from gateway.platforms.base import BasePlatformAdapter

        Bare = type("Bare", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 4096})
        Bare.__abstractmethods__ = frozenset()
        adapter = Bare.__new__(Bare)
        adapter._typing_paused = set()
        adapter._fatal_error_message = None
        adapter.supports_native_streaming = lambda chat_type=None, metadata=None: True

        cfg = StreamConsumerConfig(chat_type="dm")
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)
        assert consumer._resolve_native_streaming() is False

    def test_probe_returning_false_disables_native(self):
        adapter = _make_native_streaming_adapter(supports_native=False)
        cfg = StreamConsumerConfig(chat_type="dm")
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)
        assert consumer._resolve_native_streaming() is False

    def test_magicmock_adapter_falls_back(self):
        """MagicMock adapters are excluded by isinstance gate."""
        adapter = MagicMock()
        cfg = StreamConsumerConfig(chat_type="dm")
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)
        assert consumer._resolve_native_streaming() is False


# === LIFECYCLE ===


class TestNativeStreamingLifecycle:
    """Seed frame on run-start → mid-stream updates → finalize."""

    @pytest.mark.asyncio
    async def test_seed_frame_fires_at_run_start(self):
        """The first thing the consumer does is a seed frame for typing UI."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = asyncio.create_task(consumer.run())
        # Tiny sleep so run() can dispatch the seed before we tear down.
        await asyncio.sleep(0.02)
        consumer.finish()
        await task

        assert len(adapter.frames) >= 1
        assert adapter.frames[0]["text"] == ""
        assert adapter.frames[0]["finalize"] is False
        assert adapter.frames[0]["chat_id"] == "chat-1"

    @pytest.mark.asyncio
    async def test_full_run_routes_only_through_send_stream_frame(self):
        """No mid-stream call to send() / edit_message() in native mode."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        # Push enough text past the throttling threshold (>20 visible chars).
        consumer.on_delta("This is a substantial first chunk past the threshold.")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.on_delta(" Even more content arriving in the second chunk.")
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        assert adapter.send.await_count == 0
        assert adapter.edit_message.await_count == 0
        # Final frame must be finalize=true.
        finalize_frames = [f for f in adapter.frames if f["finalize"]]
        assert len(finalize_frames) == 1
        # Final text held the full accumulated content.
        assert "first chunk" in finalize_frames[0]["text"]
        assert "second chunk" in finalize_frames[0]["text"]

    @pytest.mark.asyncio
    async def test_consumer_marks_final_response_sent(self):
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        consumer.on_delta("Hello, this is a sufficiently long response.")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        assert consumer.final_response_sent is True
        assert consumer.final_content_delivered is True


# === THROTTLING ===


class TestNativeStreamingThrottling:
    """Mid-stream frames must not flood when text grows by tiny amounts."""

    @pytest.mark.asyncio
    async def test_tiny_increments_are_throttled(self):
        """20-char min between non-finalize frames keeps wecom under 30/min."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=1,  # aggressive flush
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.02)  # let seed frame fire
        # Many 1-char deltas — should NOT translate to 1 frame each.
        for ch in "abcdefghij":  # 10 chars total
            consumer.on_delta(ch)
            await asyncio.sleep(0.015)
        consumer.finish()
        await task

        # Seed frame + finalize at minimum. Throttling keeps mid-stream
        # frames roughly proportional to total chars / 20, not to delta count.
        non_finalize_content_frames = [
            f for f in adapter.frames if not f["finalize"] and f["text"]
        ]
        # 10 chars / 20 char min → at most 0 mid-stream content frames.
        # (The finalize frame carries the full text.)
        assert len(non_finalize_content_frames) <= 1, (
            f"throttling failed: got {len(non_finalize_content_frames)} mid frames "
            f"for 10 chars total"
        )
        # But the user still sees the full content in the finalize frame.
        finalize_frames = [f for f in adapter.frames if f["finalize"]]
        assert len(finalize_frames) == 1
        assert finalize_frames[0]["text"] == "abcdefghij"

    @pytest.mark.asyncio
    async def test_large_growth_emits_mid_frames(self):
        """When text grows by >20 chars, an interim frame should land."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.02)
        # First chunk well past 20 chars.
        consumer.on_delta("A" * 40)
        await asyncio.sleep(0.05)
        # Second chunk also past 20 chars.
        consumer.on_delta("B" * 40)
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        non_finalize_content_frames = [
            f for f in adapter.frames if not f["finalize"] and f["text"]
        ]
        assert len(non_finalize_content_frames) >= 1


# === FALLBACK ===


class TestNativeStreamingFallback:
    """When ``send_stream_frame`` returns False, native is disabled and the
    consumer takes the regular send/edit path."""

    @pytest.mark.asyncio
    async def test_seed_failure_disables_native(self):
        """If even the seed frame fails, native is off for the run."""
        adapter = _make_native_streaming_adapter(seed_succeeds=False)
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        consumer.on_delta("hello world this is enough text")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        assert consumer._use_native_streaming is False

    @pytest.mark.asyncio
    async def test_native_streaming_disables_draft(self):
        """Adapter that supports both — native takes priority, draft off."""
        adapter = _make_native_streaming_adapter()
        # Pretend it also offers draft (won't be used).
        adapter.supports_draft_streaming = lambda chat_type=None, metadata=None: True
        adapter.send_draft = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id=None),
        )

        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        consumer.on_delta("a sufficiently long content chunk here yo")
        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.05)
        consumer.finish()
        await task

        assert consumer._use_native_streaming is True
        assert consumer._use_draft_streaming is False
        adapter.send_draft.assert_not_awaited()

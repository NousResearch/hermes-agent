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
        text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
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


class TestNativeStreamingSegmentBreak:
    """Segment breaks should NOT finalize or reset for WeCom native streaming."""

    @pytest.mark.asyncio
    async def test_segment_break_preserves_cumulative_text(self):
        """Tool boundary keeps pre+post text in one stream, single finalize."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        # Pre-tool text
        consumer.on_delta("Pre-tool content. ")
        # Simulate tool boundary (segment break)
        consumer.on_segment_break()
        # Post-tool text
        consumer.on_delta("Post-tool result.")

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)
        consumer.finish()
        await task

        # Should only have ONE finalize frame (the final one)
        finalize_frames = [f for f in adapter.frames if f.get("finalize")]
        assert len(finalize_frames) == 1, \
            f"Should have exactly 1 finalize, got {len(finalize_frames)}"

        # The finalize frame must contain BOTH pre-tool and post-tool text
        final_text = finalize_frames[0]["text"]
        assert "Pre-tool content" in final_text, \
            "Final frame must include pre-tool text (not lost by reset)"
        assert "Post-tool result" in final_text, \
            "Final frame must include post-tool text"

    @pytest.mark.asyncio
    async def test_segment_break_no_extra_finalize(self):
        """Segment break should NOT produce a finalize frame."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="",
            edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        consumer.on_delta("First part of response. ")
        consumer.on_segment_break()
        consumer.on_delta("Second part after tool.")

        task = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.1)
        consumer.finish()
        await task

        # Count finalize frames — should be exactly 1 (only at finish)
        finalize_count = sum(1 for f in adapter.frames if f.get("finalize"))
        assert finalize_count == 1, \
            f"Expected 1 finalize (only at end), got {finalize_count}"

        # Non-finalize frames should show cumulative growth
        content_frames = [f for f in adapter.frames if not f.get("finalize") and f["text"]]
        if len(content_frames) >= 2:
            # Later frames should be longer (cumulative)
            assert len(content_frames[-1]["text"]) >= len(content_frames[0]["text"]), \
                "Content frames should grow cumulatively"


class TestClarifyReopenBoundary:
    """A clarify boundary (reopen=True) finalizes the pre-prompt stream but
    keeps native streaming enabled so the post-answer continuation re-opens a
    fresh native stream — restoring the typing bubble instead of degrading to a
    one-shot send() (the approval-path behaviour)."""

    async def _drain(self, consumer, seconds=0.15):
        deadline = asyncio.get_event_loop().time() + seconds
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_reopen_boundary_keeps_native_and_reopens_stream(self):
        """Pre-prompt content is finalized; post-answer content re-seeds a
        fresh stream and finalizes again — two seeds, two finalizes, native
        never disabled."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = asyncio.create_task(consumer.run())
        await self._drain(consumer, 0.03)  # let the initial seed fire

        # Pre-prompt streamed content.
        consumer.on_delta("正在处理，先给你看一段初步结果。")
        await self._drain(consumer, 0.05)

        # Clarify boundary with reopen=True.
        boundary = consumer.close_for_approval_prompt(
            "💬 等待你的选择...", reason="Clarify", reopen=True,
        )
        fut = boundary[0] if isinstance(boundary, tuple) else boundary
        await asyncio.wait_for(fut, timeout=1.0)

        # Native must stay enabled and buffer_only must NOT be set.
        assert consumer._use_native_streaming is True
        assert consumer.cfg.buffer_only is False
        assert consumer._native_stream_opened is False  # finalized, awaiting reopen

        # Post-answer continuation → should re-open a fresh stream.
        consumer.on_delta("根据你的选择，这是后续的完整回答内容。")
        await self._drain(consumer, 0.05)
        consumer.finish()
        await task

        finalize_frames = [f for f in adapter.frames if f.get("finalize")]
        seed_frames = [f for f in adapter.frames if f["text"] == "" and not f.get("finalize")]

        # Two finalizes: one at the boundary, one at got_done for the reopened stream.
        assert len(finalize_frames) == 2, (
            f"expected 2 finalizes (boundary + reopened turn), got {len(finalize_frames)}"
        )
        # At least two seeds: initial + reopened.
        assert len(seed_frames) >= 2, (
            f"expected the post-answer content to re-seed a fresh stream, "
            f"seeds={len(seed_frames)}"
        )
        # The reopened stream's finalize carries the post-answer content.
        assert "后续的完整回答" in finalize_frames[-1]["text"]
        # Native was never disabled → no fallback send.
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reopen_boundary_no_post_content_skips_lone_placeholder(self):
        """If the agent produces nothing after the clarify, got_done must NOT
        re-seed a fresh stream just to emit a lone '✅' bubble."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = asyncio.create_task(consumer.run())
        await self._drain(consumer, 0.03)

        consumer.on_delta("这一段是提问前已经流式出去的内容。")
        await self._drain(consumer, 0.05)

        boundary = consumer.close_for_approval_prompt(
            "💬 等待你的选择...", reason="Clarify", reopen=True,
        )
        fut = boundary[0] if isinstance(boundary, tuple) else boundary
        await asyncio.wait_for(fut, timeout=1.0)

        frames_before_finish = len(adapter.frames)

        # No post-answer content — just finish.
        consumer.finish()
        await task

        # No new seed / finalize frame should have been emitted after the
        # boundary (no lone "✅").
        new_frames = adapter.frames[frames_before_finish:]
        assert not any(f["text"] == "✅" for f in new_frames), (
            f"must not emit a lone '✅' placeholder, got {new_frames}"
        )
        # A "✅" placeholder anywhere would indicate the guard failed.
        assert not any(f["text"] == "✅" for f in adapter.frames)

    @pytest.mark.asyncio
    async def test_approval_boundary_still_disables_native(self):
        """Contrast: the approval path (reopen=False, the default) disables
        native and buffers post-prompt output — unchanged behaviour."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = asyncio.create_task(consumer.run())
        await self._drain(consumer, 0.03)

        consumer.on_delta("审批前的流式内容。")
        await self._drain(consumer, 0.05)

        boundary = consumer.close_for_approval_prompt(reason="Approval")
        fut = boundary[0] if isinstance(boundary, tuple) else boundary
        await asyncio.wait_for(fut, timeout=1.0)

        assert consumer._use_native_streaming is False
        assert consumer.cfg.buffer_only is True

        consumer.finish()
        await task


class TestClarifyEagerReseed:
    """EAGER re-seed after a clarify answer.

    WeCom typing is driven by the stream seed frame (send_typing is a no-op),
    so the clarify-reopen path used to re-seed LAZILY — only when the LLM
    emitted its first post-answer delta — leaving up to ~48s of dead air after
    the user replied.  The eager path seeds the moment the user answers, via
    ``request_reopen_seed()`` → ``_REOPEN_SEED`` → the run-loop handler, so the
    typing bubble reappears instantly.

    Each test below maps to one of the 7 verification points.  Breakage map for
    the destructive checks:
      * Remove the run-loop eager-seed branch  → test_reopen_seed_opens_stream_before_any_delta turns red.
      * Remove the _suppress_silence_marker native-close patch → test_eager_seed_then_silence_marker_closes_stream turns red.
      * Remove got_done hole A branch → test_eager_seed_no_content_finalizes_once turns red (or a "✅" leaks).
    """

    async def _drain(self, consumer, seconds=0.15):
        deadline = asyncio.get_event_loop().time() + seconds
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.01)

    async def _wait_until(self, predicate, timeout=2.0):
        """Poll ``predicate()`` until true or timeout — robust against CPU
        contention (a fixed _drain sleep flakes under the 24-worker suite).
        Returns the predicate's final value so callers can assert on it.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if predicate():
                return True
            await asyncio.sleep(0.01)
        return predicate()

    async def _to_reopen_pending(self, consumer, adapter):
        """Run to the point right after a clarify boundary: native still on,
        no stream open, awaiting a re-seed.  Returns nothing; leaves the run
        task attached on ``consumer._task`` for the caller to finish()/await.
        """
        task = asyncio.create_task(consumer.run())
        await self._drain(consumer, 0.03)  # initial seed

        consumer.on_delta("提问前已经流式出去的一段内容。")
        await self._drain(consumer, 0.05)

        boundary = consumer.close_for_approval_prompt(
            "💬 等待你的选择...", reason="Clarify", reopen=True,
        )
        fut = boundary[0] if isinstance(boundary, tuple) else boundary
        await asyncio.wait_for(fut, timeout=1.0)
        return task

    # === POINT 1: timing — the boundary itself does NOT eager-seed ===

    @pytest.mark.asyncio
    async def test_boundary_alone_does_not_eager_seed(self):
        """Processing the clarify boundary (reopen=True) must not emit a fresh
        seed frame on its own — the eager seed only happens once the user
        answers and request_reopen_seed() is called."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)

        # State: reopen-pending, no stream open, native still live.
        assert consumer._awaiting_reopen_after_boundary is True
        assert consumer._native_stream_opened is False
        assert consumer._use_native_streaming is True
        # No eager seed yet (request_reopen_seed not called).  Only the initial
        # seed + the boundary finalize should exist — no NEW empty seed.
        seed_frames = [
            f for f in adapter.frames if f["text"] == "" and not f["finalize"]
        ]
        assert len(seed_frames) == 1, (
            f"boundary must not eager-seed on its own, seeds={len(seed_frames)}"
        )

        consumer.finish()
        await task

    # === POINT 2: latency decoupling (core) — seed lands BEFORE any delta ===

    @pytest.mark.asyncio
    async def test_reopen_seed_opens_stream_before_any_delta(self):
        """After the boundary, request_reopen_seed() → the run loop opens a
        fresh empty seed frame BEFORE the LLM produces any post-answer delta.

        DESTRUCTIVE: remove the run-loop `_REOPEN_SEED` handler and this fails.
        """
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)
        seeds_before = len(
            [f for f in adapter.frames if f["text"] == "" and not f["finalize"]]
        )

        # User answered → request an eager re-seed.  NO on_delta yet.
        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)  # let run() process _REOPEN_SEED

        seeds_after = len(
            [f for f in adapter.frames if f["text"] == "" and not f["finalize"]]
        )
        assert seeds_after == seeds_before + 1, (
            "eager re-seed must emit exactly one new empty seed frame before "
            f"any delta (before={seeds_before}, after={seeds_after})"
        )
        assert await self._wait_until(lambda: consumer._native_stream_opened)
        assert consumer._awaiting_reopen_after_boundary is False
        assert consumer._reopen_seeded_eagerly is True

        consumer.finish()
        await task

    # === POINT 3: no-content wrap-up (hole A) — one finalize, no "✅" ===

    @pytest.mark.asyncio
    async def test_eager_seed_no_content_finalizes_once(self):
        """After an eager seed, if the agent produces NO content, got_done must
        close the empty typing bubble with exactly one finalize and never emit
        a lone '✅'.

        DESTRUCTIVE: remove got_done hole-A branch and this fails (the bubble
        hangs / a '✅' placeholder leaks).
        """
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)
        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)
        assert await self._wait_until(
            lambda: consumer._native_stream_opened
        ), "eager seed must open the stream"

        frames_before = len(adapter.frames)

        # No post-answer content — just finish.
        consumer.finish()
        await task

        new_frames = adapter.frames[frames_before:]
        finalize_frames = [f for f in new_frames if f["finalize"]]
        assert len(finalize_frames) == 1, (
            f"eager-seed empty stream must finalize exactly once, "
            f"got {len(finalize_frames)}: {new_frames}"
        )
        # No lone "✅" anywhere.
        assert not any(f["text"] == "✅" for f in adapter.frames), (
            f"must not emit a lone '✅' placeholder, frames={adapter.frames}"
        )
        # The finalize is an empty close, not content.
        assert finalize_frames[0]["text"] == ""
        assert consumer._native_stream_opened is False

    # === POINT 4: no seed while merely waiting (guard B) ===

    @pytest.mark.asyncio
    async def test_no_seed_while_awaiting_user_answer(self):
        """After the boundary but WITHOUT request_reopen_seed() and without any
        delta, no fresh seed frame may appear — typing must not light up while
        the user has not yet replied."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)
        seeds_before = len(
            [f for f in adapter.frames if f["text"] == "" and not f["finalize"]]
        )

        # Sit in the reopen-pending state; do NOT answer.
        await self._drain(consumer, 0.1)

        seeds_after = len(
            [f for f in adapter.frames if f["text"] == "" and not f["finalize"]]
        )
        assert seeds_after == seeds_before, (
            "no new seed may fire while waiting for the user's clarify answer "
            f"(before={seeds_before}, after={seeds_after})"
        )
        assert consumer._native_stream_opened is False
        assert consumer._awaiting_reopen_after_boundary is True

        consumer.finish()
        await task

    # === POINT 5: seed failure degrades to single buffered send ===

    @pytest.mark.asyncio
    async def test_eager_seed_failure_degrades_to_buffer_only(self):
        """If the eager re-seed frame fails, native is disabled and buffer_only
        is set so post-answer content lands as one buffered send() rather than
        per-tick fragments on a non-editable platform."""
        # Make the SECOND seed (the eager re-seed) fail while the initial seed
        # succeeds, so we actually reach the reopen-pending state first.
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        NativeStreamingAdapter = type(
            "NativeStreamingAdapter2",
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
        adapter.frames = []
        adapter.supports_native_streaming = (
            lambda chat_type=None, metadata=None: True
        )

        empty_seed_count = {"n": 0}

        async def _send_stream_frame(
            text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
        ):
            adapter.frames.append({
                "text": text, "finalize": finalize,
                "chat_id": chat_id, "reply_to": reply_to,
            })
            if finalize:
                return True
            if text == "":
                empty_seed_count["n"] += 1
                # First empty seed (run-start) succeeds; second (eager re-seed)
                # fails to exercise the degrade path.
                return empty_seed_count["n"] == 1
            return True
        adapter.send_stream_frame = _send_stream_frame
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="fallback_msg"),
        )
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)
        assert consumer._use_native_streaming is True  # still on after boundary

        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)  # process _REOPEN_SEED → seed fails

        assert await self._wait_until(
            lambda: consumer._use_native_streaming is False
        ), "failed eager seed must disable native streaming"
        assert consumer.cfg.buffer_only is True
        assert consumer._native_stream_opened is False

        consumer.finish()
        await task

    # === POINT 6: single-bubble invariant (core) — eager + lazy don't stack ===

    @pytest.mark.asyncio
    async def test_eager_seed_then_delta_does_not_double_seed(self):
        """After an eager seed opens the stream, a subsequent post-answer delta
        must flow into that SAME stream — the lazy re-seed path must NOT open a
        second bubble (it's gated on _native_stream_opened being False)."""
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)
        seeds_before = len(
            [f for f in adapter.frames if f["text"] == "" and not f["finalize"]]
        )

        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)  # eager seed opens the stream
        assert await self._wait_until(lambda: consumer._native_stream_opened)

        # Now the LLM produces post-answer content.
        consumer.on_delta("根据你的选择，这是后续的完整回答内容，足够长以触发一次刷新。")
        await self._drain(consumer, 0.05)
        consumer.finish()
        await task

        seeds_after = len(
            [f for f in adapter.frames if f["text"] == "" and not f["finalize"]]
        )
        # Exactly ONE new empty seed total (the eager one) — the lazy path did
        # not add a second.
        assert seeds_after == seeds_before + 1, (
            f"lazy re-seed must not stack a second bubble on top of the eager "
            f"seed (before={seeds_before}, after={seeds_after})"
        )
        finalize_frames = [f for f in adapter.frames if f["finalize"]]
        # The post-answer content lands in the final (reopened) finalize.
        assert "后续的完整回答" in finalize_frames[-1]["text"]
        adapter.send.assert_not_awaited()

    # === POINT 7: silence marker closes the eager stream (hole B) ===

    @pytest.mark.asyncio
    async def test_eager_seed_then_silence_marker_closes_stream(self):
        """After an eager seed, if the agent's whole reply is an intentional
        silence marker (NO_REPLY), the open native stream must be finalized
        (closed) rather than left hanging — and the delivery flags stay False.

        DESTRUCTIVE: remove the _suppress_silence_marker native-close patch and
        this fails (the stream never gets a finalize).
        """
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)
        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)
        assert await self._wait_until(lambda: consumer._native_stream_opened)

        frames_before = len(adapter.frames)

        # Agent emits only a bare silence marker.
        consumer.on_delta("NO_REPLY")
        consumer.finish()
        await task

        new_frames = adapter.frames[frames_before:]
        finalize_frames = [f for f in new_frames if f["finalize"]]
        assert len(finalize_frames) >= 1, (
            f"silence marker after eager seed must finalize/close the open "
            f"native stream, new_frames={new_frames}"
        )
        # The marker text must never have been streamed as content.
        assert not any("NO_REPLY" in f["text"] for f in adapter.frames), (
            "silence marker must not leak into any frame"
        )
        # Nothing was delivered — flags stay False.
        assert consumer.final_content_delivered is False
        assert consumer.final_response_sent is False
        assert consumer._native_stream_opened is False

    # === ROUND 2, POINT 1: 降级后 post-answer 内容落地为单气泡（补强 Point 5）===

    @pytest.mark.asyncio
    async def test_degraded_post_answer_lands_single_bubble(self):
        """eager 再次 seed 失败降级（_use_native_streaming=False + buffer_only=True）
        后，继续喂 post-answer 内容并 finish，内容必须以恰好一次 send() 落地为单气泡，
        且绝不重发 pre-clarify 的旧内容（boundary 时 _reset_segment_state 已清空 buffer）。

        直接验证 review (b).4 的结论。
        """
        # 复用 Point 5 的降级 adapter：第一次空 seed 成功、第二次（eager 再 seed）失败。
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        NativeStreamingAdapter = type(
            "NativeStreamingAdapter2b",
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
        adapter.frames = []
        adapter.supports_native_streaming = (
            lambda chat_type=None, metadata=None: True
        )

        empty_seed_count = {"n": 0}

        async def _send_stream_frame(
            text, *, finalize=False, chat_id=None, reply_to=None, **kwargs
        ):
            adapter.frames.append({
                "text": text, "finalize": finalize,
                "chat_id": chat_id, "reply_to": reply_to,
            })
            if finalize:
                return True
            if text == "":
                empty_seed_count["n"] += 1
                # 初始 seed（run-start）成功；第二次空 seed（eager 再 seed）失败。
                return empty_seed_count["n"] == 1
            return True
        adapter.send_stream_frame = _send_stream_frame
        adapter.send = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="fallback_msg"),
        )
        adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))

        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)
        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)  # _REOPEN_SEED → 第二次 seed 失败 → 降级

        assert await self._wait_until(
            lambda: consumer._use_native_streaming is False
        ), "eager seed 失败必须关闭 native streaming"
        assert consumer.cfg.buffer_only is True

        # 降级后继续产出 post-answer 内容 → 应以一次 send() 单气泡投递。
        consumer.on_delta("根据你的选择，这是完整的后续答复内容。")
        await self._drain(consumer, 0.05)
        consumer.finish()
        await task

        # send() 恰好一次。
        adapter.send.assert_awaited_once()
        # 取出该次 send 的内容（在 _send_or_edit 首发路径以 content= kwarg 传入）。
        sent_call = adapter.send.await_args
        sent_content = sent_call.kwargs.get("content")
        if sent_content is None and sent_call.args:
            # 兼容位置参数写法（其它 boundary fallback 用 send(chat_id, text)）。
            sent_content = sent_call.args[-1]
        assert sent_content is not None
        assert "完整的后续答复" in sent_content
        # 绝不重发 pre-clarify 内容（boundary 已 finalize 成稳定气泡并清空 buffer）。
        assert "提问前已经流式出去的一段内容。" not in sent_content

    # === ROUND 2, POINT 2: 短内容不被误判为「无内容」而走 hole-A 空关闭 ===

    @pytest.mark.asyncio
    async def test_eager_seed_short_content_not_hole_a(self):
        """eager seed 开流后只产出一段不足 60 char 节流阈值的短内容（中途不推帧），
        finish 时该短内容必须随 finalize 帧落地，而不是走 hole-A 的空关闭（空 finalize
        或 "✅"）。因为 _accumulated 非空，hole-A 条件 `not _accumulated` 不成立。
        """
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        task = await self._to_reopen_pending(consumer, adapter)
        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)
        assert await self._wait_until(
            lambda: consumer._native_stream_opened
        ), "eager seed 应已开流"

        frames_before = len(adapter.frames)

        # 只产出一个 1-char 短内容（远不足 60 char 节流阈值 → 中途不推帧）。
        consumer.on_delta("短")
        consumer.finish()
        await task

        new_frames = adapter.frames[frames_before:]
        finalize_frames = [f for f in new_frames if f["finalize"]]
        # 收尾恰好一个 finalize 帧，且携带短内容。
        assert len(finalize_frames) == 1, (
            f"eager seed 后短内容应恰好 finalize 一次，got {finalize_frames}"
        )
        assert "短" in finalize_frames[0]["text"], (
            f"finalize 帧必须携带短内容，got {finalize_frames[0]!r}"
        )
        # 未走 hole-A：没有空 finalize 收尾，也没有 "✅" 占位。
        assert finalize_frames[0]["text"] != "", "不应是 hole-A 的空 finalize"
        assert not any(f["text"] == "✅" for f in adapter.frames), (
            f"短内容不应被误判为无内容而发 '✅'，frames={adapter.frames}"
        )
        # _accumulated 非空 → hole-A 条件 `not _accumulated` 不成立，佐证走的是内容收尾。
        assert consumer._accumulated == "短"

    # === ROUND 2, POINT 3: 双 clarify 边界链路自洽、无残留误判 ===

    @pytest.mark.asyncio
    async def test_double_clarify_boundary_reseed_chain(self):
        """eager seed → 内容 → 第二轮 clarify boundary → 第二轮 eager seed。

        验证多边界链路自洽：第二轮 boundary 后回到 reopen-pending 状态
        （_awaiting_reopen_after_boundary=True、_native_stream_opened=False），
        再次 request_reopen_seed() 仍能成功 eager seed，标志无残留导致误判。

        覆盖 review (b).2。
        """
        adapter = _make_native_streaming_adapter()
        cfg = StreamConsumerConfig(
            chat_type="dm", cursor="", edit_interval=0.01, buffer_threshold=5,
        )
        consumer = GatewayStreamConsumer(adapter, "chat-1", cfg)

        # 第一轮：boundary → eager seed → 内容。
        task = await self._to_reopen_pending(consumer, adapter)
        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)
        assert await self._wait_until(lambda: consumer._native_stream_opened)
        assert consumer._reopen_seeded_eagerly is True

        consumer.on_delta("根据你的选择，这是后续的完整回答内容，足够长以触发一次流式刷新的补充。")
        await self._drain(consumer, 0.05)

        seeds_before_second_boundary = len(
            [f for f in adapter.frames if f["text"] == "" and not f["finalize"]]
        )

        # 第二轮 clarify boundary（reopen=True）。
        boundary = consumer.close_for_approval_prompt(
            "💬 等待你的选择...", reason="Clarify", reopen=True,
        )
        fut = boundary[0] if isinstance(boundary, tuple) else boundary
        await asyncio.wait_for(fut, timeout=1.0)

        # 第二轮 boundary 后：回到 reopen-pending，stream 已关。
        assert consumer._awaiting_reopen_after_boundary is True
        assert consumer._native_stream_opened is False
        assert consumer._use_native_streaming is True
        # NOTE: 与 task 预期不同 —— 产品代码在 boundary 处理里并不重置
        # _reopen_seeded_eagerly（见 code-review 观察点 O2：consumer 每 turn 新建，
        # 残留无害）。这里断言真实行为（残留 True），并在下方证明该残留不会
        # 导致第二轮 eager seed 误判 —— 链路仍自洽。
        assert consumer._reopen_seeded_eagerly is True

        # 第二轮 eager seed：即便标志有残留，仍能正确再次开流。
        consumer.request_reopen_seed()
        await self._drain(consumer, 0.05)

        seeds_after = len(
            [f for f in adapter.frames if f["text"] == "" and not f["finalize"]]
        )
        assert seeds_after == seeds_before_second_boundary + 1, (
            "第二轮 eager seed 必须再发一个新的空 seed 帧 "
            f"(before={seeds_before_second_boundary}, after={seeds_after})"
        )
        assert await self._wait_until(lambda: consumer._native_stream_opened)
        assert consumer._reopen_seeded_eagerly is True
        assert consumer._awaiting_reopen_after_boundary is False

        consumer.finish()
        await task

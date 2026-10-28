"""Continuation-call thinking-timer regression (feat/wecom-tool-timer-animation).

The first-call fix made the FIRST API call's ``💭 Thinking`` timer push its
first frame synchronously (see ``tests/gateway/test_wecom_first_call_thinking_timer.py``).
That fix is wired to the seed / stream-open path via ``_consume_pending_thinking``
and to ``_arm_tool_timer``'s synchronous first tick.

A turn that runs tools does MANY API calls (#1 reads logs, #2 runs code, …).
Every continuation call fires ``llm.request_started`` too.  But the stream is
already open (no re-seed) and — critically — a stale ``_tool_timer_handle`` from
the just-finished tool is often still set.  ``on_llm_thinking`` then takes the
"arm now" path, records ``_thinking``, but ``need_arm`` is False because the
handle is not None, so ``_arm_tool_timer`` (the ONLY place the synchronous first
tick fires) is never called.  The ``💭 Thinking`` frame then waits for the next
``call_later(1.0)`` tick — or, in production, until the model returns ~19s later.

This test drives the REAL chain — ``TurnRunner.progress_callback`` →
``GatewayStreamConsumer`` → ``ToolTimerMixin`` — through a real ``run()`` seed,
a real tool lifecycle (``on_tool_started`` / ``on_tool_completed``), then a
continuation ``llm.request_started``.  It asserts the continuation thinking
frame is pushed immediately, which currently FAILS (no immediate tick).
"""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import conversation_loop  # noqa: F401  (real callback dispatch path)
from gateway.run_turn_runner import TurnRunner
from gateway.stream_consumer import (
    GatewayStreamConsumer,
    StreamConsumerConfig,
    _TIMER_TICK,
)
from gateway.turn_context import TurnContext


def _make_native_streaming_adapter(*, supports_tool_timer: bool = True):
    """A native-streaming BasePlatformAdapter whose seed is a real ``await``."""
    from gateway.platforms.base import BasePlatformAdapter

    NativeStreamingAdapter = type(
        "NativeStreamingAdapter",
        (BasePlatformAdapter,),
        {
            "MAX_MESSAGE_LENGTH": 4096,
            "SUPPORTS_MESSAGE_EDITING": False,
            "SUPPORTS_NATIVE_STREAMING": True,
            "SUPPORTS_TOOL_TIMER": supports_tool_timer,
        },
    )
    NativeStreamingAdapter.__abstractmethods__ = frozenset()
    adapter = NativeStreamingAdapter.__new__(NativeStreamingAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.frames = []

    def _supports(chat_type=None, metadata=None):
        return True
    adapter.supports_native_streaming = _supports

    async def _send_stream_frame(text, *, finalize=False, chat_id=None, reply_to=None, **kwargs):
        adapter.frames.append({"text": text, "finalize": finalize, "chat_id": chat_id})
        return True
    adapter.send_stream_frame = _send_stream_frame

    adapter.send = AsyncMock(
        return_value=SimpleNamespace(success=True, message_id="fallback_msg"),
    )
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    return adapter


def _make_consumer(*, supports_tool_timer: bool = True) -> GatewayStreamConsumer:
    adapter = _make_native_streaming_adapter(supports_tool_timer=supports_tool_timer)
    cfg = StreamConsumerConfig(chat_type="dm", cursor="▌")
    return GatewayStreamConsumer(adapter, "chat-1", cfg)


def _fire_request_started(sc, *, label):
    """Dispatch a ``llm.request_started`` through the REAL gateway callback."""
    ctx = TurnContext()
    ctx.tool_progress_enabled = False
    ctx.tool_timer_enabled = True
    ctx.progress_queue = queue.Queue()
    ctx._run_still_current = lambda: True
    ctx.stream_consumer_holder = [sc]
    ctx._live_status_adapter = None
    ctx._thinking_enabled = False
    TurnRunner(None, ctx).progress_callback(
        "llm.request_started", "_thinking_timer", label, None,
    )


def _drain_timer_ticks(sc) -> int:
    """Count (and remove) ``_TIMER_TICK`` sentinels currently queued."""
    ticks = 0
    pending = []
    while not sc._queue.empty():
        item = sc._queue.get_nowait()
        if item is _TIMER_TICK:
            ticks += 1
        else:
            pending.append(item)
    for item in pending:
        sc._queue.put(item)
    return ticks


class TestContinuationCallThinkingArmsImmediately:
    @pytest.mark.asyncio
    async def test_continuation_thinking_after_tool_pushes_first_frame_now(self):
        """A continuation ``llm.request_started`` arriving right after a tool
        completes (stream already open, stale timer handle still set) must push
        the ``💭 Thinking`` first frame IMMEDIATELY — same guarantee the first
        call got — not wait for the next 1s ``call_later`` tick (~19s in prod).

        Currently RED: ``on_llm_thinking`` takes the arm path but ``need_arm``
        is False (handle not None), so ``_arm_tool_timer``'s synchronous first
        tick never fires.
        """
        sc = _make_consumer()
        sc._use_native_streaming = True

        task = asyncio.create_task(sc.run())
        try:
            # Seed opens the bubble (real await inside run()).
            await asyncio.sleep(0.12)
            assert sc._native_stream_opened is True

            # A tool runs, then completes.  on_tool_completed pops the tool's
            # start entry but does NOT cancel the periodic handle, so a stale
            # _tool_timer_handle survives into the next round.
            sc.on_tool_started("terminal", tool_call_id="t1")
            await asyncio.sleep(0.05)
            assert sc._tool_timer_handle is not None  # armed by the tool

            sc.on_tool_completed("terminal", 3.0, tool_call_id="t1")
            await asyncio.sleep(0.02)  # < 1s: the stale handle is still live
            assert sc._tool_timer_handle is not None  # zombie handle survives

            # Clear any ticks already queued (tool ticks / completion tick) so
            # we measure ONLY frames caused by the continuation thinking signal.
            _drain_timer_ticks(sc)
            tick_count_before = sc._tool_timer_tick_count

            # Continuation round: API call #5 fires thinking with the bubble
            # already open and the zombie handle still set.
            _fire_request_started(sc, label="claude (API call #5)")
            # A hair for the worker→loop hop, but LESS than the 1s tick cadence:
            # a correct immediate first tick must already have fired here.
            await asyncio.sleep(0.05)

            assert "_thinking" in sc._tool_start_times  # armed the entry
            # Assert the real contract: a first tick actually FIRED and rendered
            # the 💭 frame immediately.  We measure tick_count (and drain any
            # residual sentinel to keep the queue clean) rather than the queued
            # _TIMER_TICK count — the run loop legitimately drains the sentinel
            # to push the frame within this sub-second window, so a queue-residue
            # count races the drain and reads 0 even when the frame WAS pushed.
            _drain_timer_ticks(sc)
            assert sc._tool_timer_tick_count > tick_count_before, (
                "continuation-call thinking must push its first frame "
                "immediately (like the first call): the synchronous first tick "
                "must have fired within the sub-second window and rendered the "
                "💭 Thinking frame — not wait for the next call_later tick / the "
                "~19s model return (arm path must not hit the need_arm=False no-op)"
            )
        finally:
            sc.finish()
            await task

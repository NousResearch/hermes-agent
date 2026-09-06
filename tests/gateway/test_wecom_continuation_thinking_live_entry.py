"""Regression: continuation ``on_llm_thinking`` must arm an immediate first
frame even when a PRIOR thinking timer's entry is still live.

Sibling of ``test_wecom_continuation_thinking_timer.py``, which covers the
``thinking_was_new=True`` branch (the just-finished tool left a zombie
``_tool_timer_handle``; the fresh ``_thinking`` entry routes through
``_rearm_after_tool``).

THIS test covers the OTHER production case behind the user-reported
intermittent "tool timer/progress stalls a beat after body text" symptom:
a continuation ``on_llm_thinking`` arrives while

  * a zombie ``_tool_timer_handle`` is still set (handle_present=True), AND
  * ``_thinking`` is ALREADY in ``_tool_start_times`` (thinking_was_new=False).

In ``on_llm_thinking`` that combination falls into the gap between the two
arm branches:

    if not handle_present:      # False — zombie handle set
        _arm_tool_timer()
    elif thinking_was_new:      # False — _thinking already present
        _rearm_after_tool()
    # <- neither branch runs: no synchronous first tick is scheduled

The ``💭 Thinking`` frame then waits for the zombie handle's next
``call_later(1.0)`` tick — the visible "stall a beat" the user sees, and
only intermittently because it depends on the exact residual handle/entry
combination at the body→tool boundary.

Drives the REAL chain (``TurnRunner.progress_callback`` →
``GatewayStreamConsumer`` → ``ToolTimerMixin``). Proven RED on the current
gap; a fix (arm/rearm whenever no live tick is actually running) turns it
green without breaking the existing ``thinking_was_new=True`` test.
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


class TestContinuationThinkingWithLiveThinkingEntry:
    @pytest.mark.asyncio
    async def test_immediate_first_frame_when_prior_thinking_entry_still_present(self):
        """thinking_was_new=False + handle_present=True must STILL push an
        immediate first frame — not fall into the arm/rearm gap and wait for
        the next 1s tick."""
        sc = _make_consumer()
        sc._use_native_streaming = True

        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.12)
            assert sc._native_stream_opened is True

            # Round 1: a thinking timer is live (entry + handle both set).
            _fire_request_started(sc, label="claude (API call #1)")
            await asyncio.sleep(0.05)
            assert "_thinking" in sc._tool_start_times      # thinking entry live
            assert sc._tool_timer_handle is not None          # tick loop armed

            # Body text streams (does NOT clear the thinking entry in this path),
            # then a continuation round fires thinking again — the entry is still
            # present (thinking_was_new=False) and the handle is still set.
            _drain_timer_ticks(sc)
            tick_before = sc._tool_timer_tick_count

            _fire_request_started(sc, label="claude (API call #2)")
            await asyncio.sleep(0.05)  # < 1s tick cadence

            assert "_thinking" in sc._tool_start_times
            _drain_timer_ticks(sc)
            assert sc._tool_timer_tick_count > tick_before, (
                "continuation thinking with a still-live prior _thinking entry "
                "and a set handle must still push its first frame within the "
                "sub-second window — the arm/rearm gap "
                "(handle_present=True && thinking_was_new=False) must not leave "
                "the 💭 frame waiting for the next call_later(1.0) tick"
            )
        finally:
            sc.finish()
            await task

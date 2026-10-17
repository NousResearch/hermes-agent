"""Continuation thinking-timer: synchronous-first-tick regression (deterministic).

Companion to ``test_wecom_continuation_thinking_timer.py``.  That test measures
``_TIMER_TICK`` sentinels on ``sc._queue`` — but ``run()`` drains that same queue
(``stream_consumer.py:1547``), so counting queued sentinels races the consumer
loop and is only a trustworthy signal in the *negative* direction.

This test keys on a **race-free** observable instead: ``_tool_timer_tick_count``.
It is incremented ONLY inside ``ToolTimerMixin._tool_timer_tick`` under
``_timer_lock`` (``tool_timer.py:330``) and is never read or consumed by
``run()``.  A synchronous first tick — the guarantee ``_arm_tool_timer`` gives
the FIRST call (``tool_timer.py:295-301``) — bumps this counter immediately; the
buggy continuation path (``need_arm=False`` → ``_arm_tool_timer`` skipped,
``tool_timer.py:210-213``) does not.

Real chain exercised (no simplified mocks):
``TurnRunner.progress_callback('llm.request_started', …)`` →
``GatewayStreamConsumer.on_llm_thinking`` (``ToolTimerMixin``) over a real
``run()`` seed and a real ``on_tool_started`` / ``on_tool_completed`` lifecycle.

Root cause under test (independently confirmed against current source):
after a tool completes, ``on_tool_completed`` (``tool_timer.py:151-161``) pops
the tool's ``_tool_start_times`` entry but does NOT cancel ``_tool_timer_handle``.
For WeCom native streaming a tool boundary does not reset segment state
(``stream_consumer.py:2140-2144`` takes the ``pass`` branch), so the stale
periodic handle survives.  A continuation ``on_llm_thinking`` then reaches the
arm block with ``_tool_timer_handle`` still set, so
``need_arm = handle is None`` is False and ``_arm_tool_timer`` — the only place
the synchronous first tick fires — is never called.  The ``💭 Thinking`` frame
is stuck until the next ``call_later(1.0)`` tick (in production, until the model
returns ~19s later).
"""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import conversation_loop  # noqa: F401  (imported for the real dispatch path)
from gateway.run_turn_runner import TurnRunner
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.turn_context import TurnContext


def _make_native_streaming_adapter(*, supports_tool_timer: bool = True):
    """Native-streaming BasePlatformAdapter whose seed is a genuine ``await``."""
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


class TestContinuationThinkingSyncTick:
    @pytest.mark.asyncio
    async def test_first_call_control_ticks_synchronously(self):
        """Positive control: the FIRST call's thinking DOES tick synchronously.

        Proves the harness can observe a synchronous first tick via the
        race-free ``_tool_timer_tick_count`` — so the continuation failure below
        is a genuine asymmetry, not a measurement artifact.
        """
        sc = _make_consumer()
        sc._use_native_streaming = True

        # First-call signal races the seed and latches (pre-seed window).
        _fire_request_started(sc, label="claude (API call #1)")

        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.15)  # seed opens bubble, latch consumed
            assert sc._native_stream_opened is True
            assert "_thinking" in sc._tool_start_times
            # The synchronous first tick fired: the counter advanced past 0 and a
            # thinking frame was rendered — WITHOUT waiting a 1s cadence tick.
            assert sc._tool_timer_tick_count >= 1
            assert any("💭 Thinking" in ln for ln in sc._tool_progress_lines)
        finally:
            sc.finish()
            await task

    @pytest.mark.asyncio
    async def test_continuation_thinking_after_tool_arms_synchronously(self):
        """A continuation ``llm.request_started`` arriving right after a tool
        completes (bubble already open, stale ``_tool_timer_handle`` still set)
        must push its ``💭 Thinking`` first frame synchronously — the same
        guarantee the first call got.

        Race-free assertion: ``_tool_timer_tick_count`` must advance within the
        sub-second window (before any ``call_later(1.0)`` tick could fire).

        Currently RED: ``on_llm_thinking`` records ``_thinking`` but takes the
        ``need_arm=False`` no-op branch (``tool_timer.py:210-213``), so
        ``_arm_tool_timer``'s synchronous tick never fires and the counter does
        not move until the residual tick / the model return.
        """
        sc = _make_consumer()
        sc._use_native_streaming = True

        task = asyncio.create_task(sc.run())
        try:
            # (1) Seed opens the bubble (real await in run()).
            await asyncio.sleep(0.12)
            assert sc._native_stream_opened is True

            # (2) A tool runs and arms the periodic timer, then completes.
            #     on_tool_completed pops the tool entry but leaves the handle.
            sc.on_tool_started("terminal", tool_call_id="t1")
            await asyncio.sleep(0.05)
            assert sc._tool_timer_handle is not None  # tool armed the timer

            sc.on_tool_completed("terminal", 3.0, tool_call_id="t1")
            await asyncio.sleep(0.02)  # << 1s: stale handle still live
            assert sc._tool_timer_handle is not None      # zombie handle survives
            assert "_thinking" not in sc._tool_start_times  # no thinking yet

            # Snapshot the race-free tick counter right before the continuation
            # signal.  Any advance from here is caused ONLY by that signal,
            # because we stay strictly under the 1s call_later cadence.
            tick_before = sc._tool_timer_tick_count

            # (3) Continuation round — API call #5 fires thinking with the bubble
            #     open and the zombie handle set.
            _fire_request_started(sc, label="claude (API call #5)")
            # Worker→loop hop only; far less than the 1s tick cadence.  A correct
            # synchronous first tick must already have fired within this window.
            await asyncio.sleep(0.05)

            assert "_thinking" in sc._tool_start_times, "continuation armed the entry"

            # DESIRED (fails today): the continuation thinking first frame is
            # rendered synchronously, exactly like the first call.
            assert sc._tool_timer_tick_count > tick_before, (
                "continuation thinking did not tick synchronously: "
                f"tick_count stayed at {tick_before}. on_llm_thinking hit the "
                "need_arm=False branch (stale _tool_timer_handle) so "
                "_arm_tool_timer's immediate first tick never fired; the "
                "💭 Thinking frame is stuck until the next call_later(1.0) tick "
                "/ the ~19s model return"
            )
            assert any("💭 Thinking" in ln for ln in sc._tool_progress_lines), (
                "no 💭 Thinking line rendered after the continuation signal"
            )
        finally:
            sc.finish()
            await task

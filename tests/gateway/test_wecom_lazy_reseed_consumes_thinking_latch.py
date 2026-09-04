"""Lazy re-seed must consume the pending-thinking latch (bug2, 19s blank).

Companion to the continuation-thinking *sync-tick* tests (which cover the
stream-OPEN zombie-handle path).  This one covers the stream-CLOSED path:

A mid-turn boundary (clarify / proactive rotation) closes the native stream
(``_native_stream_opened=False``, awaiting reopen).  A continuation
``llm.request_started`` that arrives while the stream is closed cannot arm the
timer — ``on_llm_thinking`` latches it (``_pending_thinking=True``,
``tool_timer.py:186-189``) for a later reopen to honour.

The initial seed (``stream_consumer.py:1414``) and the eager clarify re-seed
(``:1614``) both call ``_consume_pending_thinking()`` right after opening the
bubble.  The LAZY re-seed (``:3356``, the fallback that reopens on the first
frame after a boundary) sets ``_native_stream_opened=True`` but — before the
fix — does NOT consume the latch.  So a latch honoured only by content arrival
(``_append_accumulated`` clears it at ``:661``) is orphaned when the reopen is
driven by a non-content overlay frame: the ``💭 Thinking`` animation never
appears during the gap (in production, ~19s of dead typing).

Real chain exercised (no simplified mocks): a real ``run()`` seed, the real
``close_for_approval_prompt(reopen=True)`` boundary, the real
``TurnRunner.progress_callback('llm.request_started', …)`` →
``on_llm_thinking`` latch, and a real ``on_tool_progress`` overlay frame that
drives the run loop into the lazy re-seed at ``_send_or_edit`` (``:3356``).

Currently RED: after the overlay reopens the stream, ``_pending_thinking`` is
still True and ``_thinking`` is never armed, because the lazy re-seed skips
``_consume_pending_thinking()``.
"""

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import conversation_loop  # noqa: F401  (real callback dispatch path)
from gateway.run import TurnRunner
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


class TestLazyReseedConsumesThinkingLatch:
    @pytest.mark.asyncio
    async def test_lazy_reseed_after_boundary_consumes_pending_thinking(self):
        """A continuation thinking latched while the stream is closed must be
        honoured when the lazy re-seed reopens the stream — same as the initial
        seed (``:1414``) and the eager re-seed (``:1614``) do.

        Currently RED: the lazy re-seed (``:3356``) reopens the bubble but never
        calls ``_consume_pending_thinking()``, so ``_pending_thinking`` stays
        latched and ``_thinking`` is never armed — the ``💭 Thinking`` frame is
        stuck until real content arrives (~19s later), which then clears the
        latch instead of showing it.
        """
        sc = _make_consumer()
        sc._use_native_streaming = True

        task = asyncio.create_task(sc.run())
        try:
            # (1) Seed opens the bubble and captures the timer loop.
            await asyncio.sleep(0.12)
            assert sc._native_stream_opened is True

            # (2) A real mid-turn boundary closes the stream, keeping native
            #     enabled and awaiting a lazy reopen (clarify semantics).
            fut, _cancelled = sc.close_for_approval_prompt(
                placeholder="…", reason="Clarify", reopen=True,
            )
            await asyncio.wait_for(fut, timeout=2)
            await asyncio.sleep(0.05)
            assert sc._native_stream_opened is False       # stream closed
            assert sc._awaiting_reopen_after_boundary is True
            assert not sc._accumulated                     # no content yet

            # (3) Continuation round: API call #5 fires thinking WHILE the
            #     stream is closed — on_llm_thinking latches it.
            _fire_request_started(sc, label="claude (API call #5)")
            await asyncio.sleep(0.05)
            assert sc._pending_thinking is True             # latched
            assert "_thinking" not in sc._tool_start_times  # not armed yet

            # (4) A non-content overlay frame (a tool-progress status) reaches
            #     the run loop and drives the LAZY re-seed at
            #     _send_or_edit (stream_consumer.py:3356), reopening the bubble
            #     with _accumulated still empty. This is the reopen path that
            #     the pending-thinking latch depends on being consumed at.
            sc.on_tool_progress('⚙️ terminal: "pytest"', tool_call_id="t9")
            await asyncio.sleep(0.1)

            assert sc._native_stream_opened is True, (
                "lazy re-seed should have reopened the native stream"
            )
            # FIX (fails today): the latch is consumed at the lazy re-seed, so
            # the continuation thinking arms and its 💭 frame renders instead of
            # being orphaned until content arrives.
            assert sc._pending_thinking is False, (
                "lazy re-seed left _pending_thinking latched — "
                "_consume_pending_thinking() was not called at "
                "stream_consumer.py:3365 (unlike the initial seed :1414 and the "
                "eager re-seed :1614), so the 💭 Thinking latch is orphaned "
                "until real content arrives ~19s later"
            )
            assert "_thinking" in sc._tool_start_times, (
                "continuation thinking was never armed after the lazy re-seed"
            )
            await asyncio.sleep(0.05)
            assert any("💭 Thinking" in ln for ln in sc._tool_progress_lines), (
                "no 💭 Thinking frame rendered after the lazy re-seed consumed "
                "the latch"
            )
        finally:
            sc.finish()
            await task

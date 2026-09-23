"""Tool-generation liveness regression (WeCom native tool-timer).

Symptom (user screen-recording, 2026-09-15): the model streams a chunk of BODY
text, finishes it (e.g. "…写 prompt:"), then goes quiet for 1-10s while it emits
the *arguments* of the next tool call — with the turn NOT yet done.  Nothing
changes the bubble's bytes during that window and WeCom only refreshes a native
frame on data change, so the bubble sits frozen: it looks like the LLM hung.  The
tool timer only appeared once the tool actually executed (``tool.started``).

Root cause: ``_append_accumulated`` stops the thinking timer the instant body text
arrives ("real content wins").  Between the last body delta and ``tool.started``
(or the next ``llm.request_started``) nothing re-armed an animation, because those
were the only re-arm signals and both fire only AFTER the vacuum ends.

Fix: a new agent→gateway callback, ``tool_gen_callback``, bound to
``TurnRunner._tool_gen_started_sync``.  The streaming decoders call
``_fire_tool_gen_started(name)`` from ``_emit_tool_started`` on BOTH wires
(chat_completions: first named tool-call delta; anthropic_messages:
``content_block_start(type=tool_use)``) — i.e. at the START of the tool-arg
vacuum, long before the tool executes.  ``_tool_gen_started_sync`` calls the
zombie-safe, idempotent ``on_llm_thinking`` so the native bubble animates a
``💭 Thinking`` line below the body through the vacuum instead of freezing.

These tests guard BOTH ends of that wiring, which a rebase has silently dropped
before (see the animation-lost-in-rebase history):

1. Agent end: ``_emit_tool_started`` invokes the bound ``tool_gen_callback``.
2. Gateway end: the full chain arms the thinking timer and pushes a frame the
   instant tool generation starts after body text — no 1-10s frozen gap.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.stream_consumer import (
    GatewayStreamConsumer,
    StreamConsumerConfig,
    _TIMER_TICK,
)


# ── Agent end: _emit_tool_started fires the bound tool_gen_callback ──────────

class TestEmitToolStartedFiresGenCallback:
    def test_emit_tool_started_invokes_tool_gen_callback(self):
        """``_emit_tool_started`` must call ``agent._fire_tool_gen_started`` which
        forwards to the bound ``tool_gen_callback`` with the tool name.  This is
        the agent-side half of the freeze fix; a rebase dropping the call here
        would silently reopen the frozen-bubble gap with no test failure."""
        from agent.chat_completion_helpers import _StreamingCall

        fired: list[str] = []
        agent = SimpleNamespace(
            _fire_tool_gen_started=lambda name: fired.append(name),
        )
        # first_delta_fired latch + on_first_delta are consulted by _fire_first_delta.
        call = _StreamingCall.__new__(_StreamingCall)
        call.agent = agent
        call.first_delta_fired = {"done": True}  # already fired: isolate the tool-gen call
        call.on_first_delta = None

        call._emit_tool_started("terminal")

        assert fired == ["terminal"], (
            "_emit_tool_started must forward the tool name to _fire_tool_gen_started "
            "(the tool-arg-vacuum liveness signal) — the START of the tool call, not "
            "its later execution"
        )

    def test_fire_tool_gen_started_forwards_to_bound_callback(self):
        """``_fire_tool_gen_started`` must forward to the agent's
        ``tool_gen_callback`` (what ``TurnRunner`` binds to
        ``_tool_gen_started_sync``).  Guards the agent→gateway hop."""
        from agent.stream_delivery import StreamDeliveryMixin

        seen: list[str] = []
        host = StreamDeliveryMixin.__new__(StreamDeliveryMixin)
        host.tool_gen_callback = lambda name: seen.append(name)

        host._fire_tool_gen_started("write_file")

        assert seen == ["write_file"]


# ── Binding: TurnRunner routes tool-gen start into on_llm_thinking ───────────

class TestTurnRunnerBinding:
    def test_tool_gen_started_sync_arms_thinking_on_consumer(self):
        """``TurnRunner._tool_gen_started_sync`` (bound to ``agent.tool_gen_callback``
        in ``_wire_turn_agent_callbacks``) must call ``on_llm_thinking`` on the live
        stream consumer.  This is the exact hop a rebase dropped before, silently
        reopening the frozen-bubble gap — guard it directly."""
        from gateway.run_turn_runner import TurnRunner
        from gateway.turn_context import TurnContext

        armed: list[bool] = []
        sc = SimpleNamespace(
            supports_tool_timer=True,
            on_llm_thinking=lambda *a, **k: armed.append(True),
        )
        ctx = TurnContext()
        ctx.stream_consumer_holder = [sc]

        TurnRunner(None, ctx)._tool_gen_started_sync("write_file")

        assert armed == [True], (
            "_tool_gen_started_sync must arm the thinking timer via on_llm_thinking "
            "when a timer-opted-in native consumer is live — the gateway half of the "
            "tool-arg-vacuum freeze fix"
        )

    def test_tool_gen_started_sync_no_consumer_is_safe(self):
        """No live consumer (holder empty) → no-op, no crash."""
        from gateway.run_turn_runner import TurnRunner
        from gateway.turn_context import TurnContext

        ctx = TurnContext()
        ctx.stream_consumer_holder = []
        TurnRunner(None, ctx)._tool_gen_started_sync("terminal")  # must not raise

    def test_tool_gen_started_sync_skips_non_timer_consumer(self):
        """A native consumer WITHOUT the timer opt-in must not be armed (parity with
        the ``llm.request_started`` handler's ``supports_tool_timer`` gate)."""
        from gateway.run_turn_runner import TurnRunner
        from gateway.turn_context import TurnContext

        armed: list[bool] = []
        sc = SimpleNamespace(
            supports_tool_timer=False,
            on_llm_thinking=lambda *a, **k: armed.append(True),
        )
        ctx = TurnContext()
        ctx.stream_consumer_holder = [sc]
        TurnRunner(None, ctx)._tool_gen_started_sync("terminal")
        assert armed == []


# ── Gateway end: the chain closes the freeze immediately ────────────────────

def _make_native_streaming_adapter():
    from gateway.platforms.base import BasePlatformAdapter

    NativeStreamingAdapter = type(
        "NativeStreamingAdapter",
        (BasePlatformAdapter,),
        {
            "MAX_MESSAGE_LENGTH": 4096,
            "SUPPORTS_MESSAGE_EDITING": False,
            "SUPPORTS_NATIVE_STREAMING": True,
            "SUPPORTS_TOOL_TIMER": True,
        },
    )
    NativeStreamingAdapter.__abstractmethods__ = frozenset()
    adapter = NativeStreamingAdapter.__new__(NativeStreamingAdapter)
    adapter._typing_paused = set()
    adapter._fatal_error_message = None
    adapter.frames = []
    adapter.supports_native_streaming = lambda chat_type=None, metadata=None: True

    async def _send_stream_frame(text, *, finalize=False, chat_id=None, reply_to=None, **kwargs):
        adapter.frames.append(text)
        return True
    adapter.send_stream_frame = _send_stream_frame
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m"))
    adapter.edit_message = AsyncMock(return_value=SimpleNamespace(success=True))
    return adapter


def _make_consumer() -> GatewayStreamConsumer:
    cfg = StreamConsumerConfig(chat_type="dm", cursor="▌")
    sc = GatewayStreamConsumer(_make_native_streaming_adapter(), "chat-1", cfg)
    sc._use_native_streaming = True
    return sc


def _drain_timer_ticks(sc) -> int:
    ticks, pending = 0, []
    while not sc._queue.empty():
        item = sc._queue.get_nowait()
        if item is _TIMER_TICK:
            ticks += 1
        else:
            pending.append(item)
    for item in pending:
        sc._queue.put(item)
    return ticks


class TestToolGenClosesFreeze:
    @pytest.mark.asyncio
    async def test_tool_gen_after_body_arms_thinking_immediately(self):
        """Body text arrives (stopping the thinking timer), then the model begins
        generating a tool call.  The ``tool_gen_callback`` path (``on_llm_thinking``)
        must arm the thinking heartbeat and push a ``💭 Thinking`` frame IMMEDIATELY
        — closing the 1-10s frozen gap, not waiting for ``tool.started``.

        This drives the exact call ``TurnRunner._tool_gen_started_sync`` makes.
        """
        sc = _make_consumer()
        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.12)
            assert sc._native_stream_opened is True

            # Body streams and stops — real content wins, thinking timer stopped.
            sc.on_delta("这次 cc 的任务是：写 prompt:")
            await asyncio.sleep(0.10)
            assert sc._accumulated
            assert sc._tool_timer_handle is None  # no animation while/after body
            assert "_thinking" not in sc._tool_start_times

            _drain_timer_ticks(sc)
            frames_before = len(sc.adapter.frames)

            # Tool generation begins (the vacuum starts). This is precisely what
            # _tool_gen_started_sync invokes on the agent worker thread.
            sc.on_llm_thinking()
            # Well under the 1s tick cadence: a correct immediate first tick must
            # already have rendered the frame here.
            await asyncio.sleep(0.08)

            assert "_thinking" in sc._tool_start_times, (
                "tool-generation start must arm the thinking heartbeat so the "
                "native bubble animates through the tool-arg vacuum instead of "
                "freezing (the 1-10s 'LLM hung' symptom)"
            )
            _drain_timer_ticks(sc)
            assert len(sc.adapter.frames) > frames_before, (
                "a 💭 Thinking frame must be pushed immediately when tool "
                "generation begins — not deferred to the tool's execution"
            )
            assert any("Thinking" in f for f in sc.adapter.frames[frames_before:]), (
                "the pushed frame must carry the 💭 Thinking line below the body"
            )
            # Body must be preserved above the overlay (no prefix loss).
            assert any("写 prompt:" in f for f in sc.adapter.frames[frames_before:])
        finally:
            sc.finish()
            await task

    @pytest.mark.asyncio
    async def test_tool_gen_then_tool_started_does_not_double_arm(self):
        """``on_llm_thinking`` is idempotent: a tool-gen arm followed by the later
        ``on_tool_started`` (real execution) must not stack two tick loops."""
        sc = _make_consumer()
        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.12)
            sc.on_delta("body 写 prompt:")
            await asyncio.sleep(0.08)

            sc.on_llm_thinking()          # tool-gen vacuum begins
            await asyncio.sleep(0.05)
            handle_after_gen = sc._tool_timer_handle
            assert handle_after_gen is not None

            # The tool actually executes a beat later. The timer must transition
            # from _thinking to the tool cleanly, with exactly one live handle.
            sc.on_tool_started("write_file", tool_call_id="w1")
            await asyncio.sleep(0.05)
            assert sc._tool_timer_handle is not None
            # Single live tick loop: no zombie left behind (would double the cadence).
            assert isinstance(sc._tool_timer_handle, asyncio.TimerHandle)
        finally:
            sc.finish()
            await task

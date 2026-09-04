"""First-LLM-call thinking-timer regression (feat/wecom-tool-timer-animation).

Confirmed bug: ``agent/conversation_loop.py`` fired ``llm.request_started``
only when ``api_call_count > 1``.  ``api_call_count`` resets per turn, so every
turn's FIRST model request reached the wire without the ``💭 Thinking`` timer.
A real first API call hung 342s and WeCom showed only the static typing
indicator.

Fixing that call-site exposes a second defect: the first request signal races
the native seed.  The seed frame is a real network round-trip; the agent's
first ``llm.request_started`` can arrive while ``run()`` is still awaiting it —
so ``_native_stream_opened`` is still False (and the timer loop may not be
captured yet).  ``ToolTimerMixin.on_llm_thinking`` dropped the signal in that
window, leaving thinking un-armed for the whole first call.

These tests drive the REAL production handoff — ``TurnRunner.progress_callback``
dispatching ``llm.request_started`` into a REAL ``GatewayStreamConsumer`` whose
``run()`` performs the seed — rather than calling ``on_llm_thinking`` directly.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import conversation_loop
from gateway.run import TurnRunner
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from gateway.turn_context import TurnContext


def _make_native_streaming_adapter(*, seed_ok: bool = True, supports_tool_timer: bool = True):
    """A BasePlatformAdapter that streams natively and records seed frames.

    ``send_stream_frame`` is a coroutine so the seed is a genuine ``await``
    point in ``run()`` — the race window the latch must survive.
    """
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
        return seed_ok
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


def _make_ctx(sc):
    ctx = TurnContext()
    ctx.tool_progress_enabled = False
    ctx.tool_timer_enabled = True
    ctx.progress_queue = queue.Queue()
    ctx._run_still_current = lambda: True
    ctx.stream_consumer_holder = [sc]
    ctx._live_status_adapter = None
    ctx._thinking_enabled = False
    return ctx


def _fire_request_started(sc, *, label="claude (API call #1)"):
    """Dispatch the FIRST-call signal through the real gateway callback."""
    ctx = _make_ctx(sc)
    runner = TurnRunner(None, ctx)
    runner.progress_callback("llm.request_started", "_thinking_timer", label, None)


# ── Part A: the call-site must fire on the FIRST API call ────────────────────


class TestFirstCallSignalNotGated:
    def test_llm_request_started_not_gated_behind_api_call_count(self):
        """Source guard: ``llm.request_started`` must not sit behind
        ``api_call_count > 1``.  The per-turn reset of ``api_call_count`` means
        such a gate drops every turn's first-call thinking timer (#342s hang).
        """
        tree = ast.parse(inspect.getsource(conversation_loop.run_conversation))

        # Locate the callback call that emits "llm.request_started".
        emit_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and any(
                isinstance(a, ast.Constant) and a.value == "llm.request_started"
                for a in node.args
            )
        ]
        assert emit_calls, "expected an llm.request_started emit in run_conversation"

        # Build a parent map so each emit can be bound to its enclosing `if`.
        parents = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        def _guarded_by_api_call_count_gt_1(node) -> bool:
            cur = parents.get(node)
            while cur is not None:
                if isinstance(cur, ast.If):
                    for cmp in ast.walk(cur.test):
                        if (
                            isinstance(cmp, ast.Compare)
                            and isinstance(cmp.left, ast.Name)
                            and cmp.left.id == "api_call_count"
                            and any(isinstance(op, ast.Gt) for op in cmp.ops)
                        ):
                            return True
                cur = parents.get(cur)
            return False

        for emit in emit_calls:
            assert not _guarded_by_api_call_count_gt_1(emit), (
                "llm.request_started must fire on the first API call too; a "
                "`api_call_count > 1` gate drops the first-turn thinking timer"
            )


# ── Part B: the pre-seed race must not drop the signal ───────────────────────


class TestFirstCallThinkingSurvivesSeedRace:
    @pytest.mark.asyncio
    async def test_signal_before_seed_then_arms_before_content(self):
        """Real ordering: first-call ``llm.request_started`` arrives BEFORE the
        seed opens the bubble; once ``run()`` seeds + captures the loop, the
        latched signal is consumed and thinking arms and ticks before any model
        content.
        """
        sc = _make_consumer()
        sc._use_native_streaming = True   # native resolved (run() also re-resolves)
        assert sc._native_stream_opened is False  # seed has NOT happened yet

        # (1) First-call signal fires through the real gateway callback while
        #     the bubble is still unopened — the exact race that hung 342s.
        _fire_request_started(sc)

        # (2) Now run() seeds (an await), captures the loop, and must consume
        #     the latched signal.
        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.15)

            # Seed happened, and no model content has arrived yet.
            assert sc._native_stream_opened is True
            assert sc._accumulated == ""

            # Thinking is armed before any content.
            assert "_thinking" in sc._tool_start_times
            assert sc._tool_timer_handle is not None

            # A tick renders the generic thinking status (privacy: no label).
            sc._tool_timer_tick()
            joined = "\n".join(sc._tool_progress_lines)
            assert "💭 Thinking" in joined
        finally:
            sc.finish()
            await task

    @pytest.mark.asyncio
    async def test_first_text_delta_clears_thinking(self):
        """First real text delta must stop the thinking timer (content wins)."""
        sc = _make_consumer()
        sc._use_native_streaming = True
        _fire_request_started(sc)

        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.15)
            assert "_thinking" in sc._tool_start_times  # armed pre-content

            sc.on_delta("Hello, here is the answer.")
            await asyncio.sleep(0.1)

            assert "_thinking" not in sc._tool_start_times
            assert sc._tool_timer_handle is None
        finally:
            sc.finish()
            await task

    @pytest.mark.asyncio
    async def test_got_done_clears_thinking(self):
        """got_done finalize must stop the thinking timer."""
        sc = _make_consumer()
        sc._use_native_streaming = True
        _fire_request_started(sc)

        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.15)
            assert "_thinking" in sc._tool_start_times

            sc.finish()
            await task

            # Stream is done: no lingering thinking entry / armed handle.
            assert "_thinking" not in sc._tool_start_times
            assert sc._tool_timer_handle is None
        finally:
            if not task.done():
                sc.finish()
                await task

    @pytest.mark.asyncio
    async def test_fast_first_call_leaves_no_persistent_thinking(self):
        """A first call that produces content almost immediately must not leave
        a persistent ``💭 Thinking`` timer — content clears the pending latch
        even if the signal arrived slightly before content."""
        sc = _make_consumer()
        sc._use_native_streaming = True

        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.1)  # seed completes first
            # Signal and content arrive nearly together.
            _fire_request_started(sc)
            sc.on_delta("Immediate answer.")
            await asyncio.sleep(0.15)

            assert "_thinking" not in sc._tool_start_times
            assert sc._pending_thinking is False
            assert sc._tool_timer_handle is None
        finally:
            sc.finish()
            await task

    @pytest.mark.asyncio
    async def test_timer_disabled_default_off_unchanged(self):
        """With the tool-timer opt-in OFF, the first-call signal is a no-op:
        no latch, no ``_thinking`` entry, default behaviour preserved."""
        sc = _make_consumer(supports_tool_timer=False)
        sc._use_native_streaming = True
        assert sc.supports_tool_timer is False

        _fire_request_started(sc)
        assert sc._pending_thinking is False

        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.15)
            assert "_thinking" not in sc._tool_start_times
            assert sc._tool_timer_handle is None
        finally:
            sc.finish()
            await task

    @pytest.mark.asyncio
    async def test_subsequent_call_after_open_still_arms(self):
        """A signal arriving AFTER the bubble is open (the classic
        ``api_call_count > 1`` case) arms thinking directly — no regression."""
        sc = _make_consumer()
        sc._use_native_streaming = True

        task = asyncio.create_task(sc.run())
        try:
            await asyncio.sleep(0.1)
            assert sc._native_stream_opened is True  # already open

            _fire_request_started(sc, label="claude (API call #3)")
            await asyncio.sleep(0.1)

            assert "_thinking" in sc._tool_start_times
            assert sc._tool_timer_handle is not None
        finally:
            sc.finish()
            await task

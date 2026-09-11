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

import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent import conversation_loop
from gateway.run_turn_runner import TurnRunner
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
    """Behaviour contract: driving the REAL turn loop, the very first API call of
    a turn emits ``llm.request_started`` through ``tool_progress_callback``.

    The bug this pins: ``api_call_count`` resets to 0 each turn, so gating the
    emit behind ``> 1`` dropped every turn's first-call thinking timer (#342s
    hang). We drive ``_run_conversation_turn`` with the pre-emit phases stubbed
    to no-ops and the API step raised out via a sentinel, then assert what the
    callback actually received — not the shape of the source.
    """

    class _Sentinel(Exception):
        pass

    def _drive_first_turn(self, monkeypatch, *, budget_remaining: int = 10):
        """Drive the REAL ``_run_conversation_turn`` through its first loop
        iteration, capturing every ``tool_progress_callback`` invocation.

        We inject a fake TurnContext (so no live agent/session is needed) and
        stub the phase helpers, but the emit block itself runs on this build's
        real ``_run_conversation_turn`` code — the API step raises a sentinel
        that unwinds the loop right after the emit point.

        Returns the list of ``(signal, *rest)`` tuples the callback received.
        """
        emitted = []

        # ── Fake agent: only the attributes the real loop reads before the API. ──
        agent = SimpleNamespace(
            api_mode="chat_completions",
            model="claude",
            max_iterations=500,
            iteration_budget=SimpleNamespace(remaining=budget_remaining),
            _budget_grace_call=False,
            _current_api_request_id=None,
            _last_compaction_in_place=False,
            _last_compression_attempt_recorded=False,
            _last_compression_attempt_in_place=None,
            _delivered_interim_texts=set(),
            _incremental_persistence_failed=False,
            _last_persistence_error_cause=None,
            _compression_adoption_failed=False,
            _ephemeral_reasoning_off=False,
            _auth_pool_refresh_counts={},
            _last_turn_usage=None,
            max_compression_attempts=3,
            _api_max_retries=3,
            tool_progress_callback=lambda *a: emitted.append(a),
        )
        agent._try_refresh_env_client_credentials = lambda: None

        # begin_fast_mode_turn touches the agent; stub to a no-op.
        monkeypatch.setattr(conversation_loop, "begin_fast_mode_turn", lambda *a, **k: None)

        # finalize_turn runs only when the loop exits WITHOUT making a call
        # (budget-exhausted case); raise the sentinel there too so neither path
        # needs a fully-wired agent past the point we care about. Real code
        # introspects finalize_turn's signature to build kwargs, so the stub
        # must preserve the ORIGINAL signature (functools.wraps) — a bare
        # ``**kwargs`` stub would make the caller do getattr(s, "kwargs").
        import functools as _functools

        _orig_finalize = conversation_loop.finalize_turn

        @_functools.wraps(_orig_finalize)
        def _fake_finalize(*args, **kwargs):
            raise self._Sentinel()

        monkeypatch.setattr(conversation_loop, "finalize_turn", _fake_finalize)

        # ── Fake TurnContext: build_turn_context normally builds this from a live
        #    agent+session. We hand back an object exposing exactly the fields
        #    _LoopState seeds from it (names minus the leading underscore). ──
        ctx_values = {
            "user_message": "hi",
            "original_user_message": "hi",
            "conversation_history": [],
            "effective_task_id": None,
            "turn_id": "turn-1",
            "should_review_memory": False,
            "plugin_user_context": None,
            "ext_prefetch_cache": None,
            "messages": [],
            "active_system_prompt": "sys",
            "current_turn_user_idx": 0,
            "preflight_compression_blocked": False,
        }
        fake_ctx = SimpleNamespace(**ctx_values)
        monkeypatch.setattr(conversation_loop, "build_turn_context", lambda *a, **k: fake_ctx)

        # ── Stub phase helpers. begin_iteration must increment api_call_count
        #    exactly as the real helper does (the grace flag is off), then every
        #    phase reports "proceed" so the loop reaches the emit line. ──
        def _fake_run_phase(fn, _agent, state, **extra):
            if fn is conversation_loop.begin_iteration:
                state.api_call_count += 1
            return SimpleNamespace(action="proceed")

        monkeypatch.setattr(conversation_loop, "_run_phase", _fake_run_phase)

        # The API step cuts the loop AFTER the emit block above it has run.
        def _fake_api_retry_loop(_agent, _s):
            raise self._Sentinel()

        monkeypatch.setattr(conversation_loop, "_run_api_retry_loop", _fake_api_retry_loop)

        with pytest.raises(self._Sentinel):
            conversation_loop._run_conversation_turn(agent, "hi")

        return emitted

    def test_first_call_emits_request_started(self, monkeypatch):
        """FIRST API call of a turn (api_call_count becomes 1) MUST emit the
        signal — driven through the real _run_conversation_turn."""
        emitted = self._drive_first_turn(monkeypatch)
        signals = [e[0] for e in emitted]
        assert "llm.request_started" in signals, (
            "the first API call of a turn must emit llm.request_started; a "
            "`> 1` gate drops the first-turn thinking timer (#342s hang)"
        )
        # The emit carries the first-call label and the timer channel.
        first = next(e for e in emitted if e[0] == "llm.request_started")
        assert first[1] == "_thinking_timer"
        assert "#1" in first[2]

    def test_no_emit_when_budget_exhausted(self, monkeypatch):
        """The loop never enters an iteration when there is no budget, so no
        API call is counted and no signal fires — proves the emit is bound to
        a real request, not unconditional."""
        emitted = self._drive_first_turn(monkeypatch, budget_remaining=0)
        signals = [e[0] for e in emitted]
        assert "llm.request_started" not in signals, (
            "with no iteration budget the turn makes no API call, so the "
            "thinking signal must not fire"
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

"""Issue C: timer lifecycle events must reach the stream consumer even when
``display.tool_progress`` is off, as long as ``extra.tool_timer_enabled`` is
true.

These tests drive ``TurnRunner.progress_callback`` directly with a fake
stream consumer, isolating the dispatch gate from the full turn machinery.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock

import gateway.run_turn_runner as rtr
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext


class _FakeStreamConsumer:
    def __init__(self, supports_tool_timer: bool = True):
        self.supports_tool_timer = supports_tool_timer
        self.started = []
        self.completed = []
        self.progress = []
        self.thinking = []

    def on_tool_started(self, tool_name, tool_call_id=None):
        self.started.append((tool_name, tool_call_id))

    def on_tool_completed(self, tool_name, duration, tool_call_id=None):
        self.completed.append((tool_name, duration, tool_call_id))

    def on_tool_progress(self, line, tool_call_id=None):
        self.progress.append((line, tool_call_id))

    def on_llm_thinking(self, label=None):
        self.thinking.append(label)


def _make_ctx(*, tool_progress_enabled, tool_timer_enabled, sc):
    ctx = TurnContext()
    ctx.tool_progress_enabled = tool_progress_enabled
    ctx.tool_timer_enabled = tool_timer_enabled
    # progress_queue must be truthy for the callback to proceed past its
    # queue guard; the timer path uses the consumer, not this queue.
    ctx.progress_queue = queue.Queue()
    ctx._run_still_current = lambda: True
    ctx.stream_consumer_holder = [sc]
    ctx._live_status_adapter = None
    ctx._thinking_enabled = False
    return ctx


class TestTimerLifecycleWhenProgressOff:
    def test_tool_started_and_completed_dispatch_with_progress_off(self):
        sc = _FakeStreamConsumer(supports_tool_timer=True)
        ctx = _make_ctx(
            tool_progress_enabled=False, tool_timer_enabled=True, sc=sc
        )
        runner = TurnRunner(None, ctx)

        runner.progress_callback("tool.started", "terminal", "python x.py", {}, tool_call_id="c1")
        runner.progress_callback("tool.completed", "terminal", None, None, tool_call_id="c1", duration=3.0)

        # Timer got the bare tool name — no arguments — and the completion.
        assert sc.started == [("terminal", "c1")]
        assert sc.completed == [("terminal", 3.0, "c1")]
        # No overlay progress line was injected (progress display is off).
        assert sc.progress == []

    def test_clarify_tool_started_not_dispatched(self):
        sc = _FakeStreamConsumer(supports_tool_timer=True)
        ctx = _make_ctx(
            tool_progress_enabled=False, tool_timer_enabled=True, sc=sc
        )
        runner = TurnRunner(None, ctx)

        runner.progress_callback("tool.started", "clarify", None, None, tool_call_id="c1")

        assert sc.started == []

    def test_no_dispatch_when_timer_disabled(self):
        sc = _FakeStreamConsumer(supports_tool_timer=False)
        ctx = _make_ctx(
            tool_progress_enabled=False, tool_timer_enabled=False, sc=sc
        )
        runner = TurnRunner(None, ctx)

        runner.progress_callback("tool.started", "terminal", None, None, tool_call_id="c1")
        runner.progress_callback("tool.completed", "terminal", None, None, tool_call_id="c1", duration=1.0)

        assert sc.started == []
        assert sc.completed == []

    def test_llm_thinking_reaches_consumer_with_progress_off(self):
        sc = _FakeStreamConsumer(supports_tool_timer=True)
        ctx = _make_ctx(
            tool_progress_enabled=False, tool_timer_enabled=True, sc=sc
        )
        runner = TurnRunner(None, ctx)

        runner.progress_callback(
            "llm.request_started", "_thinking_timer", "claude (API call #2)", None
        )

        assert sc.thinking == ["claude (API call #2)"]


class TestTimerCompletionWhenProgressOn:
    """Issue #4 (unsupportedpastels review of 96942): with tool_progress ON and
    the timer ON, a ``tool.completed`` event must still reach
    ``on_tool_completed``. It previously did NOT: the onboarding-hint branch
    (``event_type == "tool.completed" and not long_tool_hint_fired[0]``) returned
    early on the DEFAULT ``long_tool_hint_fired == [False]``, before the timer
    completion dispatch, so a finished tool kept rendering as active until a
    later thinking/text event cleared it.
    """

    def test_tool_completed_dispatches_timer_with_progress_on(self):
        sc = _FakeStreamConsumer(supports_tool_timer=True)
        ctx = _make_ctx(
            tool_progress_enabled=True, tool_timer_enabled=True, sc=sc
        )
        # Default gate state: the onboarding hint has not fired yet.
        assert ctx.long_tool_hint_fired == [False]
        runner = TurnRunner(None, ctx)

        # A short tool (below the long-tool hint threshold) completes.
        runner.progress_callback(
            "tool.completed", "terminal", None, None,
            tool_call_id="c1", duration=1.0,
        )

        # The completion MUST reach the timer so the tool stops rendering active.
        assert sc.completed == [("terminal", 1.0, "c1")]


class TestNativeTaskCardsSkipTimerDispatch:
    """Native Slack task cards consume the ID-bearing start/complete callbacks
    themselves, so the name-correlated timer dispatch must be skipped entirely
    (it would duplicate cards and mispair concurrent same-tool calls).
    """

    def test_task_cards_skip_timer_dispatch(self):
        sc = _FakeStreamConsumer(supports_tool_timer=True)
        ctx = _make_ctx(
            tool_progress_enabled=True, tool_timer_enabled=True, sc=sc
        )
        ctx._native_slack_task_cards = True
        runner = TurnRunner(None, ctx)

        runner.progress_callback("tool.started", "terminal", None, None, tool_call_id="c1")
        runner.progress_callback(
            "tool.completed", "terminal", None, None, tool_call_id="c1", duration=2.0
        )

        # The name-correlated timer path is bypassed; task cards handle these.
        assert sc.started == []
        assert sc.completed == []


class TestStartedFallsThroughWhenProgressOn:
    """With tool_progress ON, a ``tool.started`` event dispatches the timer AND
    falls through to the ordinary progress-rendering path (only ``tool.completed``
    returns early inside the timer block). The started line must be emitted.
    """

    def test_started_dispatches_timer_and_renders_progress(self):
        sc = _FakeStreamConsumer(supports_tool_timer=True)
        ctx = _make_ctx(
            tool_progress_enabled=True, tool_timer_enabled=True, sc=sc
        )
        ctx.progress_mode = "all"
        ctx._agent_interrupted = lambda: False
        emitted = []
        runner = TurnRunner(None, ctx)
        # Capture the ordinary progress-path emit (post-timer fall-through).
        runner._progress_emit = lambda msg, tool_call_id=None: emitted.append(msg)

        runner.progress_callback(
            "tool.started", "terminal", "python x.py", {}, tool_call_id="c1"
        )

        # Timer got the started event...
        assert sc.started == [("terminal", "c1")]
        # ...and the ordinary progress path still rendered the started line.
        assert len(emitted) == 1


class TestOnboardingHintStillFiresWithTimerOn:
    """Q3 regression (#96942 review): the onboarding hint's only entry point is
    the ``tool.completed`` branch. The unified timer block's early return
    (``event_type == "tool.completed" or not tool_progress_enabled``) would have
    swallowed the hint on every timer-enabled turn. The fix runs the hint before
    that return, so a long tool with the /verbose gate open still fires it once.
    """

    def test_hint_fires_with_timer_and_progress_on(self, monkeypatch):
        import agent.onboarding as onboarding
        import gateway.run as grun

        sc = _FakeStreamConsumer(supports_tool_timer=True)
        ctx = _make_ctx(
            tool_progress_enabled=True, tool_timer_enabled=True, sc=sc
        )
        ctx.progress_mode = "all"  # hint only fires in stream-all mode
        assert ctx.long_tool_hint_fired == [False]
        runner = TurnRunner(None, ctx)

        # Open the /verbose gate and stub the onboarding side effects.
        monkeypatch.setattr(grun, "_load_gateway_config",
                            lambda: {"display": {"tool_progress_command": True}})
        monkeypatch.setattr(rtr, "cfg_get", lambda cfg, *ks: True)
        monkeypatch.setattr(rtr, "is_truthy_value", lambda v, default=False: bool(v))
        monkeypatch.setattr(onboarding, "is_seen", lambda cfg, flag: False)
        monkeypatch.setattr(onboarding, "mark_seen", lambda path, flag: True)
        monkeypatch.setattr(onboarding, "tool_progress_hint_gateway", lambda: "HINT_TEXT")

        # A LONG tool completes (>= threshold) — hint precondition met.
        runner.progress_callback(
            "tool.completed", "terminal", None, None, tool_call_id="c1", duration=99.0
        )

        # Timer completion still dispatched...
        assert sc.completed == [("terminal", 99.0, "c1")]
        # ...AND the one-time hint reached the progress queue (regression guard).
        assert ctx.long_tool_hint_fired == [True]
        drained = []
        while not ctx.progress_queue.empty():
            drained.append(ctx.progress_queue.get_nowait())
        assert "HINT_TEXT" in drained

    def test_hint_suppressed_when_gate_closed(self, monkeypatch):
        """Counter-case: gate closed (default/our deployment) → no hint, but the
        timer completion still dispatches. Proves the fix doesn't over-fire.
        """
        import agent.onboarding as onboarding
        import gateway.run as grun

        sc = _FakeStreamConsumer(supports_tool_timer=True)
        ctx = _make_ctx(
            tool_progress_enabled=True, tool_timer_enabled=True, sc=sc
        )
        ctx.progress_mode = "all"
        runner = TurnRunner(None, ctx)

        monkeypatch.setattr(grun, "_load_gateway_config",
                            lambda: {"display": {"tool_progress_command": False}})
        monkeypatch.setattr(onboarding, "is_seen", lambda cfg, flag: False)

        runner.progress_callback(
            "tool.completed", "terminal", None, None, tool_call_id="c1", duration=99.0
        )

        assert sc.completed == [("terminal", 99.0, "c1")]
        assert ctx.long_tool_hint_fired == [False]
        assert ctx.progress_queue.empty()


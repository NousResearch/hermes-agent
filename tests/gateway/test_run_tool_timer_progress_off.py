"""Issue C: timer lifecycle events must reach the stream consumer even when
``display.tool_progress`` is off, as long as ``extra.tool_timer_enabled`` is
true.

These tests drive ``TurnRunner.progress_callback`` directly with a fake
stream consumer, isolating the dispatch gate from the full turn machinery.
"""

from __future__ import annotations

import queue
from unittest.mock import MagicMock

from gateway.run import TurnRunner
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

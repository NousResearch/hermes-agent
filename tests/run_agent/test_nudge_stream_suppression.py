"""Regression test for the internal-nudge stream leak.

Why: When the model returns empty after tool calls, the conversation loop
appends a synthetic ``user`` nudge ("You just executed tool calls but returned
an empty response...") and ``continue``s. The model's streamed response on that
nudge-retry iteration is delivered LIVE to the client (e.g. Telegram) via the
stream callback *before* the turn's final answer is confirmed — so internal
recovery text leaks to the user.

What: ``_fire_stream_delta`` must drop deltas while the per-iteration
suppression flag ``_suppress_nudge_stream`` is set, and resume delivery once it
is cleared (when the real, no-tool-call answer is produced). Normal
(non-suppressed) streaming must be unaffected.

Test: RED before the fix (suppressed deltas still reach the callback); GREEN
after the fix (suppressed deltas are dropped, post-clear deltas delivered).
"""

from run_agent import AIAgent


def _delta_agent(sink):
    """Minimal agent carrying only what _fire_stream_delta touches."""
    agent = AIAgent.__new__(AIAgent)
    agent.stream_delta_callback = lambda t: sink.append(t)
    agent._stream_callback = None
    agent._stream_needs_break = False
    agent._stream_think_scrubber = None
    agent._stream_context_scrubber = None
    agent._current_streamed_assistant_text = ""
    agent._suppress_nudge_stream = False
    agent._record_streamed_assistant_text = lambda text: None
    return agent


def test_fire_stream_delta_suppressed_during_nudge_retry():
    sink = []
    agent = _delta_agent(sink)

    # Normal delta — delivered.
    AIAgent._fire_stream_delta(agent, "hello ")

    # Enter nudge-retry suppression: the loop sets this before `continue`.
    agent._suppress_nudge_stream = True
    AIAgent._fire_stream_delta(agent, "internal recovery noise")
    AIAgent._fire_stream_delta(agent, " more leaked tokens")

    # Real answer confirmed: the loop clears the flag, then delivers the answer.
    agent._suppress_nudge_stream = False
    AIAgent._fire_stream_delta(agent, "the real final answer")

    assert "internal recovery noise" not in sink, sink
    assert " more leaked tokens" not in sink, sink
    assert "hello " in sink
    assert "the real final answer" in sink


def test_fire_stream_delta_normal_path_unaffected():
    sink = []
    agent = _delta_agent(sink)
    AIAgent._fire_stream_delta(agent, "a")
    AIAgent._fire_stream_delta(agent, "b")
    assert sink == ["a", "b"]

"""Regression tests for the best-effort single-writer fence accessors.

The streaming paths in ``chat_completion_helpers`` and ``codex_runtime`` reach
the #65991 single-writer fence through :mod:`agent.stream_single_writer` instead
of calling ``agent._claim_stream_writer()`` directly. That indirection exists so
an agent object that doesn't expose the fence (a version-skewed checkout, a
duck-typed agent, a test double) degrades to "no fence" rather than aborting the
whole turn with ``'AIAgent' object has no attribute '_claim_stream_writer'`` —
the exact AttributeError that killed a cron job.

These tests assert the fence's *contract*: it may drop a provably superseded
stream, but it must never fence (or crash) the sole legitimate writer.
"""

import run_agent
from agent.stream_single_writer import claim_stream_writer, stream_writer_is_current




class _RaisingFenceAgent:
    """An agent whose fence methods exist but blow up when called."""

    def _claim_stream_writer(self):
        raise RuntimeError("boom")

    def _stream_writer_is_current(self, token):
        raise RuntimeError("boom")


def _real_agent():
    """A real AIAgent without running the heavy __init__ (fields self-heal)."""
    return object.__new__(run_agent.AIAgent)








def test_claim_swallows_fence_exceptions():
    assert claim_stream_writer(_RaisingFenceAgent()) == 0




def test_real_agent_fence_still_supersedes_and_preserves_sole_writer():
    agent = _real_agent()

    first = claim_stream_writer(agent)
    assert first > 0
    # Sole writer so far — still current.
    assert stream_writer_is_current(agent, first) is True

    # A newer attempt claims the sink: the older token is now superseded, the
    # newer one is current. The fence drops only the provably stale writer.
    second = claim_stream_writer(agent)
    assert second > first
    assert stream_writer_is_current(agent, first) is False
    assert stream_writer_is_current(agent, second) is True

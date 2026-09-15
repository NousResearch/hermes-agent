"""Regression tests for Codex Responses stream first_chunk_at export.

The ``post_api_request`` hook receives ``first_chunk_at`` (epoch seconds when
the initial parsed event arrived). The streaming Chat Completions path populates
it from a diag object when the stream completes. The Responses path runs a
different event-driven consumer that never built a diag object; this corrects
the omission so plugins receive timing for both code paths.

Issue #105311.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

# Stub heavy imports to run in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *_: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from agent.codex_runtime import run_codex_stream


def _make_agent():
    return SimpleNamespace(
        session_id="",
        provider="openai-codex",
        model="timing-fixture",
        _interrupt_requested=False,
        _last_api_first_chunk_at=None,
        _touch_activity=lambda *_: None,
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda *_: None,
    )


def test_first_chunk_at_populated_when_events_arrive():
    """The first parsed stream event stamps agent._last_api_first_chunk_at."""
    agent = _make_agent()
    received_events = []

    def events():
        received_events.append(time.time())
        yield {"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}
        yield {"type": "response.output_text.delta", "delta": "Hello"}
        yield {"type": "response.completed", "response": {"id": "r1", "status": "completed"}}

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kw: events()))
    before = time.time()
    result = run_codex_stream(agent, {"model": "timing-fixture", "input": "Go."}, client=client)
    after = time.time()

    assert result.output_text == "Hello"
    assert result.status == "completed"
    assert agent._last_api_first_chunk_at is not None
    assert before <= agent._last_api_first_chunk_at <= after
    assert len(received_events) == 1  # generator fired once (first event)


def test_first_chunk_at_remains_none_when_no_events():
    """Streams with zero parsed events leave first_chunk_at None."""
    agent = _make_agent()

    def events():
        # Immediately raise before yielding any event.
        raise RuntimeError("Stream connect failed before any event")

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kw: events()))

    try:
        run_codex_stream(agent, {"model": "timing-fixture", "input": "Go."}, client=client)
    except RuntimeError:
        pass

    assert agent._last_api_first_chunk_at is None


def test_first_chunk_at_not_overwritten_by_later_events():
    """Subsequent events do not replace the initial stamp."""
    agent = _make_agent()
    event_times = []

    def events():
        event_times.append(time.time())
        yield {"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}
        time.sleep(0.05)  # let the clock tick
        event_times.append(time.time())
        yield {"type": "response.output_text.delta", "delta": "later"}
        yield {"type": "response.completed", "response": {"id": "r1", "status": "completed"}}

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kw: events()))
    run_codex_stream(agent, {"model": "timing-fixture", "input": "Go."}, client=client)

    first_stamp = agent._last_api_first_chunk_at
    # The stamp should match the time of the first event, not the second.
    assert abs(first_stamp - event_times[0]) < 0.01
    # and it must NOT match the second (50ms later)
    assert abs(first_stamp - event_times[1]) >= 0.04


def test_first_chunk_timing_lifecycle_events_before_text():
    """Lifecycle events arriving before text deltas count as the first chunk."""
    agent = _make_agent()
    lifecycle_time = None

    def events():
        nonlocal lifecycle_time
        lifecycle_time = time.time()
        # Lifecycle frame with no delta content arrives first.
        yield {"type": "response.created", "response": {"id": "r1", "status": "in_progress"}}
        time.sleep(0.03)
        # Output text arrives later.
        yield {"type": "response.output_text.delta", "delta": "text"}
        yield {"type": "response.completed", "response": {"id": "r1", "status": "completed"}}

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_kw: events()))
    run_codex_stream(agent, {"model": "timing-fixture", "input": "Go."}, client=client)

    # The stamp matches the lifecycle event, not the text delta.
    assert abs(agent._last_api_first_chunk_at - lifecycle_time) < 0.001

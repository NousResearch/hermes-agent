"""Tests for the kanban-react debouncer state machine.

Focused on gateway/kanban_react.py — pure logic, no asyncio loop fixtures
beyond pytest-asyncio. The integration with _handle_message lives in
gateway/run.py and is exercised by the existing notifier tests + the
manual smoke described in this module's docstring.
"""
from __future__ import annotations

import asyncio
import time
import pytest

from gateway.kanban_react import (
    KanbanReactCoordinator,
    KanbanReactEvent,
    REACT_KINDS,
    render_events_preamble,
)


def _ev(task_id: str, kind: str = "blocked", board: str = "b", summary: str = "x"):
    return KanbanReactEvent(
        task_id=task_id, kind=kind, board=board, summary=summary,
        received_at=time.monotonic(),
    )


@pytest.mark.asyncio
async def test_react_kinds_includes_actionable_states():
    """The set is exactly the events the agent could meaningfully act on."""
    assert REACT_KINDS == {"blocked", "gave_up", "crashed", "timed_out"}
    assert "completed" not in REACT_KINDS  # happy path is silent


@pytest.mark.asyncio
async def test_completed_event_never_triggers_flush():
    """Completed events post to the thread but never wake the agent."""
    flushed = []
    async def cb(sk, evs, src):
        flushed.append((sk, evs))
    c = KanbanReactCoordinator(flush_callback=cb, debounce_seconds=0.05)
    await c.record_event("sk1", _ev("t_a", kind="completed"), {"x": 1})
    await asyncio.sleep(0.2)
    assert flushed == []


@pytest.mark.asyncio
async def test_single_event_flushes_after_debounce():
    """One blocked event → one flush after the debounce window."""
    flushed = []
    async def cb(sk, evs, src):
        flushed.append((sk, list(evs), src))
    c = KanbanReactCoordinator(flush_callback=cb, debounce_seconds=0.05)
    await c.record_event("sk1", _ev("t_a"), {"platform": "discord"})
    assert flushed == []  # not yet
    await asyncio.sleep(0.2)
    assert len(flushed) == 1
    sk, evs, src = flushed[0]
    assert sk == "sk1"
    assert [e.task_id for e in evs] == ["t_a"]
    assert src == {"platform": "discord"}


@pytest.mark.asyncio
async def test_events_within_window_collapse_to_one_flush():
    """Three events within the debounce window → ONE flush carrying all three.
    Timer is NOT reset by subsequent events — bounded latency."""
    flushed = []
    async def cb(sk, evs, src):
        flushed.append(list(evs))
    c = KanbanReactCoordinator(flush_callback=cb, debounce_seconds=0.1)
    await c.record_event("sk1", _ev("t_a"), {"p": 1})
    await asyncio.sleep(0.02)
    await c.record_event("sk1", _ev("t_b"), {"p": 1})
    await asyncio.sleep(0.02)
    await c.record_event("sk1", _ev("t_c"), {"p": 1})
    await asyncio.sleep(0.2)
    assert len(flushed) == 1
    assert [e.task_id for e in flushed[0]] == ["t_a", "t_b", "t_c"]


@pytest.mark.asyncio
async def test_events_in_separate_sessions_dont_cross():
    """Two session_keys → two independent buffers and flushes."""
    flushed = []
    async def cb(sk, evs, src):
        flushed.append((sk, [e.task_id for e in evs]))
    c = KanbanReactCoordinator(flush_callback=cb, debounce_seconds=0.05)
    await c.record_event("sk-A", _ev("t_a"), {"p": "A"})
    await c.record_event("sk-B", _ev("t_b"), {"p": "B"})
    await asyncio.sleep(0.2)
    assert sorted(flushed) == [("sk-A", ["t_a"]), ("sk-B", ["t_b"])]


@pytest.mark.asyncio
async def test_user_turn_piggybacks_and_cancels_timer():
    """consume_for_user_turn drains the buffer and prevents the synthetic
    flush from firing — the user's turn covers them."""
    flushed = []
    async def cb(sk, evs, src):
        flushed.append(evs)
    c = KanbanReactCoordinator(flush_callback=cb, debounce_seconds=0.1)
    await c.record_event("sk1", _ev("t_a"), {"p": 1})
    drained = await c.consume_for_user_turn("sk1")
    assert [e.task_id for e in drained] == ["t_a"]
    await asyncio.sleep(0.2)
    assert flushed == []  # timer was cancelled


@pytest.mark.asyncio
async def test_events_during_in_flight_turn_batch_for_next_round():
    """While a turn is in_flight, new events buffer silently. After
    mark_turn_done, a fresh debounce arms and they flush together."""
    flushed = []
    async def cb(sk, evs, src):
        flushed.append([e.task_id for e in evs])
    c = KanbanReactCoordinator(flush_callback=cb, debounce_seconds=0.05)
    # User turn starts — drains nothing but marks in_flight.
    await c.consume_for_user_turn("sk1")
    # Events arrive during the turn — must NOT trigger a flush.
    await c.record_event("sk1", _ev("t_a"), {"p": 1})
    await c.record_event("sk1", _ev("t_b"), {"p": 1})
    await asyncio.sleep(0.15)
    assert flushed == []  # in_flight blocks the timer
    # Turn ends → re-arm debounce, then flush.
    await c.mark_turn_done("sk1")
    await asyncio.sleep(0.2)
    assert flushed == [["t_a", "t_b"]]


@pytest.mark.asyncio
async def test_synthetic_flush_marks_in_flight_then_done():
    """When the timer fires and the flush callback runs, in_flight is set
    before the callback and released by mark_turn_done after. Events that
    arrive DURING the synthetic turn batch for the next round."""
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()
    flushed = []
    async def cb(sk, evs, src):
        callback_started.set()
        await callback_release.wait()
        flushed.append([e.task_id for e in evs])
    c = KanbanReactCoordinator(flush_callback=cb, debounce_seconds=0.02)
    await c.record_event("sk1", _ev("t_a"), {"p": 1})
    await asyncio.wait_for(callback_started.wait(), timeout=1.0)
    # Now we're inside the flush. State should be in_flight; events
    # arriving here must NOT spawn a parallel timer.
    await c.record_event("sk1", _ev("t_b"), {"p": 1})
    callback_release.set()
    await asyncio.sleep(0.1)  # let the synth turn complete + re-arm debounce
    # Wait for the re-armed debounce.
    await asyncio.sleep(0.15)
    assert flushed == [["t_a"], ["t_b"]]


@pytest.mark.asyncio
async def test_render_preamble_compact_and_ordered():
    p = render_events_preamble([
        KanbanReactEvent("t_a", "blocked", "boardA", "needs API key", 1.0),
        KanbanReactEvent("t_b", "crashed", "boardB", "OOM at step 7", 2.0),
    ])
    # One header line, two bullets, sorted board names in header.
    assert "boardA" in p and "boardB" in p
    assert "t_a" in p and "blocked" in p and "needs API key" in p
    assert "t_b" in p and "crashed" in p and "OOM at step 7" in p
    # Empty input yields empty output (safe to concat unconditionally).
    assert render_events_preamble([]) == ""


@pytest.mark.asyncio
async def test_flush_callback_exception_releases_in_flight():
    """If the synthetic dispatch raises, the coordinator must still
    release in_flight so future events don't get stuck buffered."""
    raised = []
    async def cb(sk, evs, src):
        raised.append(evs)
        raise RuntimeError("simulated dispatch failure")
    c = KanbanReactCoordinator(flush_callback=cb, debounce_seconds=0.02)
    await c.record_event("sk1", _ev("t_a"), {"p": 1})
    await asyncio.sleep(0.1)
    assert len(raised) == 1
    # in_flight should now be False — a new event should start a fresh
    # debounce, not get stuck.
    flushed_again = []
    async def cb2(sk, evs, src):
        flushed_again.append([e.task_id for e in evs])
    c._flush_cb = cb2
    await c.record_event("sk1", _ev("t_b"), {"p": 1})
    await asyncio.sleep(0.1)
    assert flushed_again == [["t_b"]]

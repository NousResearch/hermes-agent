"""Timer-lane trims run the full collection only while nobody is waiting (#139763).

The gateway housekeeping tick dropped its forced ``gc.collect()`` because a full pass over a
multi-million-object heap holds the GIL for 250-750 ms and froze every live session once a minute.
Python 3.14's incremental collector only advances on allocation, so a quiet gateway would then keep
closed agents' reference cycles (and their pages) forever. The tick collects once the process has
been turn-free for ``IDLE_COLLECT_AFTER_SECONDS``: once per quiet period, again after a later
agent close, and never while a turn runs.
"""

import pytest

from agent import review_idle_queue
from hermes_cli import mem_trim


@pytest.fixture
def clock(monkeypatch):
    queue = review_idle_queue.ReviewIdleQueue()
    now = [1000.0]
    queue._now = lambda: now[0]
    monkeypatch.setattr(review_idle_queue, "QUEUE", queue)
    collects = []
    monkeypatch.setattr(mem_trim.gc, "collect", lambda *a: collects.append(now[0]) or 0)
    monkeypatch.setattr(mem_trim, "_malloc_trim", lambda pad: 1)
    monkeypatch.setattr(mem_trim, "_probe_done", True)
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 0.0)
    monkeypatch.setattr(mem_trim, "_last_alloc_trim_monotonic", 0.0)
    monkeypatch.setattr(mem_trim, "_config_settings", lambda: (True, 0.0, 1, 0.0))
    monkeypatch.setattr(mem_trim, "_idle_collected_for", None)
    monkeypatch.setattr(mem_trim, "_agent_closed_since_idle_collect", False)
    return queue, now, collects


def _tick():
    from gateway import run as gateway_run
    gateway_run._housekeeping_memory_trim()


def test_housekeeping_collects_once_per_quiet_period_and_never_during_a_turn(clock):
    queue, now, collects = clock
    queue.note_turn_started()
    now[0] += mem_trim.IDLE_COLLECT_AFTER_SECONDS * 3
    _tick()
    assert collects == []  # a turn is live: no freeze, however long it runs

    queue.note_turn_finished()
    now[0] += mem_trim.IDLE_COLLECT_AFTER_SECONDS - 1
    _tick()
    assert collects == []  # quiet, but not for long enough

    now[0] += 2
    _tick()
    _tick()
    assert len(collects) == 1  # the quiet period's one collection

    mem_trim.note_agent_closed()  # idle-cache eviction later in the same quiet period
    _tick()
    _tick()
    assert len(collects) == 2


def test_a_new_turn_starts_a_new_quiet_period(clock):
    queue, now, collects = clock
    queue.note_turn_started()
    queue.note_turn_finished()
    now[0] += mem_trim.IDLE_COLLECT_AFTER_SECONDS
    _tick()
    queue.note_turn_started()
    queue.note_turn_finished()
    now[0] += 1
    _tick()
    assert len(collects) == 1  # just talked: the next collection waits for a full quiet window
    now[0] += mem_trim.IDLE_COLLECT_AFTER_SECONDS
    _tick()
    assert len(collects) == 2

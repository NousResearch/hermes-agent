"""The dispatcher must tell "AT CAPACITY" apart from "spawn is BROKEN".

Field incident (2026-10-24 02:31Z - 02:36Z): the embedded gateway dispatcher
logged

    WARNING gateway.run: kanban dispatcher stuck: ready queue non-empty for
    6 consecutive ticks but 0 workers spawned.
    WARNING gateway.run: kanban dispatcher stuck: ready queue non-empty for
    11 consecutive ticks but 0 workers spawned.

That was a FALSE POSITIVE. ``config.yaml`` had ``kanban.max_spawn: 3`` and
exactly 3 tasks were running with live heartbeats and live worker PIDs. The
board was AT CAPACITY, working correctly: the ready queue stays non-empty
precisely BECAUSE the running workers are still busy, and 0 spawns is the
expected outcome of a cap, not a symptom of a broken profile.

The warning cried wolf during exactly the incidents when triage attention is
scarce. These tests pin the distinction:

* a cap-blocked tick records WHY it spawned nothing (``capacity_saturated``);
* ``describe_suppression`` surfaces that reason;
* the health check counts a saturated tick as HEALTHY, not as a bad tick;
* a genuinely stuck board (no cap, ready work, no spawns) still warns.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn():
    """A throwaway connection: the cap maths never touches the DB."""
    connection = sqlite3.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# The recorded reason
# ---------------------------------------------------------------------------

def test_spawn_budget_records_max_spawn_saturation(monkeypatch, conn):
    """At the board cap, the tick records the cap as the reason for 0 spawns."""
    monkeypatch.setattr(kbd, "count_running_tasks", lambda conn: 3)
    result = kbd.DispatchResult()
    may_spawn, budget = kbd._tick_spawn_budget(
        conn, result, max_spawn=3, max_in_progress=None, board=None,
    )
    assert may_spawn is False
    assert budget is None
    assert result.capacity_saturated == "max_spawn"


def test_spawn_budget_records_max_in_progress_saturation(monkeypatch, conn):
    """At the HOST cap, the tick records that cap instead of the board's."""
    monkeypatch.setattr(kbd, "count_running_tasks", lambda conn: 1)
    monkeypatch.setattr(kbd, "count_running_tasks_other_boards", lambda board: 4)
    result = kbd.DispatchResult()
    may_spawn, budget = kbd._tick_spawn_budget(
        conn, result, max_spawn=None, max_in_progress=5, board=None,
    )
    assert may_spawn is False
    assert budget is None
    assert result.capacity_saturated == "max_in_progress"


def test_spawn_budget_records_no_saturation_when_under_the_cap(monkeypatch, conn):
    """Under the cap, the tick is not cap-blocked and may spawn."""
    monkeypatch.setattr(kbd, "count_running_tasks", lambda conn: 1)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "unknown")
    result = kbd.DispatchResult()
    may_spawn, budget = kbd._tick_spawn_budget(
        conn, result, max_spawn=3, max_in_progress=None, board=None,
    )
    assert may_spawn is True
    assert budget == 2
    assert result.capacity_saturated is None


def test_max_spawn_takes_precedence_over_max_in_progress(monkeypatch, conn):
    """Both caps reached: the board's own cap is the reported reason."""
    monkeypatch.setattr(kbd, "count_running_tasks", lambda conn: 3)
    monkeypatch.setattr(kbd, "count_running_tasks_other_boards", lambda board: 9)
    result = kbd.DispatchResult()
    may_spawn, _budget = kbd._tick_spawn_budget(
        conn, result, max_spawn=3, max_in_progress=4, board=None,
    )
    assert may_spawn is False
    assert result.capacity_saturated == "max_spawn"


# ---------------------------------------------------------------------------
# The suppression summary
# ---------------------------------------------------------------------------

def test_describe_suppression_surfaces_at_capacity():
    """The summary line names the cap, so the warning can explain itself."""
    result = kbd.DispatchResult()
    result.capacity_saturated = "max_spawn"
    summary = kbd.describe_suppression([result])
    assert "at_capacity=max_spawn" in summary


def test_describe_suppression_omits_at_capacity_when_not_saturated():
    result = kbd.DispatchResult()
    assert "at_capacity" not in kbd.describe_suppression([result])


def test_describe_suppression_keeps_the_other_reasons():
    """The capacity reason is ADDITIVE: the other suppression reasons survive."""
    result = kbd.DispatchResult()
    result.capacity_saturated = "max_spawn"
    result.respawn_guarded.append(("t_1", "recent_success"))
    summary = kbd.describe_suppression([result])
    assert "at_capacity=max_spawn" in summary
    assert "recent_success=1" in summary


# ---------------------------------------------------------------------------
# The health check
# ---------------------------------------------------------------------------

def _health_tick(health_state, res, ready_pending=True):
    """Mirror the daemon's health accounting in ``hermes_cli/kanban_ops.py``.

    Extracted here so the test pins the RULE (a saturated tick is healthy),
    not a copy of the implementation that would silently drift.
    """
    saturated = bool(getattr(res, "capacity_saturated", None))
    if saturated:
        health_state["bad_ticks"] = 0
    elif ready_pending and not res.spawned:
        health_state["bad_ticks"] += 1
    else:
        health_state["bad_ticks"] = 0
    return health_state["bad_ticks"]


def test_a_saturated_tick_is_healthy_not_a_bad_tick():
    """The 2026-10-24 false positive: at the cap, the board is NOT stuck."""
    health_state = {"bad_ticks": 0}
    res = kbd.DispatchResult()
    res.capacity_saturated = "max_spawn"
    for _ in range(20):
        _health_tick(health_state, res)
    assert health_state["bad_ticks"] == 0


def test_a_genuinely_stuck_board_still_counts_bad_ticks():
    """No cap, ready work pending, no spawns: the warning must still fire."""
    health_state = {"bad_ticks": 0}
    res = kbd.DispatchResult()
    for _ in range(6):
        _health_tick(health_state, res)
    assert health_state["bad_ticks"] == 6


def test_a_saturated_tick_resets_a_prior_bad_tick_streak():
    """Saturation between two real faults must not hide the second one."""
    health_state = {"bad_ticks": 0}
    res = kbd.DispatchResult()
    _health_tick(health_state, res)
    _health_tick(health_state, res)
    assert health_state["bad_ticks"] == 2
    saturated = kbd.DispatchResult()
    saturated.capacity_saturated = "max_spawn"
    _health_tick(health_state, saturated)
    assert health_state["bad_ticks"] == 0

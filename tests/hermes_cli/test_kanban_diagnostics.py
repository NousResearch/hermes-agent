"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------


def test_hallucinated_cards_fires_on_blocked_event():
    task = _task(status="ready")
    events = [
        _event("created", ts=100),
        _event("completion_blocked_hallucination", ts=200,
               phantom_cards=["t_bad1", "t_bad2"],
               verified_cards=["t_good1"]),
    ]
    diags = kd.compute_task_diagnostics(task, events, [])
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "hallucinated_cards"
    assert d.severity == "error"
    assert d.data["phantom_ids"] == ["t_bad1", "t_bad2"]
    # Generic recovery actions always available; comment action too.
    kinds = [a.kind for a in d.actions]
    assert "comment" in kinds
    assert "reassign" in kinds


def test_hallucinated_cards_clears_on_subsequent_completion():
    task = _task(status="done")
    events = [
        _event("completion_blocked_hallucination", ts=100, phantom_cards=["t_x"]),
        _event("completed", ts=200, summary="retry worked"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [])
    assert diags == []


def test_prose_phantom_refs_fires_after_clean_completion():
    # Prose scan emits its event AFTER the completed event in the DB
    # path, but a subsequent clean completion clears it. Phantom id
    # must be valid hex — the scanner regex is ``t_[a-f0-9]{8,}``.
    task = _task(status="done")
    events = [
        _event("completed", ts=100, summary="referenced t_bad", result_len=0),
        _event("suspected_hallucinated_references", ts=101,
               phantom_refs=["t_deadbeef99"], source="completion_summary"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [])
    assert len(diags) == 1
    assert diags[0].kind == "prose_phantom_refs"
    assert diags[0].severity == "warning"
    assert diags[0].data["phantom_refs"] == ["t_deadbeef99"]


def test_prose_phantom_refs_clears_on_later_clean_edit():
    task = _task(status="done")
    events = [
        _event("completed", ts=100, summary="bad"),
        _event("suspected_hallucinated_references", ts=101,
               phantom_refs=["t_ffff0000cc"]),
        _event("edited", ts=200, fields=["result", "summary"]),
    ]
    diags = kd.compute_task_diagnostics(task, events, [])
    assert diags == []


def test_repeated_failures_fires_at_threshold_on_spawn():
    """A task with multiple spawn_failed runs gets a spawn-flavoured
    diagnostic (title mentions 'spawn', suggested action is ``doctor``).
    """
    task = _task(status="ready", consecutive_failures=3,
                 last_failure_error="Profile 'debugger' does not exist")
    runs = [
        _run(outcome="spawn_failed", run_id=1),
        _run(outcome="spawn_failed", run_id=2),
        _run(outcome="spawn_failed", run_id=3),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "repeated_failures"
    assert d.severity == "error"
    # CLI hints are what operators actually need here.
    suggested = [a.label for a in d.actions if a.suggested]
    assert any("doctor" in s for s in suggested)


def test_repeated_failures_fires_on_timeout_loop():
    """The rule surfaces for timeout loops too — that's the point of
    unifying the counter. Suggested action is 'check logs', not
    'fix profile'."""
    task = _task(status="ready", consecutive_failures=3,
                 last_failure_error="elapsed 600s > limit 300s")
    runs = [
        _run(outcome="timed_out", run_id=1),
        _run(outcome="timed_out", run_id=2),
        _run(outcome="timed_out", run_id=3),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "repeated_failures"
    assert d.data["most_recent_outcome"] == "timed_out"
    suggested = [a.label for a in d.actions if a.suggested]
    assert any("log" in s.lower() for s in suggested)


def test_repeated_failures_escalates_to_critical():
    task = _task(consecutive_failures=6, last_failure_error="boom")
    diags = kd.compute_task_diagnostics(task, [], [])
    assert diags[0].severity == "critical"


def test_repeated_failures_below_threshold_silent():
    task = _task(consecutive_failures=2)
    assert kd.compute_task_diagnostics(task, [], []) == []


def test_repeated_crashes_counts_trailing_streak_only():
    task = _task(status="ready", assignee="crashy")
    runs = [
        _run(outcome="completed", run_id=1),
        _run(outcome="crashed", run_id=2, error="OOM"),
        _run(outcome="crashed", run_id=3, error="OOM again"),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "repeated_crashes"
    # 2 consecutive crashes at the end → default threshold 2 → error severity.
    assert d.severity == "error"
    assert d.data["consecutive_crashes"] == 2


def test_repeated_crashes_breaks_on_recent_success():
    task = _task(status="ready", assignee="fixed")
    runs = [
        _run(outcome="crashed", run_id=1),
        _run(outcome="crashed", run_id=2),
        _run(outcome="completed", run_id=3),
    ]
    assert kd.compute_task_diagnostics(task, [], runs) == []


def test_repeated_crashes_escalates_on_many_crashes():
    task = _task(status="ready", assignee="x")
    runs = [_run(outcome="crashed", run_id=i) for i in range(1, 6)]  # 5 in a row
    diags = kd.compute_task_diagnostics(task, [], runs)
    assert diags[0].severity == "critical"


def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48


def test_stuck_in_blocked_silent_with_recent_comment():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48),
        _event("commented", ts=now - 3600 * 2, author="human"),
    ]
    assert kd.compute_task_diagnostics(task, events, [], now=now) == []


def test_stuck_in_blocked_silent_when_not_blocked():
    task = _task(status="ready")
    events = [_event("blocked", ts=1000)]
    assert kd.compute_task_diagnostics(task, events, [], now=9999999) == []


def test_repeated_crashes_surfaces_actual_error_in_title():
    """The title should lead with the actual error text so operators
    see WHAT broke (e.g. rate-limit, auth, OOM) without opening logs.
    """
    task = _task(status="ready", assignee="x")
    runs = [
        _run(outcome="crashed", run_id=1, error="openai: 429 Too Many Requests"),
        _run(outcome="crashed", run_id=2, error="openai: 429 Too Many Requests"),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    assert len(diags) == 1
    d = diags[0]
    assert "429" in d.title
    assert "Too Many Requests" in d.title
    # Full error in detail.
    assert "429 Too Many Requests" in d.detail


def test_repeated_crashes_no_error_fallback_title():
    task = _task(status="ready", assignee="x")
    runs = [
        _run(outcome="crashed", run_id=1, error=None),
        _run(outcome="crashed", run_id=2, error=None),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    assert "no error recorded" in diags[0].title


def test_repeated_failures_surfaces_actual_error_in_title():
    task = _task(consecutive_failures=5,
                 last_failure_error="insufficient_quota: billing limit reached")
    diags = kd.compute_task_diagnostics(task, [], [])
    assert len(diags) == 1
    d = diags[0]
    assert "insufficient_quota" in d.title or "billing limit" in d.title
    assert "insufficient_quota" in d.detail


def test_repeated_crashes_truncates_huge_tracebacks():
    """Full Python tracebacks can be tens of KB. The title stays one
    line (≤160 chars); the detail caps at 500 chars + ellipsis so the
    card doesn't explode visually."""
    huge = "Traceback (most recent call last):\n" + ("  File\n" * 500)
    task = _task(status="ready")
    runs = [
        _run(outcome="crashed", run_id=1, error=huge),
        _run(outcome="crashed", run_id=2, error=huge),
    ]
    diags = kd.compute_task_diagnostics(task, [], runs)
    d = diags[0]
    # Title only the first line, capped.
    assert "\n" not in d.title
    assert len(d.title) < 250
    # Detail contains the snippet with ellipsis.
    assert d.detail.endswith("…") or len(d.detail) < 700


# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------


def test_diagnostics_sorted_critical_first():
    """A task with both a critical (many spawn failures) and a warning
    (prose phantoms) diagnostic should list the critical one first."""
    task = _task(status="done", consecutive_failures=10,
                 last_failure_error="nope")
    events = [
        _event("completed", ts=100, summary="referenced t_missing"),
        _event("suspected_hallucinated_references", ts=101,
               phantom_refs=["t_missing11"]),
    ]
    diags = kd.compute_task_diagnostics(task, events, [])
    kinds = [d.kind for d in diags]
    assert kinds[0] == "repeated_failures"  # critical
    assert "prose_phantom_refs" in kinds


# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------


def test_broken_rule_is_isolated(monkeypatch):
    def _bad_rule(task, events, runs, now, cfg):
        raise RuntimeError("synthetic rule bug")

    # Insert a broken rule at the front of the registry; subsequent
    # rules should still run and produce their diagnostics.
    monkeypatch.setattr(kd, "_RULES", [_bad_rule] + kd._RULES)

    task = _task(consecutive_failures=5, last_failure_error="e")
    diags = kd.compute_task_diagnostics(task, [], [])
    # The broken rule silently drops, the real one still fires.
    kinds = [d.kind for d in diags]
    assert "repeated_failures" in kinds

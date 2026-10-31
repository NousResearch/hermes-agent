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
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc
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
















def test_running_with_open_parents_fires_only_while_running():
    """A running card whose parent is not terminal is flagged; the same graph
    on a ready/todo card (the gate is holding it) and a done parent are not."""
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    diags = kd.compute_task_diagnostics(_task(status="running", started_at=100), [], [], graph=graph)
    assert [d.kind for d in diags] == ["running_with_open_parents"]
    assert diags[0].data["open_parents"] == [{"id": "t_parent", "status": "todo"}]
    assert "hermes kanban unlink t_parent t_demo00" in diags[0].actions[0].payload["command"]
    assert kd.compute_task_diagnostics(_task(status="todo"), [], [], graph=graph) == []
    done_graph = {"parents": [{"id": "t_parent", "title": "p", "status": "done"}], "children": []}
    assert kd.compute_task_diagnostics(_task(status="running"), [], [], graph=done_graph) == []


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








# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kbc.connect()
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




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"


def test_stranded_in_ready_reports_recorded_guard_without_reassign_advice():
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 45 * 60),
        _event("respawn_guarded", ts=now - 60, reason="active_pr"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = next(d for d in diags if d.kind == "stranded_in_ready")
    assert stranded.severity == "warning"
    assert stranded.data["guard_reason"] == "active_pr"
    assert stranded.data["guard_recorded_at"] == now - 60
    assert any(action.kind == "reassign" for action in stranded.actions)


def test_fresh_ready_task_with_recorded_guard_is_not_called_stranded():
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 5 * 60),
        _event("respawn_guarded", ts=now - 60, reason="recent_success"),
    ]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    assert not any(d.kind == "stranded_in_ready" for d in diags)


def test_expired_guard_keeps_stranded_diagnostic_visible():
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 60 * 60),
        _event("respawn_guarded", ts=now - 45 * 60, reason="recent_success"),
    ]
    diagnostics = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = next(d for d in diagnostics if d.kind == "stranded_in_ready")
    assert stranded.severity == "error"
    assert stranded.data["guard_reason"] == "recent_success"
    assert any(action.kind == "reassign" for action in stranded.actions)


def test_guard_does_not_hide_stranded_after_requeue_event():
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [
        _event("created", ts=now - 60 * 60),
        _event("respawn_guarded", ts=now - 50 * 60, reason="active_pr"),
        _event("unblocked", ts=now - 45 * 60),
    ]
    diagnostics = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = next(d for d in diagnostics if d.kind == "stranded_in_ready")
    assert stranded.severity == "warning"
    assert stranded.data["ready_since"] == now - 45 * 60
    assert any(action.kind == "reassign" for action in stranded.actions)


@pytest.mark.parametrize("guard_signal", ["recent_success", "active_pr"])
def test_review_spawnability_preserves_review_handoff_inputs(
    kanban_home, monkeypatch, guard_signal,
):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="review", assignee="reviewer")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
        monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
        if guard_signal == "recent_success":
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'reviewer', 'completed', 'completed', ?, ?)",
                (task_id, int(time.time()) - 1, int(time.time())),
            )
        else:
            kb.add_comment(
                conn, task_id, "operator",
                "PR: https://github.com/example/repo/pull/1",
            )

        assert kbd.has_spawnable_review(conn)
    finally:
        conn.close()




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True

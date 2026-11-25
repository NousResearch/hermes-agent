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


def _kinds(diags):
    return [(d.kind, d.severity) for d in diags]


def test_stuck_in_blocked_one_early_comment_no_longer_hides_it_forever():
    """#154122: a comment minutes after the park used to exempt the card permanently."""
    now = int(time.time())
    blocked_at = now - 86400 * 30
    events = [
        {"kind": "blocked", "created_at": blocked_at,
         "payload": {"reason": "need the staging DB password rotated", "kind": "needs_input"}},
        _event("commented", ts=blocked_at + 600, author="ops"),
    ]
    diags = kd.compute_task_diagnostics(_task(status="blocked"), events, [], now=now)
    assert _kinds(diags) == [("stuck_in_blocked", "critical")]
    d = diags[0]
    # The concrete cause travels with the signal, not a paraphrase.
    assert "need the staging DB password rotated" in d.detail and "(needs_input)" in d.detail
    assert d.data["reason"] == "need the staging DB password rotated"
    assert d.data["last_comment_at"] == blocked_at + 600


def test_stuck_in_blocked_comment_rearms_the_warning_timer():
    now = int(time.time())
    blocked_at = now - 3600 * 50
    fresh = [_event("blocked", ts=blocked_at, reason="r"), _event("commented", ts=now - 3600 * 2, author="me")]
    assert kd.compute_task_diagnostics(_task(status="blocked"), fresh, [], now=now) == []
    stale = [_event("blocked", ts=blocked_at, reason="r"), _event("commented", ts=now - 3600 * 30, author="me")]
    diags = kd.compute_task_diagnostics(_task(status="blocked"), stale, [], now=now)
    assert _kinds(diags) == [("stuck_in_blocked", "warning")]
    assert diags[0].data["quiet_hours"] >= 30


@pytest.mark.parametrize("days, severity", [(6, "warning"), (7, "error"), (29, "error"), (30, "critical"),
                                            (114, "critical")])
def test_stuck_in_blocked_total_age_ladder_ignores_comments(days, severity):
    now = int(time.time())
    blocked_at = now - 86400 * days
    # A comment an hour ago would hold off the warning, but never the age escalation.
    events = [_event("blocked", ts=blocked_at, reason="r"), _event("commented", ts=now - 3600, author="me")]
    diags = kd.compute_task_diagnostics(_task(status="blocked"), events, [], now=now)
    assert _kinds(diags) == ([] if severity == "warning" else [("stuck_in_blocked", severity)])


def test_stuck_in_blocked_ignores_configured_automation_comments():
    now = int(time.time())
    events = [_event("blocked", ts=now - 3600 * 30, reason="r"), _event("commented", ts=now - 60, author="pm-bot")]
    assert kd.compute_task_diagnostics(_task(status="blocked"), events, [], now=now) == []
    diags = kd.compute_task_diagnostics(_task(status="blocked"), events, [], now=now,
                                        config={"blocked_ignore_comment_authors": ["pm-bot"]})
    assert _kinds(diags) == [("stuck_in_blocked", "warning")]


def test_stuck_in_blocked_includes_last_failure_error():
    now = int(time.time())
    task = _task(status="blocked", last_failure_error="pytest exited 4: no tests ran in tests/api")
    diags = kd.compute_task_diagnostics(task, [_event("blocked", ts=now - 3600 * 30, reason="CI red")], [], now=now)
    assert "pytest exited 4" in diags[0].detail
    assert diags[0].data["last_failure_error"].startswith("pytest exited 4")


def test_stuck_in_blocked_custom_ladder():
    now = int(time.time())
    events = [_event("blocked", ts=now - 86400 * 3, reason="r"), _event("commented", ts=now - 60, author="me")]
    diags = kd.compute_task_diagnostics(_task(status="blocked"), events, [], now=now,
                                        config={"stale_error_days": 2, "stale_critical_days": 3})
    assert _kinds(diags) == [("stuck_in_blocked", "critical")]


def test_stuck_in_todo_fires_and_names_open_parents():
    now = int(time.time())
    task = _task(id="t_child", status="todo", created_at=now - 86400 * 40)
    graph = {"parents": [{"id": "t_parent", "status": "blocked"}, {"id": "t_done", "status": "done"}]}
    diags = kd.compute_task_diagnostics(task, [_event("created", ts=now - 86400 * 40)], [], now=now, graph=graph)
    assert _kinds(diags) == [("stuck_in_todo", "critical")]
    d = diags[0]
    assert d.data["open_parents"] == [{"id": "t_parent", "status": "blocked"}]
    assert "t_parent (blocked)" in d.detail
    assert any("hermes kanban show t_parent" in (a.payload or {}).get("command", "") for a in d.actions)


@pytest.mark.parametrize("hours, expected", [(10, []), (80, [("stuck_in_todo", "warning")]),
                                             (24 * 8, [("stuck_in_todo", "error")])])
def test_stuck_in_todo_thresholds(hours, expected):
    now = int(time.time())
    task = _task(status="todo", created_at=now - 3600 * hours)
    diags = kd.compute_task_diagnostics(task, [_event("created", ts=now - 3600 * hours)], [], now=now, graph={"parents": []})
    assert _kinds(diags) == expected


def test_stuck_in_todo_measures_from_latest_entry():
    now = int(time.time())
    task = _task(status="todo", created_at=now - 86400 * 60)
    events = [_event("created", ts=now - 86400 * 60), _event("promoted", ts=now - 86400 * 59),
              _event("dependency_wait", ts=now - 3600, reason="parent_reopened")]
    assert kd.compute_task_diagnostics(task, events, [], now=now) == []


def test_stuck_in_todo_without_graph_does_not_claim_nothing_holds_it():
    now = int(time.time())
    task = _task(status="todo", created_at=now - 86400 * 10)
    diags = kd.compute_task_diagnostics(task, [], [], now=now)
    assert _kinds(diags) == [("stuck_in_todo", "error")]
    assert "No open parent" not in diags[0].detail and diags[0].data["open_parents"] is None


def test_stuck_in_todo_ignores_other_statuses():
    now = int(time.time())
    for status in ("ready", "triage", "done", "archived", "blocked"):
        task = _task(status=status, created_at=now - 86400 * 90)
        assert not [d for d in kd.compute_task_diagnostics(task, [], [], now=now) if d.kind == "stuck_in_todo"]








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

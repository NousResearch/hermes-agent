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
# truthful running liveness
# ---------------------------------------------------------------------------


def test_running_diagnostic_rejects_stale_heartbeat_and_foreign_pid(monkeypatch):
    """A running row is not live merely because its stale PID now exists.

    This models the two production failures safely: an orphaned child process
    survives after its worker exits, and a recycled PID belongs to a stranger.
    The diagnostic must surface both stale heartbeat and mismatched identity.
    """
    from hermes_cli import kanban_db_dispatch as kbd

    now = 100_000
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, fingerprint: False)
    task = _task(
        status="running",
        claim_lock=f"{kb._host_prefix()}123",
        worker_pid=4242,
        worker_started_at="old-boot|10",
        started_at=now - 2 * 3600,
        last_heartbeat_at=now - 3601,
    )

    diags = kd.compute_task_diagnostics(task, [], [], now=now)
    liveness = [d for d in diags if d.kind == "running_liveness_stale"]

    assert len(liveness) == 1
    assert liveness[0].severity == "error"
    assert liveness[0].data["heartbeat_stale"] is True
    assert liveness[0].data["worker_identity_matches"] is False
    assert "heartbeat" in liveness[0].detail.lower()
    assert "identity" in liveness[0].detail.lower()


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

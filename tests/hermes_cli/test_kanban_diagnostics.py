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
        "current_run_id": None,
        "claim_lock": None,
        "worker_pid": None,
        "last_heartbeat_at": None,
        "started_at": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(
    outcome="completed",
    run_id=1,
    error=None,
    status="completed",
    claim_lock=None,
    worker_pid=None,
    last_heartbeat_at=None,
    started_at=None,
    ended_at=None,
    **overrides,
):
    base = {
        "id": run_id,
        "outcome": outcome,
        "status": status,
        "error": error,
        "claim_lock": claim_lock,
        "worker_pid": worker_pid,
        "last_heartbeat_at": last_heartbeat_at,
        "started_at": started_at,
        "ended_at": ended_at,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# running worker identity diagnostics
# ---------------------------------------------------------------------------


def _running_identity(*, age, task_pid, run_pid, now=10_000, run_present=True):
    task = _task(
        status="running",
        started_at=now - age,
        current_run_id=7,
        claim_lock="host:1",
        worker_pid=task_pid,
        last_heartbeat_at=None,
    )
    runs = []
    if run_present:
        runs = [_run(
            run_id=7,
            status="running",
            claim_lock="host:1",
            worker_pid=run_pid,
            started_at=now - age,
            ended_at=None,
            last_heartbeat_at=None,
        )]
    return now, task, runs


def test_running_pid_missing_after_launch_grace_is_error():
    now, task, runs = _running_identity(age=31, task_pid=None, run_pid=None)
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    diag = next(d for d in diags if d.kind == "running_worker_pid_missing")
    assert diag.severity == "error"
    assert all(a.kind != "reclaim" for a in diag.actions)
    assert any(
        a.kind == "cli_hint" and "reconcile" in (a.payload or {}).get("command", "")
        for a in diag.actions
    )
    assert diag.data["missing_layers"] == ["task", "run"]
    assert diag.data["task_worker_pid"] is None
    assert diag.data["run_worker_pid"] is None
    assert "task row" in diag.detail
    assert "current run row" in diag.detail
    assert not any(d.kind == "running_worker_run_mismatch" for d in diags)


def test_running_pid_missing_within_grace_does_not_fire():
    now, task, runs = _running_identity(age=25, task_pid=None, run_pid=None)
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    assert not any(d.kind == "running_worker_pid_missing" for d in diags)
    assert not any(d.kind == "running_worker_run_mismatch" for d in diags)


def test_running_one_sided_task_pid_past_grace_is_mismatch_only():
    now, task, runs = _running_identity(age=31, task_pid=1234, run_pid=None)
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    kinds = [d.kind for d in diags if d.kind.startswith("running_worker_")]
    assert kinds == ["running_worker_run_mismatch"]
    diag = diags[0]
    assert diag.severity == "critical"
    assert diag.data["task_worker_pid"] == 1234
    assert diag.data["run_worker_pid"] is None
    assert "task or run" not in diag.detail


def test_running_one_sided_run_pid_past_grace_is_mismatch_only():
    now, task, runs = _running_identity(age=31, task_pid=None, run_pid=5678)
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    kinds = [d.kind for d in diags if d.kind.startswith("running_worker_")]
    assert kinds == ["running_worker_run_mismatch"]
    diag = diags[0]
    assert diag.severity == "critical"
    assert diag.data["task_worker_pid"] is None
    assert diag.data["run_worker_pid"] == 5678
    assert "task or run" not in diag.detail


def test_running_run_mismatch_missing_current_run():
    now, task, runs = _running_identity(
        age=50, task_pid=1234, run_pid=None, run_present=False,
    )
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    kinds = [d.kind for d in diags if d.kind.startswith("running_worker_")]
    assert kinds == ["running_worker_run_mismatch"]
    diag = next(d for d in diags if d.kind == "running_worker_run_mismatch")
    assert diag.severity == "critical"
    assert any("reconcile" in (a.payload or {}).get("command", "") for a in diag.actions)


def test_running_run_mismatch_task_pid_without_run_pid():
    now, task, runs = _running_identity(age=10, task_pid=1234, run_pid=None)
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    kinds = [d.kind for d in diags if d.kind.startswith("running_worker_")]
    assert kinds == ["running_worker_run_mismatch"]
    diag = diags[0]
    assert diag.severity == "critical"
    assert diag.data["task_worker_pid"] == 1234
    assert diag.data["run_worker_pid"] is None


def test_running_run_mismatch_run_pid_without_task_pid():
    now, task, runs = _running_identity(age=10, task_pid=None, run_pid=5678)
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    kinds = [d.kind for d in diags if d.kind.startswith("running_worker_")]
    assert kinds == ["running_worker_run_mismatch"]
    diag = diags[0]
    assert diag.severity == "critical"
    assert diag.data["task_worker_pid"] is None
    assert diag.data["run_worker_pid"] == 5678


def test_running_run_mismatch_run_already_ended():
    now = 10_000
    task = _task(
        status="running",
        started_at=now - 50,
        current_run_id=7,
        claim_lock="host:1",
        worker_pid=1234,
    )
    runs = [_run(
        run_id=7,
        status="completed",
        ended_at=now - 10,
        claim_lock="host:1",
        worker_pid=1234,
    )]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    diag = next(d for d in diags if d.kind == "running_worker_run_mismatch")
    assert diag.severity == "critical"
    assert not any(
        d.kind in ("running_worker_pid_missing", "running_worker_heartbeat_missing")
        for d in diags
    )


def test_running_run_mismatch_claim_lock_differ():
    now = 10_000
    task = _task(
        status="running",
        started_at=now - 50,
        current_run_id=7,
        claim_lock="host:1",
        worker_pid=1234,
    )
    runs = [_run(
        run_id=7,
        status="running",
        ended_at=None,
        claim_lock="host:2",
        worker_pid=1234,
    )]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    diag = next(d for d in diags if d.kind == "running_worker_run_mismatch")
    assert diag.severity == "critical"
    assert not any(
        d.kind in ("running_worker_pid_missing", "running_worker_heartbeat_missing")
        for d in diags
    )


def test_running_run_mismatch_pids_differ():
    now = 10_000
    task = _task(
        status="running",
        started_at=now - 50,
        current_run_id=7,
        claim_lock="host:1",
        worker_pid=1234,
    )
    runs = [_run(
        run_id=7,
        status="running",
        ended_at=None,
        claim_lock="host:1",
        worker_pid=5678,
    )]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    diag = next(d for d in diags if d.kind == "running_worker_run_mismatch")
    assert diag.severity == "critical"
    assert not any(
        d.kind in ("running_worker_pid_missing", "running_worker_heartbeat_missing")
        for d in diags
    )


def test_running_heartbeat_missing_after_120s():
    now = 10_000
    task = _task(
        status="running",
        started_at=now - 121,
        current_run_id=7,
        claim_lock="host:1",
        worker_pid=1234,
        last_heartbeat_at=None,
    )
    runs = [_run(
        run_id=7,
        status="running",
        ended_at=None,
        claim_lock="host:1",
        worker_pid=1234,
        last_heartbeat_at=None,
    )]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    diag = next(d for d in diags if d.kind == "running_worker_heartbeat_missing")
    assert diag.severity == "warning"
    assert any("reconcile" in (a.payload or {}).get("command", "") for a in diag.actions)
    assert [d.kind for d in diags if d.kind.startswith("running_worker_")] == [
        "running_worker_heartbeat_missing",
    ]


def test_running_heartbeat_suppressed_when_identity_is_split():
    now, task, runs = _running_identity(age=121, task_pid=1234, run_pid=None)
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    kinds = [d.kind for d in diags if d.kind.startswith("running_worker_")]
    assert kinds == ["running_worker_run_mismatch"]


def test_running_heartbeat_suppressed_when_both_pids_missing():
    now, task, runs = _running_identity(age=121, task_pid=None, run_pid=None)
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    kinds = [d.kind for d in diags if d.kind.startswith("running_worker_")]
    assert kinds == ["running_worker_pid_missing"]


def test_running_heartbeat_missing_within_grace_does_not_fire():
    now = 10_000
    task = _task(
        status="running",
        started_at=now - 119,
        current_run_id=7,
        claim_lock="host:1",
        worker_pid=1234,
        last_heartbeat_at=None,
    )
    runs = [_run(
        run_id=7,
        status="running",
        ended_at=None,
        claim_lock="host:1",
        worker_pid=1234,
        last_heartbeat_at=None,
    )]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    assert not any(d.kind == "running_worker_heartbeat_missing" for d in diags)


def test_running_healthy_worker_produces_no_identity_diagnostics():
    now = 10_000
    task = _task(
        status="running",
        started_at=now - 200,
        current_run_id=7,
        claim_lock="host:1",
        worker_pid=1234,
        last_heartbeat_at=now - 30,
    )
    runs = [_run(
        run_id=7,
        status="running",
        ended_at=None,
        claim_lock="host:1",
        worker_pid=1234,
        last_heartbeat_at=now - 30,
    )]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    assert not any(d.kind.startswith("running_worker_") for d in diags)


def test_terminal_task_produces_no_running_identity_diagnostics():
    now = 10_000
    task = _task(
        status="done",
        started_at=now - 500,
        current_run_id=7,
        claim_lock=None,
        worker_pid=None,
    )
    runs = [_run(
        run_id=7,
        status="completed",
        ended_at=now - 100,
    )]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    assert not any(d.kind.startswith("running_worker_") for d in diags)

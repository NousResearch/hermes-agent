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
    # Freshly started, so the run is not also flagged for missing liveness.
    diags = kd.compute_task_diagnostics(_task(status="running", started_at=int(time.time())), [], [], graph=graph)
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


def test_running_liveness_reports_unavailable_heartbeat_without_zero_age(monkeypatch):
    """A dead local worker with no timestamp must not imply a fabricated 0s age."""
    from hermes_cli import kanban_db_dispatch as kbd

    now = 100_000
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, fingerprint: False)
    task = _task(
        status="running",
        claim_lock=f"{kb._host_prefix()}123",
        worker_pid=4242,
        worker_started_at="old-boot|10",
        started_at=None,
        last_heartbeat_at=None,
    )

    diags = kd.compute_task_diagnostics(task, [], [], now=now)
    liveness = [d for d in diags if d.kind == "running_liveness_stale"]

    assert len(liveness) == 1
    assert "heartbeat timestamp unavailable" in liveness[0].detail.lower()
    assert "for 0s" not in liveness[0].detail.lower()
    assert liveness[0].data["heartbeat_age_seconds"] is None


def test_running_liveness_never_reports_runtime_as_heartbeat_age():
    """A record that never heartbeated is stale by runtime, not by heartbeat age.

    The time since ``started_at`` must surface as its own field and wording so
    a dashboard cannot read a runtime as the age of a heartbeat that never
    happened (the dispatcher keeps the two apart the same way).
    """
    now = 100_000
    task = _task(
        status="running",
        claim_lock="other-host:123",
        worker_pid=4242,
        started_at=now - 2 * 3600,
        last_heartbeat_at=None,
    )

    diags = kd.compute_task_diagnostics(task, [], [], now=now)
    liveness = [d for d in diags if d.kind == "running_liveness_stale"]

    assert len(liveness) == 1
    data = liveness[0].data
    assert data["heartbeat_age_seconds"] is None
    assert data["running_age_seconds"] == now - task["started_at"]
    assert data["heartbeat_stale"] is True
    detail = liveness[0].detail.lower()
    assert "no fresh heartbeat for" not in detail
    assert f"no heartbeat recorded; running for {data['running_age_seconds']}s" in detail


def _live_running_task(now, **overrides):
    task = _task(
        status="running",
        claim_lock=f"{kb._host_prefix()}123",
        worker_pid=4242,
        worker_started_at="boot|10",
        started_at=now - 2 * 3600,
        last_heartbeat_at=now - 60,
    )
    task.update(overrides)
    return task


def _liveness(task, now):
    diags = kd.compute_task_diagnostics(task, [], [], now=now)
    return [d for d in diags if d.kind == "running_liveness_stale"]


def test_running_liveness_silent_for_fresh_heartbeat_and_matching_identity(monkeypatch):
    """The core contract: a live record (fresh heartbeat, PID matches its spawn
    fingerprint) is not flagged."""
    from hermes_cli import kanban_db_dispatch as kbd

    now = 100_000
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, fingerprint: True)
    assert _liveness(_live_running_task(now), now) == []


def test_running_liveness_heartbeat_threshold_boundary(monkeypatch):
    """One second under the limit is silent; at the limit it fires (``>=``)."""
    from hermes_cli import kanban_db_dispatch as kbd

    now = 100_000
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, fingerprint: True)
    limit = kd._RUNNING_LIVENESS_HEARTBEAT_GAP_SECONDS
    assert _liveness(_live_running_task(now, last_heartbeat_at=now - limit + 1), now) == []
    assert len(_liveness(_live_running_task(now, last_heartbeat_at=now - limit), now)) == 1


def test_running_liveness_reads_iso_timestamps_like_the_dispatcher(monkeypatch):
    """Older rows store ISO-8601 text. A fresh ISO heartbeat must keep the rule
    silent; a stale one must still fire with its real age."""
    from datetime import UTC, datetime

    from hermes_cli import kanban_db_dispatch as kbd

    now = 1_800_000_000
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, fingerprint: True)

    def iso(age):
        return datetime.fromtimestamp(now - age, tz=UTC).isoformat()

    assert _liveness(_live_running_task(now, last_heartbeat_at=iso(60)), now) == []
    stale = _liveness(_live_running_task(now, last_heartbeat_at=iso(2 * 3600)), now)
    assert len(stale) == 1
    assert stale[0].data["heartbeat_age_seconds"] == 2 * 3600


def test_running_liveness_ages_the_current_run_not_the_first_claim(monkeypatch):
    """``tasks.started_at`` keeps the first claim's time across retries and
    reclaim clears the heartbeat. A retried task whose current run started 60s
    ago has not had time to heartbeat and must not be flagged."""
    from hermes_cli import kanban_db_dispatch as kbd

    now = 100_000
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, fingerprint: True)
    task = _live_running_task(
        now, current_run_id=9, started_at=now - 5 * 3600, last_heartbeat_at=None)
    runs = [{"id": 8, "started_at": now - 5 * 3600}, {"id": 9, "started_at": now - 60}]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    assert not [d for d in diags if d.kind == "running_liveness_stale"]


def test_running_liveness_ignores_heartbeat_left_over_from_an_earlier_run(monkeypatch):
    """A block/unblock/claim cycle can carry the previous run's old heartbeat
    onto a fresh run; the fresh run is not stale yet."""
    from hermes_cli import kanban_db_dispatch as kbd

    now = 100_000
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, fingerprint: True)
    task = _live_running_task(
        now, current_run_id=9, started_at=now - 5 * 3600, last_heartbeat_at=now - 4 * 3600)
    runs = [{"id": 9, "started_at": now - 120}]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    assert not [d for d in diags if d.kind == "running_liveness_stale"]


def test_running_liveness_fresh_run_pid_mismatch_has_no_heartbeat_noise(monkeypatch):
    """A young run whose PID identity is wrong reports the PID problem only."""
    from hermes_cli import kanban_db_dispatch as kbd

    now = 100_000
    monkeypatch.setattr(kbd, "_worker_alive", lambda pid, fingerprint: False)
    task = _live_running_task(now, started_at=now - 30, last_heartbeat_at=None)
    found = _liveness(task, now)
    assert len(found) == 1
    assert "spawn identity" in found[0].detail
    assert "heartbeat timestamp unavailable" not in found[0].detail


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

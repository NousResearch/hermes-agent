"""One-strike systemic breaker trips need immediate operator diagnostics."""
import json

import pytest

from hermes_cli.kanban_diagnostics import compute_task_diagnostics


def _diagnostics(payload, *, later=(), status="blocked", failures=1, now=101):
    task = {"id": "t_sticky", "status": status, "assignee": "default",
            "consecutive_failures": failures,
            "last_failure_error": "ModuleNotFoundError: No module named 'hermes_cli.main'"}
    events = [{"kind": "gave_up", "created_at": 100, "payload": payload}, *later]
    runs = [{"id": 1, "outcome": "crashed", "error": task["last_failure_error"]}]
    return compute_task_diagnostics(task, events, runs, now=now,
                                    config={"failure_threshold": 7, "failure_limit": 9})


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("now", [101, 100 + 30 * 86400])
def test_sticky_one_strike_trip_is_visible_without_blocked_event(as_json, now):
    payload = {"sticky": True, "failures": 1, "effective_limit": 1,
               "trigger_outcome": "crashed"}
    diagnostics = _diagnostics(json.dumps(payload) if as_json else payload, now=now)
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.kind == "repeated_failures"
    assert diagnostic.severity == "error"
    assert "sticky" in diagnostic.title.lower()
    assert "credential" not in diagnostic.title.lower()
    assert "ModuleNotFoundError" in diagnostic.detail
    assert "unblock" in diagnostic.detail.lower()
    assert any(a.payload.get("command") == "hermes kanban log t_sticky"
               for a in diagnostic.actions)
    assert diagnostic.data["failure_threshold"] == 7
    assert diagnostic.data["failure_limit"] == 9


@pytest.mark.parametrize("kind", ["unblocked", "promoted", "completed", "claimed"])
def test_recovered_sticky_trip_does_not_reappear(kind):
    assert _diagnostics({"sticky": True}, later=[{"kind": kind, "created_at": 101}]) == []


@pytest.mark.parametrize("status", ["done", "archived", "running"])
def test_sticky_trip_preserves_inactive_and_inflight_exemptions(status):
    assert _diagnostics({"sticky": True}, status=status) == []


@pytest.mark.parametrize("payload", [{}, {"sticky": False}, None, "invalid json"])
def test_ordinary_below_threshold_failure_stays_quiet(payload):
    assert _diagnostics(payload) == []


def test_newer_ordinary_trip_supersedes_old_sticky_trip():
    assert _diagnostics({"sticky": True}, later=[
        {"kind": "gave_up", "created_at": 101, "payload": {"sticky": False}},
    ]) == []


def test_terminal_provider_copy_is_preserved_even_when_sticky():
    diagnostics = _diagnostics({"terminal_provider": True, "sticky": True})
    assert len(diagnostics) == 1
    assert "credential or model" in diagnostics[0].title


def test_ordinary_threshold_and_error_fallback_are_preserved():
    diagnostics = _diagnostics({}, failures=7)
    assert len(diagnostics) == 1
    assert "ModuleNotFoundError" in diagnostics[0].title


def test_systemic_trip_round_trips_through_real_board_and_clears_on_unblock(tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_db_dispatch import _account_crashes, _DeadWorker

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    try:
        task_ids = [kb.create_task(conn, title=f"worker {i}", assignee="default")
                    for i in range(3)]
        kb.recompute_ready(conn)
        error = "ModuleNotFoundError: No module named 'hermes_cli.main'"
        dead = _DeadWorker("unknown", None, error, "crashed", {})
        assert set(_account_crashes(conn, [(tid, 0, "fixture", dead) for tid in task_ids])) == set(task_ids)
        for tid in task_ids:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
            events = kb.list_events(conn, tid)
            assert row["status"] == "blocked"
            assert row["consecutive_failures"] == 1
            assert not any(e.kind == "blocked" for e in events)
            diagnostics = compute_task_diagnostics(row, events, kb.list_runs(conn, tid))
            assert [d.kind for d in diagnostics] == ["repeated_failures"]
            assert diagnostics[0].severity == "error"
            assert error in diagnostics[0].detail
            assert kb.unblock_task(conn, tid)
            assert compute_task_diagnostics(kb.get_task(conn, tid), kb.list_events(conn, tid),
                                            kb.list_runs(conn, tid)) == []
    finally:
        conn.close()

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent import estop
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_recover as recover


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    estop.disengage()
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    monkeypatch.setattr(recover, "recover_command_enabled", lambda: True)
    return home


def _failed_task(conn, outcome: str, *, status: str = "ready", failures: int = 0) -> str:
    task_id = kb.create_task(conn, title=outcome, assignee="missing-profile")
    assert kb.claim_task(conn, task_id, claimer="acceptance") is not None
    run_id = kb.get_task(conn, task_id).current_run_id
    now = int(time.time())
    conn.execute(
        "UPDATE task_runs SET status=?, outcome=?, ended_at=? WHERE id=?",
        (outcome, outcome, now, run_id),
    )
    conn.execute(
        "UPDATE tasks SET status=?, current_run_id=NULL, claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, consecutive_failures=? WHERE id=?",
        (status, failures, task_id),
    )
    conn.commit()
    kb._append_event(conn, task_id, outcome, {"retry_status": status}, run_id=run_id)
    conn.commit()
    return task_id


def _snapshot(conn, task_id: str) -> tuple:
    task = kb.get_task(conn, task_id)
    return (
        task.status,
        task.current_run_id,
        task.consecutive_failures,
        conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)).fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0],
    )


def test_triage_is_truthful_repeatable_and_zero_mutation(kanban_home, monkeypatch):
    dispatch_calls = []

    def unexpected_dispatch(*args, **kwargs):
        dispatch_calls.append((args, kwargs))
        raise AssertionError("triage recovery must not enter the dispatcher")

    monkeypatch.setattr(recover.kbd, "dispatch_once", unexpected_dispatch)
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="needs triage", assignee="a", triage=True)
        before = (
            tuple(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()),
            tuple(tuple(row) for row in conn.execute(
                "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()),
            tuple(tuple(row) for row in conn.execute(
                "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()),
        )

    first = recover.run_recover_slash(task_id)
    second = recover.run_recover_slash(task_id)

    assert first == second
    assert first["task_status"] == "triage"
    assert first["recovery_state"] == "waiting"
    assert first["eligible"] is False
    assert first["action"] == "wait"
    assert first["mutation_performed"] is False
    assert first["dispatch_status"] == "not_eligible"
    assert first["run_id"] is None
    assert first["retry_info"] is None
    assert first["next_action"] is None
    assert "triage" in first["message"].lower()
    assert "/recover does not act" in first["message"].lower()
    assert "healthy" not in first["message"].lower()
    assert dispatch_calls == []

    with kbc.connect() as conn:
        after = (
            tuple(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()),
            tuple(tuple(row) for row in conn.execute(
                "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()),
            tuple(tuple(row) for row in conn.execute(
                "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,)
            ).fetchall()),
        )
        assert after == before
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? "
            "AND kind IN ('recovered', 'unblocked', 'routing_selected')", (task_id,)
        ).fetchone()[0] == 0


def test_waiting_and_healthy_states_are_read_only(kanban_home):
    with kbc.connect() as conn:
        waiting = kb.create_task(conn, title="waiting", assignee="a")
        healthy = kb.create_task(conn, title="healthy", assignee="a")
        conn.execute("UPDATE tasks SET status='scheduled' WHERE id=?", (waiting,))
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (healthy,))
        conn.commit()
        before_waiting = _snapshot(conn, waiting)
        before_healthy = _snapshot(conn, healthy)

    waiting_result = recover.run_recover_slash(waiting)
    healthy_result = recover.run_recover_slash(healthy)
    assert waiting_result["recovery_state"] == "waiting"
    assert waiting_result["mutation_performed"] is False
    assert healthy_result["recovery_state"] == "healthy"
    assert healthy_result["mutation_performed"] is False
    with kbc.connect() as conn:
        assert _snapshot(conn, waiting) == before_waiting
        assert _snapshot(conn, healthy) == before_healthy


def test_expired_rate_limit_is_read_only_and_ready(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect() as conn:
        task_id = _failed_task(conn, "rate_limited")
        before = _snapshot(conn, task_id)
    result = recover.run_recover_slash(task_id)
    assert result["recovery_state"] == "healthy"
    assert result["dispatch_status"] == "already_recovered"
    assert result["mutation_performed"] is False
    with kbc.connect() as conn:
        assert _snapshot(conn, task_id) == before


def test_sticky_block_default_and_requeue_are_fail_closed(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="sticky", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
        assert kb.block_task(conn, task_id, reason="operator hold")
        before = _snapshot(conn, task_id)
    first = recover.run_recover_slash(task_id)
    second = recover.run_recover_slash(f"{task_id} --requeue")
    assert first["recovery_state"] == "blocked"
    assert second["recovery_state"] == "sticky-blocked"
    assert first["mutation_performed"] is False
    assert second["mutation_performed"] is False
    with kbc.connect() as conn:
        assert _snapshot(conn, task_id) == before
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind IN ('unblocked','recovered')",
            (task_id,),
        ).fetchone()[0] == 0


def test_sticky_requeue_concurrency_has_no_side_effects(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="sticky concurrent", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
        assert kb.block_task(conn, task_id, reason="operator hold")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: recover.run_recover_slash(f"{task_id} --requeue"), range(4)))
    assert all(item["recovery_state"] == "sticky-blocked" for item in results)
    assert all(item["mutation_performed"] is False for item in results)
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='unblocked'", (task_id,)
        ).fetchone()[0] == 0


def test_requeue_is_idempotent_and_has_no_routing_metadata(kanban_home):
    with kbc.connect() as conn:
        task_id = _failed_task(conn, "gave_up", status="blocked", failures=3)
    first = recover.run_recover_slash(f"{task_id} --requeue")
    second = recover.run_recover_slash(f"{task_id} --requeue")
    assert first["mutation_performed"] is True
    assert second["mutation_performed"] is False
    assert second["dispatch_status"] == "already_recovered"
    assert all(key not in first for key in ("provider", "model", "implementation_profile", "reviewer_profile"))
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='unblocked'", (task_id,)
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='routing_selected'", (task_id,)
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)).fetchone()[0] == 1


def test_estop_blocks_default_and_requeue_without_db_mutation(kanban_home):
    with kbc.connect() as conn:
        task_id = _failed_task(conn, "gave_up", status="blocked", failures=3)
        before = _snapshot(conn, task_id)
    estop.engage(reason="acceptance")
    try:
        default = recover.run_recover_slash(task_id)
        requeue = recover.run_recover_slash(f"{task_id} --requeue")
    finally:
        estop.disengage()
    assert default["dispatch_status"] == "paused"
    assert requeue["dispatch_status"] == "paused"
    assert default["mutation_performed"] is False
    assert requeue["mutation_performed"] is False
    with kbc.connect() as conn:
        assert _snapshot(conn, task_id) == before

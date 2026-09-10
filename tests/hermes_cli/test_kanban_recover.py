from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from hermes_cli import commands
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_recover as recover


CANONICAL_RECOVER_ACTIONS = frozenset({"normalized", "none", "requeued", "reported"})


def _assert_action(result: dict, expected: str) -> None:
    assert result["action"] == expected
    assert result["action"] in CANONICAL_RECOVER_ACTIONS


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    monkeypatch.setattr(recover, "recover_command_enabled", lambda: True)
    return home


def _failure(conn, outcome: str, *, status: str = "ready", failures: int = 0) -> str:
    task_id = kb.create_task(conn, title=outcome, assignee="missing-profile")
    assert kb.claim_task(conn, task_id, claimer="test") is not None
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


def test_dispatch_once_max_spawn_zero_reclaims_without_spawn_or_routing(kanban_home):
    calls = []
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="dead worker", assignee="missing-profile")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
        assert kb.claim_task(conn, task_id, claimer="probe") is not None
        conn.execute(
            "UPDATE tasks SET worker_pid=?, started_at=?, claim_expires=? WHERE id=?",
            (999999999, time.time() - 7200, time.time() - 3600, task_id),
        )
        conn.commit()
        before_runs = conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)).fetchone()[0]
        result = kbd.dispatch_once(
            conn, task_id=task_id, max_spawn=0,
            spawn_fn=lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        after_runs = conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=?", (task_id,)).fetchone()[0]
        assert result.crashed or result.reclaimed or result.reconciled_orphans
        assert after_runs == before_runs
        assert result.spawned == []
        assert calls == []
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='routing_selected'", (task_id,)
        ).fetchone()[0] == 0


def test_recover_failed_ready_marks_ready_for_manual_continue(kanban_home):
    with kbc.connect() as conn:
        task_id = _failure(conn, "crashed")
    result = recover.run_recover_slash(task_id)
    assert result["command"] == "recover"
    assert result["recovery_state"] == "recovered"
    _assert_action(result, "normalized")
    assert result["next_action"] == f"/continue {task_id}"
    assert result["mutation_performed"] is True
    repeated = recover.run_recover_slash(task_id)
    _assert_action(repeated, "none")
    assert repeated["mutation_performed"] is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='routing_selected'", (task_id,)
        ).fetchone()[0] == 0


@pytest.mark.parametrize("outcome", ["timed_out", "spawn_failed", "reclaimed", "stale"])
def test_recover_all_authoritative_failure_outcomes(kanban_home, outcome):
    with kbc.connect() as conn:
        task_id = _failure(conn, outcome)
    result = recover.run_recover_slash(task_id)
    assert result["recovery_state"] == "recovered"
    _assert_action(result, "normalized")
    assert result["next_action"] == f"/continue {task_id}"
    assert result["mutation_performed"] is True


def test_recover_does_not_reclaim_live_worker(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="live", assignee="a")
        assert kb.claim_task(conn, task_id, claimer="live-worker") is not None
        conn.execute(
            "UPDATE tasks SET worker_pid=?, last_heartbeat_at=?, claim_expires=? WHERE id=?",
            (1, time.time(), time.time() + 3600, task_id),
        )
        conn.commit()
    result = recover.run_recover_slash(task_id)
    assert result["recovery_state"] in {"active", "healthy"}
    _assert_action(result, "none")
    assert result["mutation_performed"] is False
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "running"


def test_recover_gave_up_requires_explicit_requeue(kanban_home):
    with kbc.connect() as conn:
        task_id = _failure(conn, "gave_up", status="blocked", failures=2)
    result = recover.run_recover_slash(task_id)
    assert result["recovery_state"] == "gave-up"
    _assert_action(result, "none")
    assert result["mutation_performed"] is False
    requeued = recover.run_recover_slash(f"{task_id} --requeue")
    assert requeued["dispatch_status"] == "requeued"
    _assert_action(requeued, "requeued")
    assert requeued["mutation_performed"] is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='unblocked'", (task_id,)
        ).fetchone()[0] == 1


def test_recover_requeue_preserves_explicit_sticky_block(kanban_home):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="operator block", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
        assert kb.block_task(conn, task_id, reason="human hold")
        before = kb.get_task(conn, task_id)
    result = recover.run_recover_slash(f"{task_id} --requeue")
    assert result["recovery_state"] == "sticky-blocked"
    _assert_action(result, "none")
    assert result["mutation_performed"] is False
    with kbc.connect() as conn:
        after = kb.get_task(conn, task_id)
        assert after.status == before.status == "blocked"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='unblocked'", (task_id,)
        ).fetchone()[0] == 0


def test_recover_rate_limit_cooldown_is_read_only(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    with kbc.connect() as conn:
        task_id = _failure(conn, "rate_limited")
        before = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0]
    result = recover.run_recover_slash(task_id)
    assert result["recovery_state"] == "rate-limited"
    _assert_action(result, "none")
    assert result["mutation_performed"] is False
    with kbc.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0] == before


@pytest.mark.parametrize("status", ["done", "archived"])
def test_recover_terminal_is_read_only(kanban_home, status):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="done", assignee="a")
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
        conn.commit()
    result = recover.run_recover_slash(task_id)
    assert result["recovery_state"] == "terminal"
    _assert_action(result, "none")
    assert result["mutation_performed"] is False


def test_recover_concurrent_calls_are_idempotent(kanban_home):
    with kbc.connect() as conn:
        task_id = _failure(conn, "stale")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: recover.run_recover_slash(task_id), range(4)))
    assert sum(item["mutation_performed"] for item in results) == 1
    assert all(item["dispatch_status"] in {"recovered", "already_recovered"} for item in results)
    for item in results:
        _assert_action(item, "normalized" if item["mutation_performed"] else "none")
    with kbc.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='recovered'", (task_id,)
        ).fetchone()[0] == 1


def test_recover_concurrent_requeue_resets_breaker_once(kanban_home):
    with kbc.connect() as conn:
        task_id = _failure(conn, "gave_up", status="blocked", failures=3)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: recover.run_recover_slash(f"{task_id} --requeue"), range(4)))
    assert sum(item["mutation_performed"] for item in results) == 1
    for item in results:
        _assert_action(item, "requeued" if item["mutation_performed"] else "none")
    with kbc.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='unblocked'", (task_id,)
        ).fetchone()[0] == 1


def test_recover_estop_is_zero_mutation(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        task_id = _failure(conn, "crashed")
        before = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0]
    monkeypatch.setattr(recover.kbd, "dispatch_paused", lambda: True)
    result = recover.run_recover_slash(task_id)
    assert result["dispatch_status"] == "paused"
    _assert_action(result, "none")
    assert result["mutation_performed"] is False
    with kbc.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0] == before


def test_recover_blocker_auth_requires_no_recover_action(kanban_home, monkeypatch):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="auth blocked", assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()
    monkeypatch.setattr(recover.kbd, "check_respawn_guard", lambda *_args, **_kwargs: "blocker_auth")

    result = recover.run_recover_slash(task_id)

    assert result["recovery_state"] == "blocker-auth"
    _assert_action(result, "none")
    assert result["mutation_performed"] is False


@pytest.mark.parametrize(
    ("dispatch_case", "expected_state", "expected_action", "expected_mutation"),
    [
        ("paused", "paused", "none", False),
        ("gave_up", "gave-up", "none", False),
        ("stale", "recovered", "normalized", True),
    ],
)
def test_recover_post_dispatch_action_contract(
    kanban_home, monkeypatch, dispatch_case, expected_state, expected_action, expected_mutation,
):
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title=dispatch_case, assignee="a")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
        conn.commit()

    def dispatch_once(conn, **_kwargs):
        if dispatch_case == "paused":
            return kbd.DispatchResult(paused=True)
        if dispatch_case == "gave_up":
            conn.execute(
                "UPDATE tasks SET status='blocked', consecutive_failures=2 WHERE id=?",
                (task_id,),
            )
            conn.commit()
            return kbd.DispatchResult(auto_blocked=[task_id])
        return kbd.DispatchResult(stale=[task_id])

    monkeypatch.setattr(recover.kbd, "dispatch_paused", lambda: False)
    monkeypatch.setattr(recover.kbd, "dispatch_once", dispatch_once)

    result = recover.run_recover_slash(task_id)

    assert result["recovery_state"] == expected_state
    _assert_action(result, expected_action)
    assert result["mutation_performed"] is expected_mutation


def test_recover_command_registry_and_gate_default():
    command = commands.resolve_command("recover")
    assert command is not None
    assert command.name == "recover"
    assert command.gateway_config_gate == "kanban.recover_command"
    assert command.busy_policy == "dispatch"
    assert recover._GATE_KEY == "recover_command"

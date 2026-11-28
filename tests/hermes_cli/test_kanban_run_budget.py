"""Absolute run budgets on isolated boards; no live workers or provider calls."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda _: True)
    monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)
    monkeypatch.setattr(kb, "_pid_alive", lambda _: False)
    monkeypatch.setattr(dispatch, "_system_memory_sample", lambda: {})
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def tick(conn, spawned):
    return dispatch.dispatch_once(conn, spawn_fn=lambda task, workspace: spawned.append(task.current_run_id) or 99999999)


@pytest.mark.parametrize("exit_kind,code", [("rate_limited", 75), ("nonzero_exit", 1),
                                            ("clean_exit", 0), ("terminal_provider", 78)])
def test_one_run_stops_every_exit(board, monkeypatch, exit_kind, code):
    tid = kb.create_task(board, title="bounded", assignee="worker", max_worker_runs=1, max_retries=10)
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _: (exit_kind, code))
    spawned = []
    tick(board, spawned)
    assert kb.get_task(board, tid).worker_runs_used == 1
    tick(board, spawned)
    tick(board, spawned)
    task = kb.get_task(board, tid)
    assert len(spawned) == len(kb.list_runs(board, tid)) == 1
    assert task.status == "blocked"
    assert any(e.kind == "run_budget_exhausted" and "run-budget-exhausted" in e.payload["reason"]
               for e in kb.list_events(board, tid))
    if exit_kind == "rate_limited":
        assert task.consecutive_failures == 0
        assert kb.list_runs(board, tid)[0].outcome == "rate_limited"
    assert kb.recompute_ready(board) == 0
    assert not kb.unblock_task(board, tid)
    assert kb.claim_task(board, tid) is None


@pytest.mark.parametrize("cap,expected", [(None, 4), (1, 1)])
def test_rate_limit_uncapped_compatibility(board, monkeypatch, cap, expected):
    tid = kb.create_task(board, title="quota", assignee="worker", max_worker_runs=cap)
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _: ("rate_limited", 75))
    spawned = []
    for _ in range(4):
        tick(board, spawned)
    assert len(spawned) == expected
    assert kb.get_task(board, tid).consecutive_failures == 0


def test_concurrent_claims_reserve_last_slot_atomically(board):
    tid = kb.create_task(board, title="concurrent", max_worker_runs=1)
    barrier = threading.Barrier(8)
    db_path = Path(board.execute("PRAGMA database_list").fetchone()[2])

    def attempt(_):
        with kbc.connect_closing(db_path) as conn:
            barrier.wait(timeout=15)
            claimed = kb.claim_task(conn, tid)
            if claimed is not None:
                dispatch._record_task_failure(conn, tid, "neutral", outcome="spawn_failed",
                                              release_claim=True, end_run=True, infrastructure=True)
            return claimed is not None

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(8))) == 1
    assert kb.get_task(board, tid).worker_runs_used == 1
    assert kb.get_task(board, tid).status == "blocked"


def test_concurrent_dispatchers_cannot_spawn_second_run(board, monkeypatch):
    tid = kb.create_task(board, title="dispatch race", assignee="worker", max_worker_runs=1)
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _: ("rate_limited", 75))
    barrier = threading.Barrier(4)
    db_path = Path(board.execute("PRAGMA database_list").fetchone()[2])
    spawned = []

    def attempt(_):
        with kbc.connect_closing(db_path) as conn:
            barrier.wait(timeout=15)
            tick(conn, spawned)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(attempt, range(4)))
    tick(board, spawned)
    assert len(spawned) == kb.get_task(board, tid).worker_runs_used == 1


def test_owner_increase_and_clear_preserves_history_and_parked_status(board):
    tid = kb.create_task(board, title="raise", max_worker_runs=1)
    assert kb.claim_task(board, tid)
    dispatch._record_task_failure(board, tid, "neutral", outcome="spawn_failed",
                                  release_claim=True, end_run=True, infrastructure=True)
    assert kb.edit_task(board, tid, max_worker_runs=2)
    assert kb.get_task(board, tid).status == "blocked"
    assert kb.unblock_task(board, tid)
    assert kb.claim_task(board, tid)
    dispatch._record_task_failure(board, tid, "neutral", outcome="spawn_failed",
                                  release_claim=True, end_run=True, infrastructure=True)
    assert kb.get_task(board, tid).status == "blocked"
    assert kb.edit_task(board, tid, clear_max_worker_runs=True)
    assert kb.get_task(board, tid).status == "blocked"
    assert kb.get_task(board, tid).worker_runs_used == 2
    assert kb.unblock_task(board, tid)
    assert kb.claim_task(board, tid)
    assert kb.get_task(board, tid).worker_runs_used == 3


def test_existing_historical_runs_count_and_reassignment_never_refunds(board):
    tid = kb.create_task(board, title="existing", assignee="one")
    for _ in range(3):
        assert kb.claim_task(board, tid)
        dispatch._record_task_failure(board, tid, "neutral", outcome="spawn_failed",
                                      release_claim=True, end_run=True, infrastructure=True)
    assert kb.edit_task(board, tid, max_worker_runs=4)
    assert kb.assign_task(board, tid, "two")
    assert kb.claim_task(board, tid)
    dispatch._record_task_failure(board, tid, "neutral", outcome="spawn_failed",
                                  release_claim=True, end_run=True, infrastructure=True)
    assert kb.get_task(board, tid).worker_runs_used == 4
    assert kb.claim_task(board, tid) is None


def test_review_claim_uses_same_budget(board):
    tid = kb.create_task(board, title="review", max_worker_runs=1)
    assert kb.claim_task(board, tid)
    assert kb.request_review(board, tid, reviewer="reviewer", force=True)
    assert kb.get_task(board, tid).status == "blocked"
    assert kb.claim_review_task(board, tid) is None
    assert kb.edit_task(board, tid, max_worker_runs=2)
    assert kb.unblock_task(board, tid)
    assert kb.get_task(board, tid).status == "review"
    assert kb.claim_review_task(board, tid)


def test_review_crash_breaker_retains_review_resume_after_budget_parks(board, monkeypatch):
    tid = kb.create_task(board, title="review failure", max_worker_runs=1, max_retries=1)
    assert kb.request_review(board, tid, reviewer="worker", force=True)
    assert kb.claim_review_task(board, tid)
    dispatch._set_worker_pid(board, tid, 99999999)
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _: ("nonzero_exit", 1))
    dispatch.detect_crashed_workers(board)
    assert kb.get_task(board, tid).status == "blocked"
    assert kb.edit_task(board, tid, max_worker_runs=2)
    assert kb.unblock_task(board, tid)
    assert kb.get_task(board, tid).status == "review"


def test_zero_cap_and_scheduled_edit_never_admit_work(board):
    tid = kb.create_task(board, title="zero", max_worker_runs=0)
    assert kb.get_task(board, tid).status == "blocked"
    assert kb.claim_task(board, tid) is None
    parked = kb.create_task(board, title="scheduled")
    assert kb.schedule_task(board, parked)
    assert kb.edit_task(board, parked, max_worker_runs=4)
    assert kb.get_task(board, parked).status == "scheduled"
    assert kb.get_task(board, parked).worker_runs_used == 0


@pytest.mark.parametrize("value", [-1, True, 1.5, "2"])
def test_invalid_caps_rejected(board, value):
    with pytest.raises(ValueError, match="max_worker_runs"):
        kb.create_task(board, title="invalid", max_worker_runs=value)
    tid = kb.create_task(board, title="valid")
    with pytest.raises(ValueError, match="max_worker_runs"):
        kb.edit_task(board, tid, max_worker_runs=value)


def test_cli_create_edit_clear_and_show(board):
    result = json.loads(cli.run_slash('create "cli capped" --max-worker-runs 1 --json'))
    tid = result["id"] if "id" in result else result["task"]["id"]
    assert kb.claim_task(board, tid)
    output = json.loads(cli.run_slash(f"show {tid} --json"))["task"]
    assert (output["max_worker_runs"], output["worker_runs_used"]) == (1, 1)
    assert "1/1 run IDs used" in cli.run_slash(f"show {tid}")
    assert "Edited" in cli.run_slash(f"edit {tid} --max-worker-runs 2")
    assert kb.get_task(board, tid).max_worker_runs == 2
    assert "Edited" in cli.run_slash(f"edit {tid} --clear-max-worker-runs")
    assert kb.get_task(board, tid).max_worker_runs is None
    assert kb.list_tasks(board)[0].worker_runs_used == 1


@pytest.mark.parametrize("recovery", ["expired", "stale", "timeout", "orphan"])
def test_recovery_paths_park_instead_of_requeue(board, recovery):
    tid = kb.create_task(board, title="recovery", max_worker_runs=1, max_runtime_seconds=1)
    assert kb.claim_task(board, tid)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET started_at = 1, claim_expires = 2, last_heartbeat_at = 1 WHERE id = ?", (tid,))
        board.execute("UPDATE task_runs SET started_at = 1 WHERE task_id = ?", (tid,))
        if recovery == "orphan":
            board.execute("UPDATE tasks SET claim_lock = NULL WHERE id = ?", (tid,))
        if recovery == "timeout":
            board.execute("UPDATE tasks SET worker_pid = 99999999 WHERE id = ?", (tid,))
    if recovery == "expired":
        kb.release_stale_claims(board)
    elif recovery == "stale":
        dispatch.detect_stale_running(board, stale_timeout_seconds=1)
    elif recovery == "timeout":
        dispatch.enforce_max_runtime(board, signal_fn=lambda *_: None)
    else:
        dispatch.reconcile_orphaned_running(board)
    assert kb.get_task(board, tid).status == "blocked"
    assert kb.get_task(board, tid).worker_runs_used == 1
    assert kb.claim_task(board, tid) is None


def test_worker_cannot_raise_its_cap(board, monkeypatch):
    tid = kb.create_task(board, title="owned", max_worker_runs=1)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    with pytest.raises(ValueError, match="owner-managed"):
        kb.edit_task(board, tid, max_worker_runs=2)
    with pytest.raises(ValueError, match="owner-managed"):
        kb.edit_task(board, tid, clear_max_worker_runs=True)
    assert kb.get_task(board, tid).max_worker_runs == 1


def test_uncapped_failure_threshold_unchanged(board, monkeypatch):
    tid = kb.create_task(board, title="failures", assignee="worker", max_retries=2)
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _: ("nonzero_exit", 1))
    spawned = []
    for _ in range(4):
        tick(board, spawned)
    assert len(spawned) == 2
    assert kb.get_task(board, tid).consecutive_failures == 2
    assert not any(e.kind == "run_budget_exhausted" for e in kb.list_events(board, tid))


def test_migration_defaults_uncapped_and_preserves_history(board):
    # Recreate the pre-feature shape in this disposable board, then use the
    # real additive migration (never a live DB).
    tid = kb.create_task(board, title="legacy")
    assert kb.claim_task(board, tid)
    board.execute("ALTER TABLE tasks DROP COLUMN max_worker_runs")
    board.commit()
    kbc._migrate_add_optional_columns(board)
    task = kb.get_task(board, tid)
    assert task.max_worker_runs is None and task.worker_runs_used == 1


@pytest.mark.parametrize("cap", [None, 1])
@pytest.mark.parametrize("lane", ["ready", "review"])
@pytest.mark.parametrize("failure", ["crash", "timeout"])
def test_failure_threshold_keeps_accounting_and_resume_lane(board, monkeypatch, cap, lane, failure):
    tid = kb.create_task(board, title="threshold", assignee="worker", max_worker_runs=cap,
                         max_retries=1, max_runtime_seconds=1)
    if lane == "review":
        assert kb.request_review(board, tid, reviewer="worker", force=True)
    claim = kb.claim_review_task if lane == "review" else kb.claim_task
    assert claim(board, tid)
    dispatch._set_worker_pid(board, tid, 99999999)
    if failure == "crash":
        monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _: ("nonzero_exit", 1))
        assert dispatch.detect_crashed_workers(board) == [tid]
    else:
        with kb.write_txn(board):
            board.execute("UPDATE task_runs SET started_at = 1 WHERE task_id = ?", (tid,))
        assert dispatch.enforce_max_runtime(board, signal_fn=lambda *_: None) == [tid]

    task = kb.get_task(board, tid)
    run = kb.list_runs(board, tid)[0]
    trip = next(e for e in kb.list_events(board, tid) if e.kind == "gave_up")
    assert task is not None and trip.payload is not None
    assert task.status == "blocked"
    assert task.consecutive_failures == trip.payload["failures"] == 1
    assert task.last_failure_error == trip.payload["error"] == run.error
    assert task.last_failure_error
    assert trip.payload["retry_status"] == lane
    assert task.worker_runs_used == 1
    assert (any(e.kind == "run_budget_exhausted" for e in kb.list_events(board, tid))) == (cap == 1)
    if cap == 1:
        assert kb.edit_task(board, tid, max_worker_runs=2)
    assert kb.recompute_ready(board) == 0
    assert kb.unblock_task(board, tid)
    assert kb.get_task(board, tid).status == lane


@pytest.mark.parametrize("lane", ["ready", "review"])
@pytest.mark.parametrize("failure", ["spawn", "workspace"])
@pytest.mark.parametrize("recovery", ["raise", "clear"])
def test_dispatch_failure_exhausts_both_budgets_and_requires_owner_admission(
    board, monkeypatch, lane, failure, recovery,
):
    tid = kb.create_task(board, title="dispatch failure", assignee="worker",
                         max_worker_runs=1, max_retries=1)
    if lane == "review":
        assert kb.request_review(board, tid, reviewer="worker", force=True)
    calls = []

    def fail(*args, **kwargs):
        calls.append(failure)
        raise RuntimeError("fixture: dispatch launch refused")

    with monkeypatch.context() as patcher:
        if failure == "workspace":
            patcher.setattr(dispatch._kbw, "resolve_workspace", fail)
        result = dispatch.dispatch_once(board, spawn_fn=fail)
    assert calls == [failure]
    assert result.auto_blocked == [tid]
    task = kb.get_task(board, tid)
    assert task is not None
    assert task.status == "blocked"
    assert task.claim_lock is task.current_run_id is task.worker_pid is None
    assert task.worker_runs_used == task.consecutive_failures == 1
    events = kb.list_events(board, tid)
    budget = [e for e in events if e.kind == "run_budget_exhausted"]
    assert len(budget) == 1
    assert budget[0].payload is not None
    assert "run-budget-exhausted: 1/1" in budget[0].payload["reason"]
    trip = next(e for e in events if e.kind == "gave_up")
    assert trip.payload is not None
    assert trip.payload["retry_status"] == budget[0].payload["resume_status"] == lane
    assert trip.payload["error"] == task.last_failure_error
    assert not kb.unblock_task(board, tid)
    assert kb.recompute_ready(board) == 0
    for _ in range(2):
        assert dispatch.dispatch_once(board, spawn_fn=fail).spawned == []
    assert calls == [failure]
    assert len([e for e in kb.list_events(board, tid) if e.kind == "run_budget_exhausted"]) == 1

    edit = {"max_worker_runs": 2} if recovery == "raise" else {"clear_max_worker_runs": True}
    assert kb.edit_task(board, tid, **edit)
    # The budget hold remains independent of the breaker after an owner edit.
    assert kb.assign_task(board, tid, "worker")  # resets consecutive_failures only
    assert kb.get_task(board, tid).status == "blocked"
    assert kb.recompute_ready(board) == 0
    assert kb.unblock_task(board, tid)
    assert kb.get_task(board, tid).status == lane
    spawned = []
    tick(board, spawned)
    assert len(spawned) == 1
    assert len(kb.list_runs(board, tid)) == kb.get_task(board, tid).worker_runs_used == 2


def test_exhaustion_diagnostic_preserves_existing_blocker_and_is_idempotent(board):
    tid = kb.create_task(board, title="needs credentials", max_worker_runs=1)
    assert kb.claim_task(board, tid)
    assert kb.block_task(board, tid, reason="owner must supply credentials", kind="needs_input")
    blocked = [e for e in kb.list_events(board, tid) if e.kind == "blocked"]
    assert len(blocked) == 1
    assert blocked[0].payload is not None
    assert blocked[0].payload["reason"] == "owner must supply credentials"
    assert kb.get_task(board, tid).block_kind == "needs_input"
    assert len([e for e in kb.list_events(board, tid) if e.kind == "run_budget_exhausted"]) == 1
    for _ in range(2):
        kb.add_comment(board, tid, author="owner", body="still waiting")
        assert kb.claim_task(board, tid) is None
    assert len([e for e in kb.list_events(board, tid) if e.kind == "run_budget_exhausted"]) == 1
    assert [e.payload for e in kb.list_events(board, tid) if e.kind == "blocked"] == [blocked[0].payload]


@pytest.mark.parametrize("transition", ["done", "scheduled"])
def test_breaker_accounting_does_not_overwrite_intervening_owner_transition(board, monkeypatch, transition):
    tid = kb.create_task(board, title="owner transition", max_worker_runs=1, max_retries=1)
    assert kb.claim_task(board, tid)
    dispatch._set_worker_pid(board, tid, 99999999)
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _: ("nonzero_exit", 1))
    sweep = dispatch._reclaim_dead_workers(board)
    if transition == "done":
        assert kb.complete_task(board, tid, summary="owner verified completion independently", force=True)
    else:
        assert kb.schedule_task(board, tid)
    dispatch._account_crashes(board, sweep.crash_details)
    task = kb.get_task(board, tid)
    assert task is not None
    assert task.status == transition
    assert task.consecutive_failures == 0
    assert task.last_failure_error is None


def test_review_resume_survives_cap_clear_between_reclaim_and_accounting(board, monkeypatch):
    tid = kb.create_task(board, title="review owner edit", max_worker_runs=1, max_retries=1)
    assert kb.request_review(board, tid, reviewer="worker", force=True)
    assert kb.claim_review_task(board, tid)
    dispatch._set_worker_pid(board, tid, 99999999)
    monkeypatch.setattr(dispatch, "_classify_worker_exit", lambda _: ("nonzero_exit", 1))
    sweep = dispatch._reclaim_dead_workers(board)
    assert kb.edit_task(board, tid, clear_max_worker_runs=True)
    dispatch._account_crashes(board, sweep.crash_details)
    assert kb.unblock_task(board, tid)
    task = kb.get_task(board, tid)
    assert task is not None and task.status == "review"

"""Rejected acceptance is a durable dispatch hold, not an error-text heuristic."""
import argparse

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_pr_acceptance_store as store


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kbc.init_db()
    with kbc.connect() as conn:
        yield conn


def rejected(board):
    tid = kb.create_task(board, title="Publication", assignee="default",
                         workspace_kind="scratch", completion_contract="example/widgets")
    assert not kb.complete_task(board, tid, result="done")
    return tid


def tick(board):
    spawned = []
    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return None
    dispatch.dispatch_once(board, spawn_fn=spawn, max_spawn=1, reconcile_orphans=False)
    return spawned


def test_rejected_completion_without_worker_error_cannot_spawn(board):
    tid = rejected(board)
    assert tick(board) == []
    assert tick(board) == []
    task = kb.get_task(board, tid)
    assert task.status == "ready" and task.acceptance_hold
    assert task.last_failure_error is None
    assert dispatch.check_respawn_guard(board, tid) == "acceptance_rejected"
    assert store.clear_acceptance_hold(board, tid)
    assert kb.get_task(board, tid).current_run_id is None
    assert tick(board) == [tid]


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_hold_survives_rate_limit_bypass_and_blocks_claim(board, monkeypatch, lane):
    tid = rejected(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status=? WHERE id=?", (lane, tid))
        kb._synthesize_ended_run(board, tid, outcome="rate_limited")
    monkeypatch.setattr(kb, "_resolve_rate_limit_cooldown_seconds", lambda: 0)
    assert dispatch.check_respawn_guard(board, tid, lane=lane) == "acceptance_rejected"
    claim = kb.claim_review_task if lane == "review" else kb.claim_task
    assert claim(board, tid) is None
    assert tick(board) == []
    assert kb.get_task(board, tid).status == lane


def test_stale_receipt_cannot_reinstall_or_clear_operator_decision(board):
    tid = rejected(board)
    stale = store.prepare_acceptance(board, tid, None, None)
    assert store.clear_acceptance_hold(board, tid)
    with kb.write_txn(board):
        assert not store.record_acceptance(board, tid, stale)
    assert not kb.get_task(board, tid).acceptance_hold
    assert not kb.complete_task(board, tid, result="retry")
    with kb.write_txn(board):
        assert not store.record_acceptance(board, tid, (stale[0], {"ok": True}))
    assert kb.get_task(board, tid).acceptance_hold


def test_success_clears_acceptance_not_worker_failure(board):
    tid = rejected(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET last_failure_error='401 auth failed' WHERE id=?", (tid,))
        assert store.record_acceptance(board, tid, (store._snapshot(board, tid), {"ok": True}))
    assert not kb.get_task(board, tid).acceptance_hold
    assert dispatch.check_respawn_guard(board, tid) == "blocker_auth"


def test_failed_transaction_rolls_back_hold_and_revision(board):
    tid = kb.create_task(board, title="Atomic", completion_contract="example/widgets")
    prepared = store.prepare_acceptance(board, tid, None, None)
    with pytest.raises(RuntimeError):
        with kb.write_txn(board):
            assert not store.record_acceptance(board, tid, prepared)
            raise RuntimeError("abort")
    assert store._snapshot(board, tid) == prepared[0]
    assert not kb.get_task(board, tid).acceptance_hold


def test_operator_cli_clears_only_acceptance_and_workers_cannot(board, monkeypatch):
    from hermes_cli import kanban as cli
    tid = rejected(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET last_failure_error='401 auth failed' WHERE id=?", (tid,))
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["kanban", "unblock", tid, "--acceptance-only"])
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    assert cli.kanban_command(args) != 0
    assert kb.get_task(board, tid).acceptance_hold
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert cli.kanban_command(args) == 0
    assert not kb.get_task(board, tid).acceptance_hold
    assert kb.get_task(board, tid).status == "ready"
    assert dispatch.check_respawn_guard(board, tid) == "blocker_auth"
    assert cli.kanban_command(args) != 0


def test_running_worker_and_review_handoff_do_not_clear_hold(board):
    tid = kb.create_task(board, title="Running", assignee="default", completion_contract="example/widgets")
    claimed = kb.claim_task(board, tid)
    assert not kb.complete_task(board, tid, result="done", expected_run_id=claimed.current_run_id)
    task = kb.get_task(board, tid)
    assert task.status == "running" and task.current_run_id == claimed.current_run_id
    assert kb.request_review(board, tid, summary="review", reviewer="reviewer", expected_run_id=claimed.current_run_id)
    assert kb.get_task(board, tid).acceptance_hold
    assert kb.reopen_review_task(board, tid)
    assert kb.get_task(board, tid).acceptance_hold
    assert tick(board) == []


def test_normal_unblock_does_not_implicitly_clear_acceptance(board):
    tid = rejected(board)
    assert kb.block_task(board, tid, reason="repair")
    assert kb.unblock_task(board, tid)
    assert kb.get_task(board, tid).acceptance_hold
    assert tick(board) == []


def test_migration_is_inert_for_legacy_cards(board):
    tid = kb.create_task(board, title="Legacy")
    with kb.write_txn(board):
        board.execute("ALTER TABLE tasks DROP COLUMN acceptance_hold")
        board.execute("ALTER TABLE tasks DROP COLUMN acceptance_revision")
    kbc.init_db()
    assert not kb.get_task(board, tid).acceptance_hold
    assert store._snapshot(board, tid)[3] == 0
    kbc.init_db()
    assert not kb.get_task(board, tid).acceptance_hold

"""Acceptance rejection must not erase the worker failure that holds a card."""
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "board"))
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kbc.init_db()
    with kbc.connect() as conn:
        yield conn


@pytest.mark.parametrize("failure", ["401 auth failed", "quota exhausted"])
def test_rejected_completion_preserves_auth_hold_across_dispatch(board, failure):
    tid = kb.create_task(board, title="Held publication", assignee="default",
                         workspace_kind="scratch",
                         completion_contract="https://github.com/example/widgets/pull/1")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (failure, tid))
    spawned = []
    def spawn(task, workspace, board=None):
        spawned.append(task.id)
        return None
    dispatch.dispatch_once(board, spawn_fn=spawn, max_spawn=1, reconcile_orphans=False)
    assert spawned == []
    assert dispatch.check_respawn_guard(board, tid) == "blocker_auth"
    assert kb.complete_task(board, tid, summary="Rejected", result="Rejected",
                            metadata={"published_pr": "https://github.com/other/repo/pull/2"}) is False
    dispatch.dispatch_once(board, spawn_fn=spawn, max_spawn=1, reconcile_orphans=False)
    assert spawned == [], "rejected completion released the pre-existing auth hold"
    task = kb.get_task(board, tid)
    assert task.status == "ready"
    assert task.last_failure_error == failure
    receipt = json.loads(board.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance' ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()[0])
    assert receipt["ok"] is False
    assert receipt["classification"] == "missing"
    assert dispatch.check_respawn_guard(board, tid) == "blocker_auth"
    # Recovery remains the existing explicit operator path, not a side effect
    # of a rejected publication.
    assert kb.block_task(board, tid, reason="Credentials repaired")
    assert kb.unblock_task(board, tid)
    assert dispatch.check_respawn_guard(board, tid) is None


@pytest.mark.parametrize("previous", [None, "", "ordinary worker failure"])
def test_rejection_keeps_diagnostics_without_replacing_worker_error(board, previous):
    tid = kb.create_task(board, title="Rejected", completion_contract="example/widgets")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (previous, tid))
    assert not kb.complete_task(board, tid, result="done")
    error = kb.get_task(board, tid).last_failure_error
    assert error == previous if previous else error.startswith("PR acceptance missing:")
    assert kb.get_task(board, tid).status == "ready"


def test_success_and_stale_receipt_keep_existing_ownership_rules(board):
    from hermes_cli.kanban_pr_acceptance_store import record_acceptance

    tid = kb.create_task(board, title="Owned", completion_contract="example/widgets")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", ("401 auth failed", tid))
        snapshot = (None, "ready", "example/widgets")
        assert record_acceptance(board, tid, (snapshot, {"ok": True}))
        assert not record_acceptance(board, tid, ((999, "ready", "example/widgets"), {"ok": False}))
    assert kb.get_task(board, tid).last_failure_error == "401 auth failed"
    assert board.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 1


def test_failed_receipt_write_rolls_back(board):
    from hermes_cli.kanban_pr_acceptance_store import record_acceptance

    tid = kb.create_task(board, title="Atomic", completion_contract="example/widgets")
    with pytest.raises(RuntimeError, match="abort"):
        with kb.write_txn(board):
            record_acceptance(board, tid, ((None, "ready", "example/widgets"), {
                "ok": False, "classification": "missing", "recovery": "retry",
            }))
            raise RuntimeError("abort")
    assert kb.get_task(board, tid).last_failure_error is None
    assert board.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0

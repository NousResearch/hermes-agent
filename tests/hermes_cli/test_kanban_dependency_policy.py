"""CP46: successful parents, not terminal cleanup, release every dispatch path."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_transfer as transfer
from hermes_cli.kanban_diagnostics import _rule_running_with_open_parents


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    yield conn
    conn.close()


def status(conn, tid, value=None):
    if value is not None:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (value, tid))
    return kb.get_task(conn, tid).status


def pair(conn, parent_status="archived"):
    parent = kb.create_task(conn, title="parent", assignee="writer")
    status(conn, parent, parent_status)
    child = kb.create_task(conn, title="child", assignee="writer", parents=[parent])
    return parent, child


# 1: empty dependencies are vacuously satisfied.
def test_no_parent_is_ready(board):
    tid = kb.create_task(board, title="root", assignee="writer")
    assert status(board, tid) == "ready"
    assert kb.claim_task(board, tid) is not None


# 2: creation uses success rather than terminal status, including custom workflows.
@pytest.mark.parametrize("parent_status", ["done", "archived", "blocked", "running", "review", "todo", "ready", "triage", "scheduled"])
def test_create_success_only(board, parent_status):
    parent, child = pair(board, parent_status)
    assert status(board, child) == ("ready" if parent_status == "done" else "todo")
    assert kb.unsatisfied_parents(board, child) == ([] if parent_status == "done" else [(parent, parent_status)])


# 3: all parents, not any parent, must succeed.
def test_fan_in_waits_for_every_parent(board):
    first, child = pair(board, "done")
    second = kb.create_task(board, title="second", assignee="writer")
    kb.link_tasks(board, second, child)
    status(board, second, "archived")
    assert kb.recompute_ready(board) == 0
    assert status(board, child) == "todo"
    status(board, second, "done")
    assert kb.recompute_ready(board) == 1
    assert status(board, child) == "ready"


# 4: recompute must not auto-unblock sticky human blocks or breaker holds.
@pytest.mark.parametrize("kind", ["needs_input", "capability", "transient"])
def test_recompute_preserves_explicit_blocks(board, kind):
    parent, child = pair(board, "done")
    assert kb.block_task(board, child, reason="operator needed", kind=kind)
    kb.recompute_ready(board)
    assert status(board, child) == "blocked"


# 5: final claim checks defeat stale readiness (TOCTOU).
@pytest.mark.parametrize("phase", ["ready", "review"])
def test_claim_regates_under_write_lock(board, phase):
    parent, child = pair(board, "done")
    status(board, child, phase)
    status(board, parent, "archived")
    claim = kb.claim_review_task if phase == "review" else kb.claim_task
    assert claim(board, child) is None
    assert status(board, child) == "todo"
    assert board.execute("SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (child,)).fetchone()[0] == 0
    status(board, parent, "done")
    kb.recompute_ready(board)
    assert status(board, child) == phase
    assert claim(board, child) is not None


# 6: manual unblock cannot override dependencies, and preserves review intent.
@pytest.mark.parametrize("phase", ["ready", "review"])
def test_manual_unblock_regates(board, phase):
    parent, child = pair(board, "done")
    status(board, child, phase)
    claim = kb.claim_review_task if phase == "review" else kb.claim_task
    assert claim(board, child)
    assert kb.block_task(board, child, reason="input", kind="needs_input")
    status(board, parent, "archived")
    assert kb.unblock_task(board, child)
    assert status(board, child) == "todo"
    kb.recompute_ready(board)
    assert status(board, child) == "todo"
    status(board, parent, "done")
    kb.recompute_ready(board)
    assert status(board, child) == phase


# 7: explicit promotion has the same gate, rechecked inside its transaction.
def test_promotion_regates_after_preview(board, monkeypatch):
    parent, child = pair(board, "archived")
    assert kb.promote_task(board, child, actor="operator")[0] is False
    status(board, parent, "done")
    real_txn = kb.write_txn

    @contextmanager
    def reopen_before_lock(conn, **kwargs):
        with real_txn(conn):
            conn.execute("UPDATE tasks SET status = 'archived' WHERE id = ?", (parent,))
        with real_txn(conn, **kwargs):
            yield

    monkeypatch.setattr(kb, "write_txn", reopen_before_lock)
    assert kb.promote_task(board, child, actor="operator")[0] is False
    assert status(board, child) == "todo"


# 8: link and unlink respect archived gating but allow an explicit edge removal.
def test_link_and_unlink_archived_parent(board):
    parent, child = pair(board, "done")
    kb.unlink_tasks(board, parent, child)
    status(board, parent, "archived")
    kb.link_tasks(board, parent, child)
    assert status(board, child) == "todo"
    kb.unlink_tasks(board, parent, child)
    kb.recompute_ready(board)
    assert status(board, child) == "ready"


# 9: completion and review handoff cannot bypass an archived parent, even forced.
def test_terminal_handoffs_refuse_unsuccessful_parent(board):
    parent, child = pair(board, "done")
    claimed = kb.claim_task(board, child)
    status(board, parent, "archived")
    assert not kb.complete_task(board, child, summary="work", force=True)
    assert not kb.request_review(board, child, summary="work", force=True)
    assert kb.get_task(board, child).current_run_id == claimed.current_run_id
    assert status(board, child) == "running"


# 10: dependency block remains a wait when the parent was archived unsuccessfully.
def test_dependency_block_archived_parent(board):
    parent, child = pair(board, "done")
    assert kb.claim_task(board, child)
    status(board, parent, "archived")
    assert kb.block_task(board, child, reason="waiting", kind="dependency")
    assert status(board, child) == "todo"
    assert kb.get_task(board, child).block_recurrences == 0


# 11: every recovery writer gates before publishing a runnable status.
@pytest.mark.parametrize("recovery", ["ttl", "manual", "spawn", "crash", "timeout", "stale", "orphan"])
@pytest.mark.parametrize("phase", ["ready", "review"])
def test_recovery_paths_regate_and_restore_phase(board, monkeypatch, recovery, phase):
    parent, child = pair(board, "done")
    status(board, child, phase)
    claim = kb.claim_review_task if phase == "review" else kb.claim_task
    claimed = claim(board, child)
    status(board, parent, "archived")
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(dispatch, "_worker_alive", lambda *args: False)
    monkeypatch.setattr(dispatch, "_poll_worker_exit", lambda *args: True)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    now = int(time.time())
    with kb.write_txn(board):
        board.execute(
            "UPDATE tasks SET claim_expires = ?, worker_pid = ?, max_runtime_seconds = 1, "
            "last_heartbeat_at = NULL, "
            "claim_lock = CASE WHEN ? = 'orphan' THEN NULL ELSE claim_lock END WHERE id = ?",
            (now - 100, 987654, recovery, child),
        )
        board.execute("UPDATE task_runs SET started_at = ? WHERE id = ?", (now - 10000, claimed.current_run_id))
    signal_fn = lambda *args: None
    actions = {
        "ttl": lambda: kb.release_stale_claims(board, signal_fn=signal_fn),
        "manual": lambda: kb.reclaim_task(board, child, signal_fn=signal_fn),
        "spawn": lambda: dispatch._record_task_failure(
            board, child, error="spawn failed", outcome="spawn_failed", failure_limit=5,
            release_claim=True, end_run=True,
        ),
        "crash": lambda: dispatch.detect_crashed_workers(board),
        "timeout": lambda: dispatch.enforce_max_runtime(board, signal_fn=signal_fn),
        "stale": lambda: dispatch.detect_stale_running(
            board, stale_timeout_seconds=1, signal_fn=signal_fn,
        ),
        "orphan": lambda: dispatch.reconcile_orphaned_running(board),
    }
    actions[recovery]()
    assert status(board, child) == "todo"
    assert kb.get_task(board, child).current_run_id is None
    assert board.execute("SELECT ended_at FROM task_runs WHERE id = ?", (claimed.current_run_id,)).fetchone()[0]
    status(board, parent, "done")
    kb.recompute_ready(board)
    assert status(board, child) == phase


# 12: breaker holds survive dependency parking and require manual intervention.
def test_recovery_breaker_still_holds(board):
    parent, child = pair(board, "done")
    assert kb.claim_task(board, child)
    status(board, parent, "archived")
    assert dispatch._record_task_failure(board, child, error="failed", outcome="spawn_failed",
                                         failure_limit=1, release_claim=True, end_run=True)
    assert status(board, child) == "blocked"
    status(board, parent, "done")
    kb.recompute_ready(board, failure_limit=1)
    assert status(board, child) == "blocked"
    assert kb.unblock_task(board, child)
    assert status(board, child) == "ready"


# 13: restart/connect plus reconciliation retains archived dependency state.
def test_restart_rehydrates_dependency_wait(board):
    parent, child = pair(board, "done")
    claimed = kb.claim_review_task(board, child)  # wrong phase must not claim
    assert claimed is None
    status(board, child, "review")
    assert kb.claim_review_task(board, child)
    status(board, parent, "archived")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET claim_lock = NULL WHERE id = ?", (child,))
    db_path = board.execute("PRAGMA database_list").fetchone()[2]
    with kbc.connect(Path(db_path)) as restarted:
        assert child in dispatch.reconcile_orphaned_running(restarted)
        assert status(restarted, child) == "todo"
        kb.recompute_ready(restarted)
        assert status(restarted, child) == "todo"
        status(restarted, parent, "done")
        kb.recompute_ready(restarted)
        assert status(restarted, child) == "review"


# 14: transfer uses the same policy with a real SQLite snapshot.
@pytest.mark.parametrize("phase", ["ready", "review"])
def test_transfer_regates_snapshot(board, tmp_path, phase):
    parent, child = pair(board, "done")
    status(board, child, phase)
    claim = kb.claim_review_task if phase == "review" else kb.claim_task
    assert claim(board, child)
    status(board, parent, "archived")
    with sqlite3.connect(str(tmp_path / "snapshot.db")) as snapshot:
        snapshot.row_factory = sqlite3.Row
        board.backup(snapshot)
        transfer._scrub_local_state(snapshot)
        snapshot.commit()
        assert status(snapshot, child) == "todo"
        assert kb.get_task(snapshot, child).claim_lock is None
        status(snapshot, parent, "done")
        kb.recompute_ready(snapshot)
        assert status(snapshot, child) == phase


# 15: diagnostics report the same unsuccessful parent the kernel refuses.
def test_diagnostics_include_archived_parent(board):
    parent, child = pair(board, "archived")
    diagnostics = _rule_running_with_open_parents(
        {"id": child, "status": "running"}, [], [], int(time.time()),
        {"_graph": {"parents": [{"id": parent, "status": "archived"}]}},
    )
    assert diagnostics[0].data["open_parents"] == [{"id": parent, "status": "archived"}]


# 16: archival is not success even if the parent used to be completed.
def test_archiving_completed_parent_does_not_release_new_child(board):
    parent = kb.create_task(board, title="completed", assignee="writer")
    assert kb.complete_task(board, parent, summary="finished")
    assert kb.archive_task(board, parent)
    child = kb.create_task(board, title="child", assignee="writer", parents=[parent])
    kb.recompute_ready(board)
    assert status(board, child) == "todo"
    assert kb.claim_task(board, child) is None


@pytest.mark.parametrize("phase", ["ready", "review", "running"])
def test_archiving_completed_parent_regates_existing_runnable_child(board, phase):
    parent, child = pair(board, "done")
    if phase == "running":
        assert kb.claim_task(board, child) is not None
    else:
        status(board, child, phase)

    assert kb.archive_task(board, parent)
    assert status(board, child) == "todo"
    assert kb.claim_task(board, child) is None
    waits = [event for event in kb.list_events(board, child) if event.kind == "dependency_wait"]
    assert waits[-1].payload["parent"] == parent
    assert waits[-1].payload["source_status"] == phase

    status(board, parent, "done")
    assert kb.recompute_ready(board) == 1
    assert status(board, child) == ("review" if phase == "review" else "ready")


def test_archiving_completed_parent_regates_transitive_descendants(board):
    parent = kb.create_task(board, title="parent", assignee="writer")
    status(board, parent, "done")
    intermediate = kb.create_task(board, title="intermediate", assignee="writer", parents=[parent])
    status(board, intermediate, "done")
    descendant = kb.create_task(board, title="descendant", assignee="writer", parents=[intermediate])
    assert kb.claim_task(board, descendant) is not None

    assert kb.archive_task(board, parent)
    assert status(board, intermediate) == "todo"
    assert status(board, descendant) == "todo"
    assert kb.claim_task(board, descendant) is None
    waits = [event for event in kb.list_events(board, descendant) if event.kind == "dependency_wait"]
    assert waits[-1].payload["parent"] == parent
    assert waits[-1].payload["source_status"] == "running"

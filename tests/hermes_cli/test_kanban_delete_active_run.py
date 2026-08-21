"""Stranded-worker guard on the ARCHIVED-task purge path.

Sibling commit fa3efdcc7 guarded ``archive_task`` / ``delete_task`` (and the
``--rm`` CLI loop's try/except around ``delete_archived_task``), but
``delete_archived_task`` itself never raised the refusal — the ``--rm`` purge
path (the exact RV2 post-verification cleanup that deleted running children
mid-run) was unguarded. These tests close that gap:

* ``delete_archived_task`` refuses while a run is active,
* stale terminal-status runs never block a delete,
* the CLI ``archive --rm`` loop skips the in-flight card and keeps going.
"""

from __future__ import annotations

import argparse
import time

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed_archived_with_active_run(kanban_home):
    """Create + claim a card, then force it to archived while its run is
    still active — the RV2 incident shape (post-verification cleanup trying
    to purge children whose worker runs are still in flight)."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="archived but running", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = int(
            conn.execute(
                "SELECT current_run_id FROM tasks WHERE id = ?", (tid,)
            ).fetchone()["current_run_id"]
        )
        conn.execute(
            "UPDATE tasks SET status = 'archived' WHERE id = ?", (tid,)
        )
        conn.commit()
    return tid, run_id


def test_delete_archived_task_refused_while_run_active(kanban_home):
    tid, run_id = _seed_archived_with_active_run(kanban_home)

    with kb.connect() as conn:
        with pytest.raises(kb.TaskHasActiveRunError) as ei:
            kb.delete_archived_task(conn, tid)
        assert ei.value.task_id == tid
        assert ei.value.run_id == run_id
        # Card + run untouched.
        assert kb.get_task(conn, tid) is not None
        assert (
            conn.execute(
                "SELECT status FROM task_runs WHERE id = ?", (run_id,)
            ).fetchone()["status"]
            == "running"
        )


def test_delete_archived_task_succeeds_after_run_ends(kanban_home):
    tid, run_id = _seed_archived_with_active_run(kanban_home)
    with kb.connect() as conn:
        kb._end_run(conn, tid, outcome="completed", status="done")
    with kb.connect() as conn:
        assert kb.delete_archived_task(conn, tid) is True
        assert kb.get_task(conn, tid) is None


def test_delete_archived_task_clears_task_links(kanban_home):
    """Deleting an archived card with no active run still cleans links."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent], assignee="worker")
        kb.complete_task(conn, child, result="ok")
        assert kb.archive_task(conn, child)
        assert kb.delete_archived_task(conn, child) is True
        leftover = conn.execute(
            "SELECT COUNT(*) FROM task_links "
            "WHERE parent_id = ? OR child_id = ?",
            (child, child),
        ).fetchone()[0]
        assert leftover == 0


def test_terminal_run_statuses_do_not_block_delete(kanban_home):
    """Stale runs in ANY terminal status must not block a delete — only
    'running' counts as active. Guards against an over-broad check."""
    for terminal in ("done", "blocked", "crashed", "timed_out", "failed", "released"):
        with kb.connect() as conn:
            tid = kb.create_task(conn, title=f"stale {terminal}", assignee="worker")
            cur = conn.execute(
                "INSERT INTO task_runs (task_id, status, started_at, ended_at) "
                "VALUES (?, ?, ?, ?)",
                (tid, terminal, int(time.time()), int(time.time())),
            )
            run_id = int(cur.lastrowid)
            conn.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?",
                (run_id, tid),
            )
            conn.commit()

        with kb.connect() as conn:
            assert kb.delete_task(conn, tid) is True, f"terminal={terminal}"
            assert kb.get_task(conn, tid) is None


def test_cli_archive_purge_loop_skips_active_run(capsys, monkeypatch, kanban_home):
    """The CLI ``hermes kanban archive --rm <ids>`` loop must skip a card
    whose run is still active and continue the batch — not abort."""
    from hermes_cli.kanban import _cmd_archive

    active_tid, _ = _seed_archived_with_active_run(kanban_home)
    with kb.connect() as conn:
        clean_tid = kb.create_task(conn, title="clean", assignee="worker")
        kb.complete_task(conn, clean_tid, result="ok")
        assert kb.archive_task(conn, clean_tid)

    args = argparse.Namespace(task_ids=[], purge_ids=[active_tid, clean_tid])
    rc = _cmd_archive(args)

    captured = capsys.readouterr()
    # Refusal is tracked (exit 1 signals retry needed) but the batch
    # continues: the clean card still got deleted.
    assert rc == 1
    assert active_tid in captured.err
    assert "active worker run" in captured.err or "active run" in captured.err
    assert f"Deleted {clean_tid}" in captured.out
    with kb.connect() as conn:
        assert kb.get_task(conn, clean_tid) is None
        assert kb.get_task(conn, active_tid) is not None

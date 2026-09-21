"""
W1 Batch 2 V2 - Atomic cross-board move tests.

Behavioural tests exercising the real move_task_atomic() helper and
live CLI dispatch against temporary SQLite board databases.  Crash and
concurrency tests use subprocess isolation.  No source-text/change-detector
tests; no env-var failpoints.
"""

from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_boards(tmp_path, monkeypatch):
    """Create two real board DBs under an isolated HERMES_HOME."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    kb.init_db(board="default")
    kb.init_db(board="target")

    src_path = kb.kanban_db_path(board="default")
    tgt_path = kb.kanban_db_path(board="target")
    assert src_path != tgt_path
    assert src_path.exists()
    assert tgt_path.exists()
    return {
        "home": home,
        "src_path": src_path,
        "tgt_path": tgt_path,
        "src_board": "default",
        "tgt_board": "target",
    }


def _create_task(board, title="test-task", **kwargs):
    with kb.connect_closing(board=board) as conn:
        return kb.create_task(conn, title=title, **kwargs)


def _get_task(board, task_id):
    with kb.connect_closing(board=board) as conn:
        return kb.get_task(conn, task_id)


def _count_rows(board, table, task_id=None):
    with kb.connect_closing(board=board) as conn:
        if task_id:
            r = conn.execute(
                "SELECT COUNT(*) FROM " + table + " WHERE task_id = ?", (task_id,)
            ).fetchone()
        else:
            r = conn.execute("SELECT COUNT(*) FROM " + table).fetchone()
        return r[0]


def _journal_mode(path):
    conn = sqlite3.connect(str(path))
    r = conn.execute("PRAGMA journal_mode").fetchone()
    conn.close()
    return r[0]


def _fk_check(path):
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON")
    r = conn.execute("PRAGMA foreign_key_check").fetchall()
    conn.close()
    return r


# ---------------------------------------------------------------------------
# Success: full 7-table transfer
# ---------------------------------------------------------------------------

class TestMoveSuccess:
    def test_move_transfers_task_row(self, two_boards):
        tid = _create_task("default", title="ship feature", assignee="alice")
        rc = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert rc["status"] == "moved"
        assert _get_task("target", tid) is not None
        assert _get_task("default", tid) is None

    def test_move_transfers_events(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            kb._append_event(conn, tid, "created", None)
            conn.commit()
        src_evts = _count_rows("default", "task_events", tid)
        assert src_evts >= 1
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert _count_rows("target", "task_events", tid) >= src_evts
        assert _count_rows("default", "task_events", tid) == 0

    def test_move_transfers_comments(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (tid, "bob", "looks good", int(time.time())),
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert _count_rows("target", "task_comments", tid) == 1
        assert _count_rows("default", "task_comments", tid) == 0

    def test_move_transfers_runs_and_remaps_event_run_id(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            now = int(time.time())
            cur = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, "default", "done", now, now),
            )
            run_id = cur.lastrowid
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'completed', NULL, ?)",
                (tid, run_id, now),
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert _count_rows("target", "task_runs", tid) == 1
        assert _count_rows("default", "task_runs", tid) == 0
        with kb.connect_closing(board="target") as conn:
            ev = conn.execute(
                "SELECT run_id FROM task_events WHERE task_id = ? AND kind = 'completed'",
                (tid,),
            ).fetchone()
            assert ev is not None
            assert ev["run_id"] is not None
            run = conn.execute(
                "SELECT id FROM task_runs WHERE id = ? AND task_id = ?",
                (ev["run_id"], tid),
            ).fetchone()
            assert run is not None

    def test_move_transfers_attachments_metadata(self, two_boards):
        tid = _create_task("default", title="t1")
        att_dir = kb.task_attachments_dir(tid, board="default")
        att_dir.mkdir(parents=True, exist_ok=True)
        att_file = att_dir / "doc.txt"
        att_file.write_text("hello world")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO task_attachments (task_id, filename, stored_path, size, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, "doc.txt", str(att_file), len("hello world"), int(time.time())),
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert _count_rows("target", "task_attachments", tid) == 1
        assert _count_rows("default", "task_attachments", tid) == 0
        with kb.connect_closing(board="target") as conn:
            row = conn.execute(
                "SELECT stored_path FROM task_attachments WHERE task_id = ?",
                (tid,),
            ).fetchone()
            assert row is not None
            assert Path(row["stored_path"]).exists()

    def test_move_transfers_notify_subs(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, thread_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, "discord", "123", "", int(time.time())),
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert _count_rows("target", "kanban_notify_subs", tid) == 1
        assert _count_rows("default", "kanban_notify_subs", tid) == 0

    def test_move_transfers_terminal_lifecycle_approvals(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO profile_lifecycle_approvals "
                "(id, task_id, op, profile, status, token, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("appr1", tid, "create", "test-profile", "rejected", "tok", int(time.time())),
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert _count_rows("target", "profile_lifecycle_approvals", tid) == 1
        assert _count_rows("default", "profile_lifecycle_approvals", tid) == 0

    def test_move_emits_exactly_one_moved_event(self, two_boards):
        tid = _create_task("default", title="t1")
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        with kb.connect_closing(board="target") as conn:
            moved = conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'moved'",
                (tid,),
            ).fetchone()
            assert moved[0] == 1
            ev = conn.execute(
                "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'moved'",
                (tid,),
            ).fetchone()
            payload = json.loads(ev["payload"])
            assert payload["from_board"] == "default"
            assert payload["to_board"] == "target"

    def test_move_fk_integrity_both_dbs(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            now = int(time.time())
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) "
                "VALUES (?, ?, 'done', ?, ?)", (tid, "default", now, now)
            )
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)", (tid, "bob", "ok", now)
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert _fk_check(two_boards["src_path"]) == []
        assert _fk_check(two_boards["tgt_path"]) == []

    def test_move_journal_modes_restored(self, two_boards):
        tid = _create_task("default", title="t1")
        src_mode_before = _journal_mode(two_boards["src_path"])
        tgt_mode_before = _journal_mode(two_boards["tgt_path"])

        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)

        assert _journal_mode(two_boards["src_path"]) == src_mode_before
        assert _journal_mode(two_boards["tgt_path"]) == tgt_mode_before


# ---------------------------------------------------------------------------
# Rejection guards
# ---------------------------------------------------------------------------

class TestMoveRejections:
    def test_move_rejects_running_status(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"
        assert "running" in result["reason"].lower()
        assert _get_task("default", tid) is not None
        assert _get_task("target", tid) is None

    def test_move_rejects_non_null_current_run_id(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute("UPDATE tasks SET current_run_id = 42 WHERE id = ?", (tid,))
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"
        assert "current_run_id" in result["reason"].lower()

    def test_move_rejects_active_run(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            now = int(time.time())
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) "
                "VALUES (?, ?, 'running', ?)", (tid, "default", now)
            )
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"
        assert "run" in result["reason"].lower()

    def test_move_rejects_run_without_terminal_end(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            now = int(time.time())
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) "
                "VALUES (?, ?, 'done', ?, NULL)", (tid, "default", now)
            )
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"

    def test_move_rejects_pending_lifecycle_approval(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO profile_lifecycle_approvals "
                "(id, task_id, op, profile, status, token, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("appr1", tid, "create", "test-profile", "pending", "tok", int(time.time())),
            )
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"
        assert "approval" in result["reason"].lower()

    def test_move_rejects_approved_lifecycle_approval(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO profile_lifecycle_approvals "
                "(id, task_id, op, profile, status, token, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("appr1", tid, "create", "test-profile", "approved", "tok", int(time.time())),
            )
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"

    def test_move_rejects_same_path(self, two_boards):
        tid = _create_task("default", title="t1")
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["src_path"], tid)
        assert result["status"] == "rejected"
        assert "same" in result["reason"].lower() or "identical" in result["reason"].lower()

    def test_move_rejects_task_not_found(self, two_boards):
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], "nonexistent_task")
        assert result["status"] == "rejected"
        assert "not found" in result["reason"].lower() or "not on source" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

class TestMoveInvariants:
    def test_move_nulls_epic_id(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO epics (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)", ("epic1", "My Epic", int(time.time()), int(time.time()))
            )
            conn.execute("UPDATE tasks SET epic_id = ? WHERE id = ?", ("epic1", tid))
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        task = _get_task("target", tid)
        assert task.epic_id is None

    def test_move_preserves_manual_block(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute("UPDATE tasks SET status = 'blocked', block_kind = 'review' WHERE id = ?", (tid,))
            kb._append_event(conn, tid, "blocked", {"reason": "manual review"})
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        task = _get_task("target", tid)
        assert task.status == "blocked"
        assert task.block_kind == "review"

    def test_move_severs_cross_board_links(self, two_boards):
        parent = _create_task("default", title="parent")
        child = _create_task("default", title="child")
        with kb.connect_closing(board="default") as conn:
            kb.link_tasks(conn, parent, child)
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], child)
        with kb.connect_closing(board="default") as conn:
            links = conn.execute(
                "SELECT COUNT(*) FROM task_links WHERE parent_id = ? OR child_id = ?",
                (child, child),
            ).fetchone()
            assert links[0] == 0

    def test_move_preserves_intra_board_links(self, two_boards):
        t1 = _create_task("default", title="t1")
        t2 = _create_task("default", title="t2")
        with kb.connect_closing(board="default") as conn:
            kb.link_tasks(conn, t1, t2)
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], t1)
        with kb.connect_closing(board="default") as conn:
            links = conn.execute(
                "SELECT COUNT(*) FROM task_links WHERE parent_id = ? OR child_id = ?",
                (t1, t1),
            ).fetchone()
            assert links[0] == 0

    def test_move_no_source_only_visible_state(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            now = int(time.time())
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at) "
                "VALUES (?, ?, 'done', ?, ?)", (tid, "default", now, now)
            )
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)", (tid, "bob", "ok", now)
            )
            conn.execute(
                "INSERT INTO kanban_notify_subs (task_id, platform, chat_id, thread_id, created_at) "
                "VALUES (?, ?, ?, ?, ?)", (tid, "discord", "123", "", now)
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        tables = ["tasks", "task_events", "task_comments", "task_runs",
                  "task_attachments", "kanban_notify_subs", "profile_lifecycle_approvals"]
        with kb.connect_closing(board="default") as conn:
            for tbl in tables:
                if tbl == "tasks":
                    r = conn.execute("SELECT COUNT(*) FROM tasks WHERE id = ?", (tid,)).fetchone()
                else:
                    r = conn.execute("SELECT COUNT(*) FROM " + tbl + " WHERE task_id = ?", (tid,)).fetchone()
                assert r[0] == 0, "Source still has rows in " + tbl + " for " + tid

    def test_move_clears_runtime_state(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "UPDATE tasks SET claim_lock = 'lock123', claim_expires = 999, "
                "worker_pid = 42, last_heartbeat_at = 888 WHERE id = ?",
                (tid,),
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        task = _get_task("target", tid)
        assert task.claim_lock is None
        assert task.claim_expires is None
        assert task.worker_pid is None
        assert task.last_heartbeat_at is None


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestMoveIdempotency:
    def test_move_idempotent_already_moved(self, two_boards):
        tid = _create_task("default", title="t1")
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "already_moved"
        with kb.connect_closing(board="target") as conn:
            moved = conn.execute(
                "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'moved'",
                (tid,),
            ).fetchone()
            assert moved[0] == 1

    def test_move_ambiguous_duplicate_fails_closed(self, two_boards):
        tid = _create_task("default", title="t1")
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at) "
                "VALUES (?, ?, 'todo', ?)",
                (tid, "dup", int(time.time())),
            )
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

class TestMoveAttachments:
    def test_move_attachment_containment_rejects_symlink(self, two_boards):
        tid = _create_task("default", title="t1")
        att_dir = kb.task_attachments_dir(tid, board="default")
        att_dir.mkdir(parents=True, exist_ok=True)
        outside = two_boards["home"] / "outside.txt"
        outside.write_text("secret")
        link = att_dir / "link.txt"
        link.symlink_to(outside)
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO task_attachments (task_id, filename, stored_path, size, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, "link.txt", str(link), 6, int(time.time())),
            )
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"
        assert ("symlink" in result["reason"].lower()
                or "unsafe" in result["reason"].lower()
                or "not contained" in result["reason"].lower())

    def test_move_attachment_rejects_traversal_filename(self, two_boards):
        tid = _create_task("default", title="t1")
        att_dir = kb.task_attachments_dir(tid, board="default")
        att_dir.mkdir(parents=True, exist_ok=True)
        att_file = att_dir / "doc.txt"
        att_file.write_text("hello")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO task_attachments (task_id, filename, stored_path, size, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, "../../../etc/passwd", str(att_file), 5, int(time.time())),
            )
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"
        assert "traversal" in result["reason"].lower() or "unsafe" in result["reason"].lower() or "filename" in result["reason"].lower()

    def test_move_attachment_rejects_size_mismatch(self, two_boards):
        tid = _create_task("default", title="t1")
        att_dir = kb.task_attachments_dir(tid, board="default")
        att_dir.mkdir(parents=True, exist_ok=True)
        att_file = att_dir / "doc.txt"
        att_file.write_text("hello")
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO task_attachments (task_id, filename, stored_path, size, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, "doc.txt", str(att_file), 999, int(time.time())),
            )
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"
        assert "size" in result["reason"].lower()

    def test_move_attachment_hash_verification(self, two_boards):
        tid = _create_task("default", title="t1")
        att_dir = kb.task_attachments_dir(tid, board="default")
        att_dir.mkdir(parents=True, exist_ok=True)
        att_file = att_dir / "doc.txt"
        content = b"hello world"
        att_file.write_bytes(content)
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO task_attachments (task_id, filename, stored_path, size, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, "doc.txt", str(att_file), len(content), int(time.time())),
            )
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        with kb.connect_closing(board="target") as conn:
            row = conn.execute(
                "SELECT stored_path FROM task_attachments WHERE task_id = ?",
                (tid,),
            ).fetchone()
            assert row is not None
            assert Path(row["stored_path"]).read_bytes() == content


# ---------------------------------------------------------------------------
# Dependency recompute
# ---------------------------------------------------------------------------

class TestMoveDependencyRecompute:
    def test_move_source_child_promoted_after_link_severed(self, two_boards):
        parent = _create_task("default", title="parent")
        child = _create_task("default", title="child")
        with kb.connect_closing(board="default") as conn:
            kb.link_tasks(conn, parent, child)
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
            conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], parent)
        child_task = _get_task("default", child)
        assert child_task.status == "ready"

    def test_move_target_recompute_preserves_manual_block(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute("UPDATE tasks SET status = 'blocked', block_kind = 'review' WHERE id = ?", (tid,))
            kb._append_event(conn, tid, "blocked", {"reason": "manual"})
            conn.commit()
        kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        task = _get_task("target", tid)
        assert task.status == "blocked"


# ---------------------------------------------------------------------------
# Schema compatibility
# ---------------------------------------------------------------------------

class TestMoveSchema:
    def test_move_rejects_incompatible_schema(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="target") as conn:
            # SQLite cannot ALTER ADD a NOT NULL/no-default column. Rebuild a
            # task-owned table that has no dependent tables, retaining its real
            # FK while making the target schema genuinely incompatible.
            ddl = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'task_comments'"
            ).fetchone()[0]
            rebuilt = ddl.replace(
                "CREATE TABLE task_comments", "CREATE TABLE task_comments_new", 1
            ).rsplit(")", 1)[0] + ", required_field TEXT NOT NULL)"
            conn.execute(rebuilt)
            conn.execute("DROP TABLE task_comments")
            conn.execute("ALTER TABLE task_comments_new RENAME TO task_comments")
            conn.commit()
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "rejected"
        assert "schema" in result["reason"].lower() or "incompatible" in result["reason"].lower()


# ---------------------------------------------------------------------------
# Atomicity / crash injection
# ---------------------------------------------------------------------------

class TestMoveAtomicity:
    def test_move_rollback_on_target_copy_failure(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            kb._append_event(conn, tid, "created", None)
            conn.commit()
        src_event_count = _count_rows("default", "task_events", tid)

        def fail_after_copy():
            raise RuntimeError("injected: after_target_copy")

        result = kb.move_task_atomic(
            two_boards["src_path"], two_boards["tgt_path"], tid,
            _test_hooks={"after_target_copy": fail_after_copy},
        )
        assert result["status"] == "error"
        assert _get_task("default", tid) is not None
        assert _count_rows("default", "task_events", tid) == src_event_count
        assert _get_task("target", tid) is None

    def test_move_rollback_on_pre_commit_failure(self, two_boards):
        tid = _create_task("default", title="t1")

        def fail_before_commit():
            raise RuntimeError("injected: before_commit")

        result = kb.move_task_atomic(
            two_boards["src_path"], two_boards["tgt_path"], tid,
            _test_hooks={"before_commit": fail_before_commit},
        )
        assert result["status"] == "error"
        assert _get_task("default", tid) is not None
        assert _get_task("target", tid) is None


# ---------------------------------------------------------------------------
# CLI dispatch (live, not source-text)
# ---------------------------------------------------------------------------

class TestMoveCLI:
    def test_cli_move_success(self, two_boards):
        tid = _create_task("default", title="ship feature")
        out = kc.run_slash("move " + tid + " --to target")
        assert "Moved" in out or "moved" in out
        assert _get_task("target", tid) is not None
        assert _get_task("default", tid) is None

    def test_cli_move_rejection_running(self, two_boards):
        tid = _create_task("default", title="t1")
        with kb.connect_closing(board="default") as conn:
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
            conn.commit()
        out = kc.run_slash("move " + tid + " --to target")
        assert "reject" in out.lower() or "cannot" in out.lower() or "running" in out.lower()
        assert _get_task("default", tid) is not None

    def test_cli_move_idempotent_retry(self, two_boards):
        tid = _create_task("default", title="t1")
        out1 = kc.run_slash("move " + tid + " --to target")
        assert "Moved" in out1 or "moved" in out1
        # The first CLI move switches the active board to target. Name the
        # original source explicitly so this exercises idempotent recovery,
        # rather than the separate same-path rejection guard.
        out2 = kc.run_slash("--board default move " + tid + " --to target")
        assert "already" in out2.lower()

    def test_cli_move_error_not_found(self, two_boards):
        out = kc.run_slash("move nonexistent_task --to target")
        assert "not found" in out.lower() or "error" in out.lower()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestMoveConcurrency:
    def test_move_both_orders_concurrently(self, two_boards):
        t1 = _create_task("default", title="t1")
        t2 = _create_task("target", title="t2")
        results = {}

        def move_default_to_target():
            results["d2t"] = kb.move_task_atomic(
                two_boards["src_path"], two_boards["tgt_path"], t1
            )

        def move_target_to_default():
            results["t2d"] = kb.move_task_atomic(
                two_boards["tgt_path"], two_boards["src_path"], t2
            )

        th1 = threading.Thread(target=move_default_to_target)
        th2 = threading.Thread(target=move_target_to_default)
        th1.start()
        th2.start()
        th1.join(timeout=30)
        th2.join(timeout=30)
        assert results["d2t"]["status"] == "moved"
        assert results["t2d"]["status"] == "moved"
        assert _get_task("target", t1) is not None
        assert _get_task("default", t2) is not None


# ---------------------------------------------------------------------------
# Crash recovery (subprocess SIGKILL)
# ---------------------------------------------------------------------------

class TestMoveCrashRecovery:
    def _make_crash_script(self, src, tgt, tid, hook_name):
        lines = [
            "import signal, os",
            "from hermes_cli import kanban_db as kb",
            "src = " + repr(str(src)),
            "tgt = " + repr(str(tgt)),
            "tid = " + repr(tid),
            "def crash_hook():",
            "    os.kill(os.getpid(), signal.SIGKILL)",
            "kb.move_task_atomic(src, tgt, tid, _test_hooks={" + repr(hook_name) + ": crash_hook})",
        ]
        return "\n".join(lines)

    def test_crash_before_commit_rolls_back(self, two_boards):
        tid = _create_task("default", title="t1")
        env = os.environ.copy()
        env["HERMES_HOME"] = str(two_boards["home"])
        script = self._make_crash_script(
            two_boards["src_path"], two_boards["tgt_path"], tid, "before_commit"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode != 0
        assert _get_task("default", tid) is not None
        assert _get_task("target", tid) is None

    def test_crash_after_commit_idempotent_recovery(self, two_boards):
        tid = _create_task("default", title="t1")
        env = os.environ.copy()
        env["HERMES_HOME"] = str(two_boards["home"])
        script = self._make_crash_script(
            two_boards["src_path"], two_boards["tgt_path"], tid, "after_commit"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode != 0
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "already_moved"
        assert _journal_mode(two_boards["src_path"]) == "wal"
        assert _journal_mode(two_boards["tgt_path"]) == "wal"

    def test_crash_attachment_orphan_retry(self, two_boards):
        tid = _create_task("default", title="t1")
        att_dir = kb.task_attachments_dir(tid, board="default")
        att_dir.mkdir(parents=True, exist_ok=True)
        att_file = att_dir / "doc.txt"
        content = b"hello world"
        att_file.write_bytes(content)
        with kb.connect_closing(board="default") as conn:
            conn.execute(
                "INSERT INTO task_attachments (task_id, filename, stored_path, size, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tid, "doc.txt", str(att_file), len(content), int(time.time())),
            )
            conn.commit()
        env = os.environ.copy()
        env["HERMES_HOME"] = str(two_boards["home"])
        script = self._make_crash_script(
            two_boards["src_path"], two_boards["tgt_path"], tid, "after_target_copy"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode != 0
        assert _get_task("default", tid) is not None
        result = kb.move_task_atomic(two_boards["src_path"], two_boards["tgt_path"], tid)
        assert result["status"] == "moved"
        assert _get_task("target", tid) is not None

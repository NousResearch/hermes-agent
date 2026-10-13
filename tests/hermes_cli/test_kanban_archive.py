"""DB/CLI behavior contract for Kanban archive/unarchive.

Covers the archive/unarchive policy from
``company/kanban-archive-design-2026-10-13.md``: archive is only ever allowed
from ``done``/``blocked``, any active (non-done/non-archived) recursive
descendant refuses the transition, archive provenance is persisted on the
task row (not just ``task_events``) so unarchive can restore it without
depending on event retention, and the CLI surfaces reason/actor plumbing and
JSON output for both verbs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _done_task(conn, title: str = "x", assignee: str = "a") -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    assert kb.claim_task(conn, tid) is not None
    assert kb.complete_task(conn, tid, summary="done")
    return tid


def _blocked_task(conn, title: str = "x") -> str:
    tid = kb.create_task(conn, title=title)
    assert kb.block_task(conn, tid, reason="waiting")
    return tid


# ---------------------------------------------------------------------------
# archive_task: accepted source statuses + provenance
# ---------------------------------------------------------------------------


def test_archive_done_task_persists_provenance_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        result = kb.archive_task(conn, tid, reason="no longer needed", actor="alex")

        assert result.ok is True
        assert result.status == "archived"
        assert result.archived_from_status == "done"
        assert result.archived_at is not None

        task = kb.get_task(conn, tid)
        assert task.status == "archived"
        assert task.archived_from_status == "done"
        assert task.archived_at == result.archived_at
        assert task.archive_reason == "no longer needed"
        assert task.archived_by == "alex"

        events = [e for e in kb.list_events(conn, tid) if e.kind == "archived"]
        assert len(events) == 1
        assert events[0].payload["from_status"] == "done"
        assert events[0].payload["reason"] == "no longer needed"
        assert events[0].payload["actor"] == "alex"


def test_archive_blocked_task_persists_provenance(kanban_home):
    with kbc.connect() as conn:
        tid = _blocked_task(conn)
        result = kb.archive_task(conn, tid, reason="duplicate_of:t_abc123", actor="ethan")

        assert result.ok is True
        task = kb.get_task(conn, tid)
        assert task.status == "archived"
        assert task.archived_from_status == "blocked"
        assert task.archive_reason == "duplicate_of:t_abc123"
        assert task.archived_by == "ethan"


def _make_triage_task(conn) -> str:
    return kb.create_task(conn, title="x", triage=True)


def _make_todo_task(conn) -> str:
    tid = kb.create_task(conn, title="x")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (tid,))
    return tid


def _make_scheduled_task(conn) -> str:
    tid = kb.create_task(conn, title="x")
    assert kb.schedule_task(conn, tid, reason="later")
    return tid


def _make_ready_task(conn) -> str:
    return kb.create_task(conn, title="x")


def _make_running_task(conn) -> str:
    tid = kb.create_task(conn, title="x")
    assert kb.claim_task(conn, tid) is not None
    return tid


def _make_review_task(conn) -> str:
    tid = kb.create_task(conn, title="x")
    assert kb.request_review(conn, tid, summary="s", force=True)
    return tid


_NON_TERMINAL_STATUS_SETUP = {
    "triage": _make_triage_task,
    "todo": _make_todo_task,
    "scheduled": _make_scheduled_task,
    "ready": _make_ready_task,
    "running": _make_running_task,
    "review": _make_review_task,
}


@pytest.mark.parametrize(
    "status", ["triage", "todo", "scheduled", "ready", "running", "review"],
)
def test_archive_refuses_every_non_terminal_status(kanban_home, status):
    with kbc.connect() as conn:
        tid = _NON_TERMINAL_STATUS_SETUP[status](conn)

        before = kb.get_task(conn, tid)
        assert before.status == status

        result = kb.archive_task(conn, tid, reason="x", actor="tester")

        assert result.ok is False
        assert result.error_code == "invalid_source_status"
        after = kb.get_task(conn, tid)
        assert after.status == status
        assert after.archived_from_status is None
        assert after.claim_lock == before.claim_lock
        assert after.current_run_id == before.current_run_id


def test_archive_requires_nonblank_reason_and_actor(kanban_home):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        with pytest.raises(ValueError):
            kb.archive_task(conn, tid, reason="  ", actor="tester")
        with pytest.raises(ValueError):
            kb.archive_task(conn, tid, reason="x", actor="")
        assert kb.get_task(conn, tid).status == "done"


def test_archive_unknown_task_is_not_found(kanban_home):
    with kbc.connect() as conn:
        result = kb.archive_task(conn, "t_doesnotexist", reason="x", actor="tester")
        assert result.ok is False
        assert result.error_code == "not_found"


# ---------------------------------------------------------------------------
# Active-descendant safety: recursive, not just direct children
# ---------------------------------------------------------------------------


def test_archive_refuses_direct_active_child(kanban_home):
    with kbc.connect() as conn:
        parent = _done_task(conn, title="parent")
        child = kb.create_task(conn, title="child")  # ready
        kb.link_tasks(conn, parent, child)

        result = kb.archive_task(conn, parent, reason="x", actor="tester")

        assert result.ok is False
        assert result.error_code == "active_descendants"
        assert (child, "ready") in result.active_descendants
        assert kb.get_task(conn, parent).status == "done"


def test_archive_refuses_active_grandchild_under_done_child(kanban_home):
    """A done parent whose child is ALSO done but the grandchild is still
    active must still be refused — the walk is recursive, not direct-only."""
    with kbc.connect() as conn:
        grandparent = _done_task(conn, title="grandparent")
        child = _done_task(conn, title="child")
        kb.link_tasks(conn, grandparent, child)
        grandchild = kb.create_task(conn, title="grandchild")  # ready
        kb.link_tasks(conn, child, grandchild)

        result = kb.archive_task(conn, grandparent, reason="x", actor="tester")

        assert result.ok is False
        assert result.error_code == "active_descendants"
        assert (grandchild, "ready") in result.active_descendants
        assert kb.get_task(conn, grandparent).status == "done"


def test_archive_succeeds_when_all_descendants_are_done_or_archived(kanban_home):
    with kbc.connect() as conn:
        parent = _done_task(conn, title="parent")
        child = _done_task(conn, title="child")
        kb.link_tasks(conn, parent, child)
        grandchild = _done_task(conn, title="grandchild")
        kb.link_tasks(conn, child, grandchild)
        assert kb.archive_task(conn, grandchild, reason="x", actor="tester").ok

        result = kb.archive_task(conn, parent, reason="x", actor="tester")

        assert result.ok is True
        assert kb.get_task(conn, parent).status == "archived"


def test_bulk_archive_is_per_task_atomic(kanban_home):
    """One refused id in a batch must not roll back another's success."""
    with kbc.connect() as conn:
        good = _done_task(conn, title="good")
        bad = kb.create_task(conn, title="bad")  # ready: refused

        good_result = kb.archive_task(conn, good, reason="x", actor="tester")
        bad_result = kb.archive_task(conn, bad, reason="x", actor="tester")

        assert good_result.ok is True
        assert bad_result.ok is False
        assert kb.get_task(conn, good).status == "archived"
        assert kb.get_task(conn, bad).status == "ready"


# ---------------------------------------------------------------------------
# Default list / dispatcher exclusion unchanged
# ---------------------------------------------------------------------------


def test_archived_task_excluded_from_default_list_included_with_flag(kanban_home):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        kb.archive_task(conn, tid, reason="x", actor="tester")

        default_ids = {t.id for t in kb.list_tasks(conn)}
        assert tid not in default_ids

        included_ids = {t.id for t in kb.list_tasks(conn, include_archived=True)}
        assert tid in included_ids

        # get_task/show always sees it by id regardless of status filters.
        assert kb.get_task(conn, tid) is not None


def test_archived_parent_is_terminal_for_recompute_ready(kanban_home):
    with kbc.connect() as conn:
        parent = _done_task(conn, title="parent")
        kb.archive_task(conn, parent, reason="x", actor="tester")
        child = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent, child)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# unarchive_task
# ---------------------------------------------------------------------------


def test_unarchive_restores_done_exactly(kanban_home):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        kb.archive_task(conn, tid, reason="x", actor="tester")

        result = kb.unarchive_task(conn, tid, reason="restore", actor="ryan")

        assert result.ok is True
        assert result.status == "done"
        task = kb.get_task(conn, tid)
        assert task.status == "done"
        # Archive provenance is preserved (new archive would overwrite it), not wiped.
        assert task.archived_from_status == "done"
        events = [e for e in kb.list_events(conn, tid) if e.kind == "unarchived"]
        assert len(events) == 1
        assert events[0].payload["restored_status"] == "done"
        assert events[0].payload["actor"] == "ryan"


def test_unarchive_restores_blocked_exactly(kanban_home):
    with kbc.connect() as conn:
        tid = _blocked_task(conn)
        kb.archive_task(conn, tid, reason="x", actor="tester")

        result = kb.unarchive_task(conn, tid, reason="restore", actor="ryan")

        assert result.ok is True
        assert result.status == "blocked"
        assert kb.get_task(conn, tid).status == "blocked"


def test_unarchive_does_not_promote_children(kanban_home):
    """Archive only ever accepted zero active descendants, so restoring it
    must not make a (newly created, unrelated-at-archive-time) child runnable
    as a side effect of unarchive specifically."""
    with kbc.connect() as conn:
        parent = _done_task(conn, title="parent")
        kb.archive_task(conn, parent, reason="x", actor="tester")
        child = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent, child)
        assert kb.get_task(conn, child).status == "ready"  # archived is terminal

        kb.unarchive_task(conn, parent, reason="restore", actor="ryan")

        # Restoring the parent to 'done' doesn't retroactively break the
        # child that was already satisfied; recompute_ready is simply not
        # invoked by unarchive (done is terminal either way here).
        assert kb.get_task(conn, child).status == "ready"


def test_unarchive_refuses_non_archived_task(kanban_home):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        result = kb.unarchive_task(conn, tid, reason="x", actor="tester")
        assert result.ok is False
        assert result.error_code == "not_archived"


def test_unarchive_refuses_unknown_task(kanban_home):
    with kbc.connect() as conn:
        result = kb.unarchive_task(conn, "t_doesnotexist", reason="x", actor="tester")
        assert result.ok is False
        assert result.error_code == "not_found"


def test_unarchive_refuses_legacy_row_with_no_metadata(kanban_home):
    """A row archived before these columns existed (``archived_from_status``
    NULL) must be refused rather than guessed at."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="legacy")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (tid,))

        result = kb.unarchive_task(conn, tid, reason="x", actor="tester")

        assert result.ok is False
        assert result.error_code == "legacy_archive_metadata_missing"
        assert kb.get_task(conn, tid).status == "archived"


def test_unarchive_requires_nonblank_reason_and_actor(kanban_home):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        kb.archive_task(conn, tid, reason="x", actor="tester")
        with pytest.raises(ValueError):
            kb.unarchive_task(conn, tid, reason="", actor="tester")
        with pytest.raises(ValueError):
            kb.unarchive_task(conn, tid, reason="x", actor="  ")
        assert kb.get_task(conn, tid).status == "archived"


# ---------------------------------------------------------------------------
# Migration: additive columns apply safely to a reduced legacy schema
# ---------------------------------------------------------------------------


def test_migration_adds_archive_columns_to_reduced_legacy_schema(tmp_path, monkeypatch):
    """A pre-existing board created before these columns existed must gain
    them on the next connect() without losing any row, matching the pattern
    _BASE_TASK_COLUMNS covers for other legacy boards (#112953)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    raw = sqlite3.connect(str(db_path))
    try:
        raw.executescript(
            """
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, body TEXT, assignee TEXT,
                status TEXT NOT NULL, priority INTEGER DEFAULT 0, created_by TEXT,
                created_at INTEGER NOT NULL, started_at INTEGER, completed_at INTEGER,
                workspace_kind TEXT NOT NULL DEFAULT 'scratch', workspace_path TEXT,
                claim_lock TEXT, claim_expires INTEGER
            );
            CREATE TABLE task_links (parent_id TEXT NOT NULL, child_id TEXT NOT NULL,
                PRIMARY KEY (parent_id, child_id));
            CREATE TABLE task_comments (id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL, author TEXT NOT NULL, body TEXT NOT NULL,
                created_at INTEGER NOT NULL);
            CREATE TABLE task_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT,
                created_at INTEGER NOT NULL);
            """
        )
        raw.execute(
            "INSERT INTO tasks (id, title, status, created_at, workspace_kind) "
            "VALUES ('t_legacy01', 'legacy row', 'archived', 1000, 'scratch')"
        )
        raw.commit()
    finally:
        raw.close()

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, "t_legacy01")
        assert task is not None
        assert task.status == "archived"
        assert task.archived_from_status is None
        assert task.archive_reason is None
        assert task.archived_by is None

        # Reads/show work fine; unarchive refuses safely rather than guessing.
        result = kb.unarchive_task(conn, "t_legacy01", reason="x", actor="tester")
        assert result.ok is False
        assert result.error_code == "legacy_archive_metadata_missing"

        # A NEW archive on a fresh task still writes full provenance.
        fresh = kb.create_task(conn, title="fresh")
        assert kb.claim_task(conn, fresh) is not None
        assert kb.complete_task(conn, fresh, summary="done")
        fresh_result = kb.archive_task(conn, fresh, reason="x", actor="tester")
        assert fresh_result.ok is True
        assert kb.get_task(conn, fresh).archived_from_status == "done"


# ---------------------------------------------------------------------------
# Dispatcher visibility: archived cards are never ready/claim candidates
# ---------------------------------------------------------------------------


def test_archived_task_never_claimable(kanban_home):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        kb.archive_task(conn, tid, reason="x", actor="tester")
        assert kb.claim_task(conn, tid) is None


def test_archived_task_excluded_from_board_stats(kanban_home):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        before = kb.board_stats(conn)
        kb.archive_task(conn, tid, reason="x", actor="tester")
        after = kb.board_stats(conn)
        assert after["by_status"].get("done", 0) == before["by_status"].get("done", 0) - 1
        assert after["by_status"].get("archived", 0) == 0


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _parser():
    import argparse

    from hermes_cli import kanban as kc

    p = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = p.add_subparsers(dest="command")
    kc.build_parser(sub)
    return p


def _cli(*args: str) -> int:
    from hermes_cli import kanban as kc

    ns = _parser().parse_args(["kanban", *args])
    return kc.kanban_command(ns)


def test_cli_archive_requires_reason(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = _done_task(conn)
    rc = _cli("archive", tid)
    assert rc != 0
    out = capsys.readouterr()
    assert "reason" in (out.out + out.err).lower()


def test_cli_archive_and_unarchive_roundtrip_json(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = _done_task(conn)

    import json as _json

    rc = _cli("archive", tid, "--reason", "cli test", "--json")
    assert rc == 0
    archived_payload = _json.loads(capsys.readouterr().out)
    assert archived_payload["ok"] is True
    assert archived_payload["status"] == "archived"
    assert archived_payload["archived_from_status"] == "done"

    rc = _cli("unarchive", tid, "--reason", "cli restore", "--json")
    assert rc == 0
    restored_payload = _json.loads(capsys.readouterr().out)
    assert restored_payload["ok"] is True
    assert restored_payload["status"] == "done"


def test_cli_list_hides_archived_by_default_shows_with_flag(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = _done_task(conn)
        kb.archive_task(conn, tid, reason="x", actor="tester")

    import json as _json

    rc = _cli("list", "--json")
    assert rc == 0
    default_ids = {t["id"] for t in _json.loads(capsys.readouterr().out)}
    assert tid not in default_ids

    rc = _cli("list", "--archived", "--status", "archived", "--json")
    assert rc == 0
    archived_ids = {t["id"] for t in _json.loads(capsys.readouterr().out)}
    assert tid in archived_ids


def test_cli_unarchive_refuses_legacy_row(kanban_home, capsys):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="legacy")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (tid,))

    rc = _cli("unarchive", tid, "--reason", "x", "--json")
    assert rc != 0
    import json as _json

    payload = _json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error_code"] == "legacy_archive_metadata_missing"

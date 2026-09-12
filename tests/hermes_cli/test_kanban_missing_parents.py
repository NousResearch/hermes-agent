"""Kanban dependency gates fail closed when a linked parent row is missing."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _delete_parent_keep_link(parent_id: str, child_id: str) -> None:
    """Create the corrupt shape that foreign-key enforcement normally prevents."""
    conn = sqlite3.connect(kb.kanban_db_path())
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DELETE FROM tasks WHERE id = ?", (parent_id,))
        conn.commit()
        assert conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        ).fetchone()
    finally:
        conn.close()


def _logical_snapshot(conn: sqlite3.Connection) -> tuple[list[tuple], list[tuple], list[tuple]]:
    return (
        [tuple(row) for row in conn.execute(
            "SELECT id, status, claim_lock, current_run_id FROM tasks ORDER BY id"
        )],
        [tuple(row) for row in conn.execute(
            "SELECT task_id, kind, payload, run_id FROM task_events ORDER BY id"
        )],
        [tuple(row) for row in conn.execute(
            "SELECT task_id, status, outcome FROM task_runs ORDER BY id"
        )],
    )


def test_missing_parent_rolls_back_readiness_batch_and_refuses_claim(
    kanban_home, caplog,
):
    with kbc.connect() as conn:
        healthy = kb.create_task(conn, title="healthy sibling", assignee="worker")
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (healthy,))
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(
            conn, title="corrupt child", assignee="worker", parents=(parent,),
        )
        conn.commit()

    _delete_parent_keep_link(parent, child)

    with kbc.connect() as conn:
        before = _logical_snapshot(conn)
        with pytest.raises(RuntimeError, match=rf"{child}.*{parent}") as caught:
            kb.recompute_ready(conn)
        assert isinstance(caught.value, kb.MissingParentError)
        assert _logical_snapshot(conn) == before

        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        conn.commit()
        before_claim = _logical_snapshot(conn)
        with pytest.raises(kb.MissingParentError, match=rf"{child}.*{parent}"):
            kb.claim_task(conn, child)
        assert _logical_snapshot(conn) == before_claim

        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (child,))
        conn.commit()
        before_review = _logical_snapshot(conn)
        with pytest.raises(kb.MissingParentError, match=rf"{child}.*{parent}"):
            kb.claim_review_task(conn, child)
        assert _logical_snapshot(conn) == before_review

        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))
        unrelated = kb.create_task(conn, title="unrelated completion", assignee="worker")
        conn.commit()
        assert kb.complete_task(conn, unrelated, result="done")
        assert kb.get_task(conn, unrelated).status == "done"
        assert kb.get_task(conn, child).status == "todo"
        assert "committed, but dependency readiness recomputation was refused" in caplog.text

        conn.execute(
            "INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind) "
            "VALUES (?, 'restored parent', 'ready', 0, 1, 'scratch')",
            (parent,),
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, healthy).status == "ready"
        assert kb.get_task(conn, child).status == "todo"
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        conn.commit()
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, child).status == "ready"


def test_dispatch_detects_missing_parent_before_reclaim_or_spawn(
    kanban_home, monkeypatch, capsys,
):
    with kbc.connect() as conn:
        stale = kb.create_task(conn, title="expired claim", assignee="worker")
        assert kb.claim_task(conn, stale, ttl_seconds=1) is not None
        conn.execute("UPDATE tasks SET claim_expires = 0 WHERE id = ?", (stale,))

        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(
            conn, title="already ready child", assignee="worker", parents=(parent,),
        )
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (child,))
        conn.commit()

    _delete_parent_keep_link(parent, child)

    effects: list[str] = []
    monkeypatch.setattr(kbd, "reap_worker_zombies", lambda: effects.append("reap"))
    with kbc.connect() as conn:
        before = _logical_snapshot(conn)
        with pytest.raises(RuntimeError, match=rf"{child}.*{parent}") as caught:
            kbd.dispatch_once(
                conn,
                spawn_fn=lambda *_args, **_kwargs: effects.append("spawn"),
            )
        assert isinstance(caught.value, kb.MissingParentError)
        assert effects == []
        assert _logical_snapshot(conn) == before

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["kanban", "dispatch", "--dry-run", "--json"])
    assert kc.kanban_command(args) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert child in output.err
    assert parent in output.err

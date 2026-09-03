"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Workspace flag parsing
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# run_slash smoke tests (end-to-end via the same entry both CLI and gateway use)
# ---------------------------------------------------------------------------



def test_kanban_list_json_includes_session_id(kanban_home):
    """JSON output exposes `session_id` so external clients (Scarf, web
    dashboards) don't need a side query to filter by chat session."""
    from hermes_cli import kanban_db as kb
    with kb.connect() as conn:
        kb.create_task(
            conn, title="acp task", assignee="alice", session_id="acp-x"
        )
    raw = kc.run_slash("list --json")
    payload = json.loads(raw)
    assert any(
        row.get("title") == "acp task"
        and row.get("session_id") == "acp-x"
        for row in payload
    )


def test_kanban_show_text_renders_graph_with_open_connection(kanban_home):
    with kb.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="parent task")
        child_id = kb.create_task(conn, title="child task")
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)

    output = kc.run_slash(f"show {child_id}")

    assert f"Task {child_id}: child task" in output
    assert f"parents:   {parent_id}" in output
    assert "Cannot operate on a closed database" not in output


def test_board_override_is_isolated_per_concurrent_call(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    barrier = threading.Barrier(2)
    original_init_db = kb.init_db

    def slow_init_db(*args, **kwargs):
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return original_init_db(*args, **kwargs)

    monkeypatch.setattr(kb, "init_db", slow_init_db)

    failures: list[str] = []

    def worker(board: str, title: str) -> None:
        args = parser.parse_args(["kanban", "--board", board, "create", title])
        rc = kc.kanban_command(args)
        if rc != 0:
            failures.append(f"{board}:{rc}")

    t1 = threading.Thread(target=worker, args=("alpha", "alpha-task"))
    t2 = threading.Thread(target=worker, args=("beta", "beta-task"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert failures == []

    with kb.connect_closing(board="alpha") as conn:
        alpha_titles = [row.title for row in kb.list_tasks(conn, limit=100)]
    with kb.connect_closing(board="beta") as conn:
        beta_titles = [row.title for row in kb.list_tasks(conn, limit=100)]

    assert alpha_titles == ["alpha-task"]
    assert beta_titles == ["beta-task"]


# ---------------------------------------------------------------------------
# Integration with the COMMAND_REGISTRY
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# reclaim + reassign CLI smoke tests
# ---------------------------------------------------------------------------

def test_run_slash_reclaim_running_task(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb

    out1 = kc.run_slash("create 'stuck worker task' --assignee broken-model")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    tid = m.group(1)

    # Simulate a running claim outside TTL.
    conn = kb.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reclaim {tid} --reason 'test'")
    assert "Reclaimed" in out, out
    # Status back to ready.
    out2 = kc.run_slash(f"show {tid}")
    assert "ready" in out2.lower()


def _create_args(**overrides):
    """Build a minimal argparse.Namespace for ``_cmd_create``."""
    import argparse

    base = {
        "workspace": "scratch",
        "branch": None,
        "max_runtime": None,
        "max_retries": None,
        "title": "durability test task",
        "body": None,
        "assignee": None,
        "created_by": None,
        "project": None,
        "tenant": None,
        "priority": 0,
        "parent": [],
        "triage": False,
        "idempotency_key": None,
        "skills": [],
        "model_override": None,
        "provider_override": None,
        "goal_mode": False,
        "goal_max_turns": None,
        "initial_status": "running",
        "json": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_create_fails_closed_when_row_not_durable(kanban_home, monkeypatch, capsys):
    """#76153: a row visible to the writer's own connection but missing for a
    FRESH connection (the silent-COMMIT symptom) must fail closed with a
    non-zero exit and a clear message — never print ``Created`` + exit 0."""
    real_get_task = kb.get_task
    calls = {"n": 0}

    def flaky_get_task(conn, task_id):
        # First read (inside the write block, same connection) sees the row;
        # the post-commit verification read (fresh connection) does not —
        # exactly the "Created t_<hex> printed, row never in kanban.db" shape.
        calls["n"] += 1
        if calls["n"] == 1:
            return real_get_task(conn, task_id)
        return None

    monkeypatch.setattr(kb, "get_task", flaky_get_task)

    rc = kc._cmd_create(_create_args(title="silent commit check"))
    captured = capsys.readouterr()

    assert rc == 1
    assert "Created" not in captured.out, captured.out
    assert "not durably visible" in captured.err, captured.err


def test_cmd_create_verifies_row_from_fresh_connection(kanban_home, capsys):
    """Happy path: after create, the row must be readable from a brand-new
    connection (proof the COMMIT is visible to other processes, not just the
    writer)."""
    rc = kc._cmd_create(_create_args(title="fresh-conn check"))
    captured = capsys.readouterr()

    assert rc == 0
    assert "Created t_" in captured.out, captured.out
    m = __import__("re").search(r"(t_[a-f0-9]+)", captured.out)
    assert m
    # The row is really there from a separate connection.
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, m.group(1)) is not None


def test_cmd_create_non_default_board_reports_db_path(kanban_home, monkeypatch, capsys):
    """#76153 reproduction: `HERMES_KANBAN_BOARD=operations` writes to the
    board's own kanban.db, NOT the default `<root>/kanban.db`. The CLI must
    surface which board/db the row landed in so external verifiers (sqlite3,
    wrapper scripts) check the right file instead of reporting a phantom
    silent-COMMIT."""
    import sqlite3

    from hermes_cli import kanban_db as kb

    kb.create_board("operations")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "operations")

    rc = kc._cmd_create(_create_args(title="board-scoped create", assignee="ashitaka"))
    captured = capsys.readouterr()

    assert rc == 0
    assert "board=operations" in captured.out, captured.out
    assert "db=" in captured.err, captured.err

    # The row lives in the board's own DB — the default kanban.db has nothing.
    board_db = kb.kanban_db_path()  # resolves via HERMES_KANBAN_BOARD -> operations
    assert "boards" in str(board_db), board_db
    m = __import__("re").search(r"(t_[a-f0-9]+)", captured.out)
    assert m
    with sqlite3.connect(board_db) as c:
        rows = c.execute(
            "SELECT id FROM tasks WHERE id=?", (m.group(1),)
        ).fetchall()
    assert rows, "row must be present in the board-scoped database"
    default_db = kanban_home / "kanban.db"
    with sqlite3.connect(default_db) as c:
        rows = c.execute(
            "SELECT id FROM tasks WHERE id=?", (m.group(1),)
        ).fetchall()
    assert not rows, "row must NOT be in the default kanban.db"




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------



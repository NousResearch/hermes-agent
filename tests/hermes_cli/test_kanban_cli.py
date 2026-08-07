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




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# amend — edit a live task's title/body (CLI surface)
# ---------------------------------------------------------------------------

def _kanban_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser


def _amend_rc(argv: list[str]) -> int:
    """Parse ``hermes kanban …`` argv and dispatch, returning the exit code."""
    parser = _kanban_parser()
    args = parser.parse_args(["kanban", *argv])
    return kc.kanban_command(args)


def _create_task_via_slash(title: str, body: str | None = None) -> str:
    import re

    cmd = f"create '{title}'"
    if body is not None:
        cmd += f" --body '{body}'"
    out = kc.run_slash(cmd)
    m = re.search(r"(t_[a-f0-9]+)", out)
    assert m, out
    return m.group(1)


def test_run_slash_amend_title_and_body(kanban_home):
    tid = _create_task_via_slash("wrong source", body="old body")

    out = kc.run_slash(f"amend {tid} --title 'right source' --body 'new body'")
    assert "Amended" in out, out
    assert "title" in out and "body" in out

    show = json.loads(kc.run_slash(f"show {tid} --json"))
    assert show["task"]["title"] == "right source"
    assert show["task"]["body"] == "new body"

    # Auditable `edited` event with the documented payload shape.
    with kb.connect() as conn:
        rows = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()
    edited = [r for r in rows if r["kind"] == "edited"]
    assert len(edited) == 1
    payload = json.loads(edited[0]["payload"])
    assert payload["fields"] == ["title", "body"]
    assert payload["title"] == "right source"
    assert payload["body_len"] == len("new body")


def test_amend_rejects_body_and_body_file_together(kanban_home, tmp_path, capsys):
    tid = _create_task_via_slash("t")
    bf = tmp_path / "body.md"
    bf.write_text("from file")
    rc = _amend_rc(["amend", tid, "--body", "inline", "--body-file", str(bf)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "only one of --body / --body-file" in err
    # Task untouched.
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).title == "t"


def test_amend_no_flags_is_usage_error(kanban_home, capsys):
    tid = _create_task_via_slash("t")
    rc = _amend_rc(["amend", tid])
    assert rc == 2
    assert "nothing to edit" in capsys.readouterr().err


def test_amend_unknown_task_id(kanban_home, capsys):
    rc = _amend_rc(["amend", "t_ghost", "--title", "x"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "t_ghost" in err
    assert "unknown id" in err


def test_amend_blank_title_rejected(kanban_home, capsys):
    tid = _create_task_via_slash("keep me")
    rc = _amend_rc(["amend", tid, "--title", "  "])
    assert rc == 2
    assert "blank" in capsys.readouterr().err
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).title == "keep me"


def test_amend_body_file_reads_file(kanban_home, tmp_path):
    tid = _create_task_via_slash("t", body="old")
    bf = tmp_path / "body.md"
    bf.write_text("# repointed\n\nmulti-line body\n", encoding="utf-8")
    rc = _amend_rc(["amend", tid, "--body-file", str(bf)])
    assert rc == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).body == "# repointed\n\nmulti-line body\n"


def test_amend_body_file_dash_reads_stdin(kanban_home, monkeypatch):
    import io
    import sys as _sys

    tid = _create_task_via_slash("t", body="old")
    monkeypatch.setattr(_sys, "stdin", io.StringIO("body from stdin\n"))
    rc = _amend_rc(["amend", tid, "--body-file", "-"])
    assert rc == 0
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).body == "body from stdin\n"


def test_amend_body_file_missing_path(kanban_home, tmp_path, capsys):
    tid = _create_task_via_slash("t", body="old")
    missing = tmp_path / "does-not-exist.md"
    rc = _amend_rc(["amend", tid, "--body-file", str(missing)])
    assert rc == 2
    assert "--body-file" in capsys.readouterr().err
    # Body untouched on failure.
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).body == "old"


def test_amend_archived_task_rejected(kanban_home, capsys):
    tid = _create_task_via_slash("t")
    with kb.connect() as conn:
        assert kb.archive_task(conn, tid) is True
    rc = _amend_rc(["amend", tid, "--title", "new"])
    assert rc == 2
    assert "archived" in capsys.readouterr().err

"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import builtins
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


# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BrokenPipeError handling (Sentry NICHE-BOTS-F)
# ---------------------------------------------------------------------------


def test_cmd_list_broken_pipe_is_handled_cleanly(kanban_home, monkeypatch):
    """`hermes kanban list --json` piped into a reader that closes its end
    early (e.g. `| head`) must not crash with an unhandled BrokenPipeError
    traceback; kanban_command should catch it, log it, and return 0."""
    with kb.connect() as conn:
        kb.create_task(conn, title="some task", assignee="alice")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban", "list", "--json"])

    real_print = builtins.print

    def flaky_print(*p_args, **p_kwargs):
        # Simulate the stdout reader having closed its pipe early, exactly
        # like piping into `head` and closing the read end.
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(builtins, "print", flaky_print)
    try:
        rc = kc.kanban_command(args)
    finally:
        monkeypatch.setattr(builtins, "print", real_print)

    assert rc == 0


def test_cmd_list_broken_pipe_is_logged(kanban_home, monkeypatch, caplog):
    """The BrokenPipeError must be logged (not silently swallowed)."""
    import logging

    with kb.connect() as conn:
        kb.create_task(conn, title="some task", assignee="alice")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    args = parser.parse_args(["kanban", "list", "--json"])

    def flaky_print(*p_args, **p_kwargs):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(builtins, "print", flaky_print)
    with caplog.at_level(logging.INFO, logger="hermes_cli.kanban"):
        rc = kc.kanban_command(args)

    assert rc == 0
    assert any("BrokenPipeError" in rec.message for rec in caplog.records)


def test_hermes_kanban_list_survives_reader_closing_pipe_early(kanban_home):
    """End-to-end: run `hermes kanban list --json` as a real subprocess piped
    into a reader that closes its stdin early, mirroring the exact Sentry
    reproduction. The CLI process must exit cleanly with no traceback on
    stderr."""
    import subprocess
    import sys as _sys

    with kb.connect() as conn:
        for i in range(50):
            kb.create_task(conn, title=f"task {i}" * 20, assignee="alice")

    env = dict(os.environ)
    env["HERMES_HOME"] = str(kanban_home)

    hermes_cli_dir = Path(kc.__file__).resolve().parent.parent

    proc = subprocess.Popen(
        [_sys.executable, "-m", "hermes_cli.main", "kanban", "list", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(hermes_cli_dir),
        env=env,
    )
    # Read a small prefix, then close the read end early -- this is what
    # triggers BrokenPipeError in the writer once its buffer fills.
    try:
        proc.stdout.read(16)
    finally:
        proc.stdout.close()

    _, stderr = proc.communicate(timeout=15)
    stderr_text = stderr.decode("utf-8", "replace")

    # This is the actual signature of the Sentry crash report: an
    # *unhandled, propagating* BrokenPipeError with _cmd_list/kanban_command
    # in the traceback, surfaced through main()'s excepthook. It must be
    # gone, and the process must report success.
    #
    # NOTE: CPython may still print a benign, unrelated
    # "Exception ignored in: <stdout> ... BrokenPipeError" line during
    # interpreter shutdown when a large buffered write hits a pipe closed
    # by the reader -- that comes from sys.unraisablehook at GC time (not
    # sys.excepthook) and happens for *any* Python program in this
    # situation (e.g. `python3 -c "print('x'*999999)" | head -c1`
    # reproduces the identical line on stock CPython). It is not the bug
    # this test guards against, so we only assert there's no propagating
    # traceback rooted in our own code.
    assert "Traceback (most recent call last)" not in stderr_text, stderr_text
    assert "_cmd_list" not in stderr_text, stderr_text
    assert "kanban_command" not in stderr_text, stderr_text
    assert proc.returncode == 0, (proc.returncode, stderr_text)

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




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# #68613: unblock positional preflight (no partial mutation)
# ---------------------------------------------------------------------------


def test_unblock_rejects_reason_looking_positional_before_mutation(kanban_home, capsys):
    """Reason text before a real id must not partially unblock."""
    import re

    out = kc.run_slash("create 'x' --assignee alice")
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"claim {tid}")
    assert "Blocked" in kc.run_slash(f"block {tid} 'need decision'")

    with kb.connect() as conn:
        before = kb.get_task(conn, tid)
        assert before is not None and before.status == "blocked"

    ns = argparse.Namespace(
        task_ids=["skill external_dirs fixed; retry review", tid],
        reason=None,
    )
    code = kc._cmd_unblock(ns)
    captured = capsys.readouterr()
    assert code == 1
    assert "not a task id" in captured.err
    assert "--reason" in captured.err
    assert "Did you mean" in captured.err
    assert "No tasks were modified" in captured.err

    with kb.connect() as conn:
        after = kb.get_task(conn, tid)
        assert after is not None and after.status == "blocked"


def test_unblock_reason_flag_with_bad_positional_no_mutation(kanban_home, capsys):
    """--reason plus a reason-looking positional must not traceback or mutate."""
    import re

    out = kc.run_slash("create 'x' --assignee alice")
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"claim {tid}")
    kc.run_slash(f"block {tid} 'need decision'")

    ns = argparse.Namespace(
        task_ids=["not a real id phrase", tid],
        reason="already captured",
    )
    code = kc._cmd_unblock(ns)
    captured = capsys.readouterr()
    assert code == 1
    assert "Traceback" not in captured.err
    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


def test_unblock_with_reason_flag_and_valid_id(kanban_home):
    import re

    out = kc.run_slash("create 'x' --assignee alice")
    tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
    kc.run_slash(f"claim {tid}")
    kc.run_slash(f"block {tid} 'need decision'")
    ns = argparse.Namespace(task_ids=[tid], reason="skill external_dirs fixed")
    assert kc._cmd_unblock(ns) == 0
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status != "blocked"


def test_looks_like_task_id_accepts_short_and_long_hex():
    assert kc._looks_like_task_id("t_abcd")
    assert kc._looks_like_task_id("t_cc0254fd")
    assert kc._looks_like_task_id("t_deadbeefcafe")
    assert not kc._looks_like_task_id("skill external_dirs fixed")
    assert not kc._looks_like_task_id("t_short")  # non-hex
    assert not kc._looks_like_task_id("")


def test_unblock_bulk_valid_ids(kanban_home):
    import re

    ids = []
    for title in ("a", "b"):
        out = kc.run_slash(f"create '{title}' --assignee alice")
        tid = re.search(r"(t_[a-f0-9]+)", out).group(1)
        kc.run_slash(f"claim {tid}")
        kc.run_slash(f"block {tid} 'need decision'")
        ids.append(tid)
    ns = argparse.Namespace(task_ids=ids, reason=None)
    assert kc._cmd_unblock(ns) == 0
    with kb.connect() as conn:
        for tid in ids:
            assert kb.get_task(conn, tid).status != "blocked"


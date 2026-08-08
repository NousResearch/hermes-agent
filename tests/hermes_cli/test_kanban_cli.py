"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


def _run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return kc.kanban_command(parser.parse_args(["kanban", *argv]))


def _create_ready(conn, *, title: str, assignee: str | None = None) -> str:
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL "
        "WHERE id=?",
        (task_id,),
    )
    conn.commit()
    return task_id


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


def test_claim_json_returns_external_worker_receipt(kanban_home, capsys):
    with kb.connect_closing() as conn:
        task_id = _create_ready(
            conn,
            title="external work",
            assignee="codex-alpha-frontend",
        )

    assert _run_cli(["claim", task_id, "--json"]) == 0

    receipt = json.loads(capsys.readouterr().out)
    assert receipt["task_id"] == task_id
    assert receipt["assignee"] == "codex-alpha-frontend"
    assert isinstance(receipt["run_id"], int)
    assert receipt["claim_lock"]
    assert receipt["claim_expires"] > int(time.time())
    assert Path(receipt["workspace"]).is_absolute()


def test_cli_heartbeat_extends_exact_external_claim(kanban_home, monkeypatch):
    with kb.connect_closing() as conn:
        task_id = _create_ready(conn, title="heartbeat")
        task = kb.claim_task(
            conn, task_id, claimer="host:external-worker", ttl_seconds=1
        )
        assert task is not None
        before = task.claim_expires
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:external-worker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))

    assert _run_cli(["heartbeat", task_id, "--note", "turn active"]) == 0

    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).claim_expires > before


def test_cli_heartbeat_rejects_stale_external_claim(kanban_home, monkeypatch):
    with kb.connect_closing() as conn:
        task_id = _create_ready(conn, title="stale")
        task = kb.claim_task(conn, task_id, claimer="host:current")
        assert task is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:stale")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))

    assert _run_cli(["heartbeat", task_id]) == 1

    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).claim_lock == "host:current"


def test_priority_command_updates_one_task(kanban_home):
    with kb.connect_closing() as conn:
        task_id = _create_ready(conn, title="reprioritize")

    assert _run_cli(["priority", task_id, "-25"]) == 0

    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).priority == -25


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

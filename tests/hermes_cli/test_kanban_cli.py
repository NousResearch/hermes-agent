"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations
from hermes_cli import kanban_worker_process as worker_process
from hermes_cli import kanban_db_dispatch as dispatch_impl

import argparse
import json
import os
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
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
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as conn:
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


def test_kanban_list_json_includes_worker_execution_settings(kanban_home):
    """JSON output must expose the settings that govern worker safety."""
    import hermes_cli.kanban_db_connect as _hermes_cli_kanban_db_connect
    with _hermes_cli_kanban_db_connect.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="local worker",
            assignee="ci-static-fixer",
            max_runtime_seconds=3600,
            model_override="qwen3.5:4b",
            provider_override="ollama-launch",
            reasoning_effort="none",
        )

    payload = json.loads(kc.run_slash(f"show {task_id} --json"))
    task = payload["task"]
    assert task["max_runtime_seconds"] == 3600
    assert task["model_override"] == "qwen3.5:4b"
    assert task["provider_override"] == "ollama-launch"
    assert task["reasoning_effort"] == "none"


def test_kanban_show_json_includes_runtime_limit(kanban_home):
    with kbc.connect() as conn:
        bounded_id = kb.create_task(
            conn, title="bounded task", max_runtime_seconds=2700
        )
        uncapped_id = kb.create_task(conn, title="uncapped task")

    bounded = json.loads(kc.run_slash(f"show {bounded_id} --json"))
    uncapped = json.loads(kc.run_slash(f"show {uncapped_id} --json"))

    assert bounded["task"]["max_runtime_seconds"] == 2700
    assert uncapped["task"]["max_runtime_seconds"] is None
def test_kanban_show_text_renders_graph_with_open_connection(kanban_home):
    with kbc.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="parent task")
        child_id = kb.create_task(conn, title="child task")
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)

    output = kc.run_slash(f"show {child_id}")

    assert f"Task {child_id}: child task" in output
    assert f"parents:   {parent_id}" in output
    assert "Cannot operate on a closed database" not in output


def test_kanban_edit_updates_documented_task_fields(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="old title", body="old body", priority=2)

    output = kc.run_slash(
        f"edit {task_id} --title 'new title' --body 'new body' --priority 70"
    )

    assert f"Edited {task_id}" in output
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
    assert (task.title, task.body, task.priority) == ("new title", "new body", 70)
    assert any(event.kind == "reprioritized" for event in events)


def test_worker_link_preserves_foreign_child_rules(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        worker = kb.create_task(conn, title="worker")
        assert kb.claim_task(conn, worker, claimer="worker") is not None
        worker_run_id = kb.get_task(conn, worker).current_run_id
        parent = kb.create_task(conn, title="unfinished parent")
        ready_child = kb.create_task(conn, title="foreign ready child")
        running_child = kb.create_task(conn, title="foreign running child")
        assert kb.claim_task(conn, running_child, claimer="other") is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", worker)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker_run_id))

    assert kc._cmd_link(argparse.Namespace(
        parent_id=parent, child_id=ready_child,
    )) == 0
    with pytest.raises(ValueError, match="child is already running"):
        kc._cmd_link(argparse.Namespace(
            parent_id=parent, child_id=running_child,
        ))
    # Owner handoff: the worker links its own running card, proving ownership
    # with HERMES_KANBAN_RUN_ID — the one path _cmd_link forwards a run id for.
    assert kc._cmd_link(argparse.Namespace(
        parent_id=parent, child_id=worker,
    )) == 0

    with kbc.connect_closing() as conn:
        assert kb.parent_ids(conn, ready_child) == [parent]
        assert kb.parent_ids(conn, running_child) == []
        assert kb.parent_ids(conn, worker) == [parent]


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

    with kbc.connect_closing(board="alpha") as conn:
        alpha_titles = [row.title for row in kb.list_tasks(conn, limit=100)]
    with kbc.connect_closing(board="beta") as conn:
        beta_titles = [row.title for row in kb.list_tasks(conn, limit=100)]

    assert alpha_titles == ["alpha-task"]
    assert beta_titles == ["beta-task"]


def test_dispatch_fails_closed_when_config_cannot_be_loaded(kanban_home, monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: (_ for _ in ()).throw(RuntimeError("invalid config")),
    )
    args = argparse.Namespace(
        dry_run=True, max=None, json=True, failure_limit=3,
    )

    assert kc._cmd_dispatch(args) == 1
    assert json.loads(capsys.readouterr().out) == {
        "error": "RuntimeError",
        "status": "config_unavailable",
    }


# ---------------------------------------------------------------------------
# Integration with the COMMAND_REGISTRY
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# reclaim + reassign CLI smoke tests
# ---------------------------------------------------------------------------

def test_run_slash_reclaim_running_task(kanban_home, monkeypatch):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    monkeypatch.setattr(
        kb,
        "_terminate_reclaimed_worker",
        lambda pid, lock, **_kwargs: {
            "prev_pid": pid,
            "host_local": True,
            "termination_attempted": True,
            "terminated": True,
            "sigkill": False,
        },
    )

    out1 = kc.run_slash("create 'stuck worker task' --assignee broken-model")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    tid = m.group(1)

    # Simulate a running claim outside TTL.
    conn = kbc.connect()
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


def test_unblock_reason_records_operator_outside_worker(kanban_home, monkeypatch):
    import hermes_cli.kanban_db_connect as _hermes_cli_kanban_db_connect
    monkeypatch.setenv("HERMES_PROFILE_NAME", "default")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    with _hermes_cli_kanban_db_connect.connect_closing() as conn:
        task_id = kb.create_task(conn, title="validated intake")
        assert kb.block_task(conn, task_id, reason="awaiting operator")

    output = kc.run_slash(f"unblock {task_id} --reason 'validated for local repair'")

    assert f"Unblocked {task_id}" in output
    with _hermes_cli_kanban_db_connect.connect_closing() as conn:
        comments = kb.list_comments(conn, task_id)
    assert [(comment.author, comment.body) for comment in comments] == [
        ("operator", "UNBLOCK: validated for local repair")
    ]


def test_unblock_is_orchestrator_only_inside_worker(kanban_home, monkeypatch):
    """A Kanban worker cannot self-unblock a task; it must hand off to the orchestrator."""
    import hermes_cli.kanban_db_connect as _hermes_cli_kanban_db_connect
    monkeypatch.setenv("HERMES_PROFILE_NAME", "repair-worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_12345678")
    with _hermes_cli_kanban_db_connect.connect_closing() as conn:
        task_id = kb.create_task(conn, title="worker retry")
        assert kb.block_task(conn, task_id, reason="transient")

    output = kc.run_slash(f"unblock {task_id} --reason 'worker retry'")

    assert "orchestrator-only" in output
    with _hermes_cli_kanban_db_connect.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        comments = kb.list_comments(conn, task_id)
    assert task.status == "blocked"
    assert comments == []




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------

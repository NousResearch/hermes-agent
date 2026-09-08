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


def test_kanban_show_text_renders_graph_with_open_connection(kanban_home):
    with kbc.connect_closing() as conn:
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

    with kbc.connect_closing(board="alpha") as conn:
        alpha_titles = [row.title for row in kb.list_tasks(conn, limit=100)]
    with kbc.connect_closing(board="beta") as conn:
        beta_titles = [row.title for row in kb.list_tasks(conn, limit=100)]

    assert alpha_titles == ["alpha-task"]
    assert beta_titles == ["beta-task"]


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (
            ["kanban", "workflow", "create", "wf_cli", "--name", "release",
             "--tenant", "tenant-a", "--acceptance-task", "t_accept",
             "--root-task", "t_root", "--mutation-id", "create-1", "--json"],
            {"workflow_action": "create", "workflow_id": "wf_cli", "name": "release",
             "tenant": "tenant-a", "acceptance_task": "t_accept", "root_task": "t_root",
             "mutation_id": "create-1", "json": True},
        ),
        (
            ["kanban", "workflow", "show", "wf_cli", "--tenant", "tenant-a", "--generation", "2", "--outcomes", "--json"],
            {"workflow_action": "show", "workflow_id": "wf_cli", "tenant": "tenant-a",
             "generation": 2, "outcomes": True, "json": True},
        ),
        (
            ["kanban", "workflow", "add-member", "wf_cli", "task_2", "--tenant", "tenant-a",
             "--stage-key", "implementation", "--stage-role", "implementation", "--required",
             "--expected-version", "1", "--mutation-id", "add-1", "--json"],
            {"workflow_action": "add-member", "workflow_id": "wf_cli", "task_id": "task_2",
             "tenant": "tenant-a", "stage_key": "implementation", "stage_role": "implementation",
             "required": True, "expected_version": 1, "mutation_id": "add-1", "json": True},
        ),
        (
            ["kanban", "workflow", "subscribe", "wf_cli", "--tenant", "tenant-a",
             "--platform", "telegram", "--chat-id", "chat", "--notifier-profile", "default",
             "--target-states", '["PASS","CANCELLED"]', "--delivery-metadata", '{"thread":1}',
             "--expected-version", "2", "--mutation-id", "sub-1"],
            {"workflow_action": "subscribe", "workflow_id": "wf_cli", "tenant": "tenant-a",
             "platform": "telegram", "chat_id": "chat", "notifier_profile": "default",
             "target_states": '["PASS","CANCELLED"]', "delivery_metadata": '{"thread":1}',
             "expected_version": 2, "mutation_id": "sub-1"},
        ),
        (
            ["kanban", "workflow", "reopen", "wf_cli", "--tenant", "tenant-a",
             "--acceptance-task", "task_3", "--members", '[{"task_id":"task_3","stage_key":"acceptance","stage_role":"acceptance","required":true}]',
             "--reason", "retry", "--expected-version", "3", "--mutation-id", "reopen-1"],
            {"workflow_action": "reopen", "workflow_id": "wf_cli", "tenant": "tenant-a",
             "acceptance_task": "task_3", "expected_version": 3, "mutation_id": "reopen-1"},
        ),
        (
            ["kanban", "workflow", "remove-member", "wf_cli", "task_2", "--tenant", "tenant-a",
             "--reason", "obsolete", "--expected-version", "4", "--mutation-id", "remove-1"],
            {"workflow_action": "remove-member", "task_id": "task_2", "expected_version": 4,
             "mutation_id": "remove-1", "reason": "obsolete"},
        ),
        (
            ["kanban", "workflow", "outcome", "wf_cli", "task_2", "PASS", "--tenant", "tenant-a",
             "--run-id", "7", "--supersedes-outcome-id", "6", "--summary", "ok",
             "--metadata", '{"evidence":"x"}', "--expected-version", "5", "--mutation-id", "outcome-1"],
            {"workflow_action": "outcome", "outcome": "PASS", "run_id": 7,
             "supersedes_outcome_id": 6, "expected_version": 5, "mutation_id": "outcome-1"},
        ),
        (
            ["kanban", "workflow", "cancel", "wf_cli", "--tenant", "tenant-a", "--reason", "stop",
             "--expected-version", "6", "--mutation-id", "cancel-1"],
            {"workflow_action": "cancel", "reason": "stop", "expected_version": 6,
             "mutation_id": "cancel-1"},
        ),
        (
            ["kanban", "workflow", "disable", "wf_cli", "--tenant", "tenant-a", "--role", "origin",
             "--reason", "manual"],
            {"workflow_action": "disable", "role": "origin", "reason": "manual"},
        ),
        (
            ["kanban", "workflow", "skip", "wf_cli", "42", "--tenant", "tenant-a", "--role", "origin",
             "--reason", "ack", "--expected-version", "7", "--mutation-id", "skip-1"],
            {"workflow_action": "skip", "event_id": 42, "expected_version": 7,
             "mutation_id": "skip-1"},
        ),
        (
            ["kanban", "workflow", "resume", "wf_cli", "--tenant", "tenant-a"],
            {"workflow_action": "resume", "role": "origin"},
        ),
    ],
)
def test_workflow_parser_wires_each_supported_workflow_operation(
    kanban_home, monkeypatch, tokens, expected,
):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))

    args = parser.parse_args(tokens)

    for field, value in expected.items():
        assert getattr(args, field) == value
    dispatched = []
    monkeypatch.setitem(kc._HANDLERS, "workflow", lambda parsed: dispatched.append(parsed) or 0)

    assert kc.kanban_command(args) == 0
    assert dispatched == [args]


@pytest.mark.parametrize(
    ("raw", "expected_type", "message"),
    [
        ("[]", dict, "--metadata must be a JSON object"),
        ("{}", list, "--members must be a JSON array"),
        ("not-json", dict, "--delivery-metadata must be valid JSON"),
    ],
)
def test_workflow_json_flags_reject_ambiguous_values(raw, expected_type, message):
    with pytest.raises(ValueError, match=message):
        kc._workflow_json_value(raw, flag=message.split(" must ")[0], expected=expected_type)


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
    from hermes_cli import kanban_db_connect as kbc

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




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------

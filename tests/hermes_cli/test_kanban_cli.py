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
# Workflow CLI
# ---------------------------------------------------------------------------


def test_workflow_cli_create_show_add_outcome_and_cancel_json(kanban_home):
    with kb.connect() as conn:
        acceptance = kb.create_task(
            conn, title="accept", assignee="orchestrator", tenant="tenant-a"
        )
        implementation = kb.create_task(
            conn, title="implement", assignee="builder", tenant="tenant-a"
        )

    created = json.loads(kc.run_slash(
        f"workflow create wf_cli --name release --tenant tenant-a "
        f"--acceptance-task {acceptance} --mutation-id create-cli --json"
    ))
    assert created["workflow"]["state"] == "ACTIVE"

    added = json.loads(kc.run_slash(
        f"workflow add-member wf_cli {implementation} --tenant tenant-a "
        "--stage-key implementation --stage-role implementation --required "
        "--expected-version 1 --mutation-id add-cli --json"
    ))
    assert len(added["members"]) == 2

    outcome = json.loads(kc.run_slash(
        f"workflow outcome wf_cli {implementation} PASS --tenant tenant-a "
        "--expected-version 2 --mutation-id outcome-cli --json"
    ))
    assert outcome["workflow"]["state"] == "ACTIVE"

    shown = json.loads(kc.run_slash("workflow show wf_cli --tenant tenant-a --json"))
    assert shown["workflow"]["version"] == 3

    cancelled = json.loads(kc.run_slash(
        "workflow cancel wf_cli --tenant tenant-a --reason stop "
        "--expected-version 3 --mutation-id cancel-cli --json"
    ))
    assert cancelled["workflow"]["state"] == "CANCELLED"


def test_workflow_cli_reopen_subscribe_plain_json_and_auth_error(kanban_home):
    with kb.connect() as conn:
        acceptance = kb.create_task(
            conn, title="accept", assignee="orchestrator", tenant="tenant-a"
        )
    json.loads(kc.run_slash(
        f"workflow create wf_cli_reopen --name release --tenant tenant-a "
        f"--acceptance-task {acceptance} --mutation-id create-reopen-cli --json"
    ))
    passed = json.loads(kc.run_slash(
        f"workflow outcome wf_cli_reopen {acceptance} PASS --tenant tenant-a "
        "--expected-version 1 --mutation-id pass-reopen-cli --json"
    ))
    assert passed["workflow"]["state"] == "PASS"
    with kb.connect() as conn:
        next_acceptance = kb.create_task(
            conn, title="accept 2", assignee="orchestrator", tenant="tenant-a"
        )
        remediation = kb.create_task(
            conn, title="fix", assignee="builder", tenant="tenant-a"
        )
        reverification = kb.create_task(
            conn, title="verify", assignee="x_qa", tenant="tenant-a"
        )
    members = json.dumps([
        {"task_id": next_acceptance, "stage_key": "acceptance-2",
         "stage_role": "acceptance", "required": True},
        {"task_id": remediation, "stage_key": "remediation-2",
         "stage_role": "remediation", "required": True},
        {"task_id": reverification, "stage_key": "reverification-2",
         "stage_role": "reverification", "required": True},
    ])
    reopened = json.loads(kc.run_slash(
        "workflow reopen wf_cli_reopen --tenant tenant-a "
        f"--acceptance-task {next_acceptance} --members-json '{members}' "
        "--reason defect --expected-version 2 --mutation-id reopen-cli --json"
    ))
    assert reopened["workflow"]["active_generation"] == 2
    plain = kc.run_slash("workflow show wf_cli_reopen --tenant tenant-a")
    assert "Workflow wf_cli_reopen: ACTIVE v3 (generation 2)" in plain
    subscribed = kc.run_slash(
        "workflow subscribe wf_cli_reopen --tenant tenant-a --platform api_server "
        "--chat-id origin-session --notifier-profile default --expected-version 3 "
        "--mutation-id subscribe-cli"
    )
    assert "Workflow wf_cli_reopen: ACTIVE v4" in subscribed
    denied = kc.run_slash("workflow show wf_cli_reopen --tenant tenant-b")
    assert "tenant" in denied.lower()


def test_workflow_cli_generates_mutation_id_and_reports_conflict(kanban_home):
    with kb.connect() as conn:
        acceptance = kb.create_task(
            conn, title="accept", assignee="orchestrator", tenant="tenant-a"
        )
    output = kc.run_slash(
        f"workflow create wf_generated --name generated --tenant tenant-a "
        f"--acceptance-task {acceptance}"
    )
    assert "wf_generated" in output
    conflict = kc.run_slash(
        "workflow cancel wf_generated --tenant tenant-a --reason stale "
        "--expected-version 99 --mutation-id stale-cancel"
    )
    assert "current version" in conflict.lower()


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------



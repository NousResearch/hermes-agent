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


def test_delivery_unknown_cli_lists_and_guardedly_reconciles_with_audit(kanban_home):
    from hermes_cli import kanban_db_notify as kbn

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="ambiguous delivery")
        kbn.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="operator-chat")
        kb.complete_task(conn, task_id, summary="done")
        sub = kbn.list_notify_subs(conn, task_id)[0]
        _, _, events = kbn.claim_unseen_events_for_sub(
            conn, task_id=task_id, platform="telegram", chat_id="operator-chat", kinds=["completed"],
        )
        row = kbn.enqueue_delivery(conn, event=events[0], sub=sub)
        claim = kbn.claim_delivery(conn, delivery_key=row["delivery_key"], now=100)
        assert claim is not None
        assert kbn.mark_delivery_ambiguous(
            conn, delivery_key=row["delivery_key"], lease_token=claim["lease_token"],
            error="transport timed out", now=101,
        )

    listed = json.loads(kc.run_slash("delivery-list --json"))
    assert [(item["delivery_key"], item["state"]) for item in listed] == [
        (row["delivery_key"], "delivery_unknown")
    ]

    refused = kc.run_slash(
        f"delivery-reconcile {row['delivery_key']} --action retry --reason operator-check"
    )
    assert "--accept-duplicate-risk" in refused
    with kbc.connect_closing() as conn:
        assert conn.execute(
            "SELECT state FROM kanban_delivery_outbox WHERE delivery_key=?", (row["delivery_key"],)
        ).fetchone()["state"] == "delivery_unknown"

    accepted = kc.run_slash(
        f"delivery-reconcile {row['delivery_key']} --action retry --reason operator-check "
        "--accept-duplicate-risk"
    )
    assert "Reconciled" in accepted
    with kbc.connect_closing() as conn:
        stored = conn.execute(
            "SELECT state,next_attempt_at,lease_token FROM kanban_delivery_outbox WHERE delivery_key=?",
            (row["delivery_key"],),
        ).fetchone()
        assert stored["state"] == "retry_wait"
        assert stored["lease_token"] is None
        audit = [event for event in kb.list_events(conn, task_id) if event.kind == "delivery_reconciled"]
        assert len(audit) == 1
        assert audit[0].payload["action"] == "retry"
        assert audit[0].payload["duplicate_risk_accepted"] is True
        assert audit[0].payload["reason"] == "operator-check"

    guarded = kc.run_slash(
        f"delivery-reconcile {row['delivery_key']} --action retry --reason again --accept-duplicate-risk"
    )
    assert "not in delivery_unknown" in guarded


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



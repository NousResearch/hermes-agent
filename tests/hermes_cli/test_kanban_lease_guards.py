from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban as kc
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from tools import kanban_tools


def _running_task(tmp_path, suffix):
    path = tmp_path / f"{suffix}.db"
    conn = kbc.connect(db_path=path)
    task_id = kb.create_task(
        conn, title=suffix, assignee="worker", tenant="business-a",
        workspace_kind="dir", workspace_path=str(tmp_path / f"workspace-{suffix}"),
    )
    assert kb.claim_task(conn, task_id, claimer=f"worker:{suffix}")
    task = kb.get_task(conn, task_id)
    guards = {
        "expected_run_id": task.current_run_id,
        "expected_claim_lock": task.claim_lock,
        "expected_tenant": task.tenant,
        "expected_workspace_path": task.workspace_path,
    }
    return conn, task_id, guards


def _snapshot(conn, task_id):
    return (
        asdict(kb.get_task(conn, task_id)),
        [asdict(run) for run in kb.list_runs(conn, task_id)],
        [asdict(event) for event in kb.list_events(conn, task_id)],
    )


@pytest.mark.parametrize("action", ("heartbeat", "reclaim", "complete"))
def test_old_lease_cannot_mutate_a_successor_attempt(tmp_path, action):
    conn, task_id, old = _running_task(tmp_path, action)
    try:
        assert kb.reclaim_task(conn, task_id, **old)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET tenant = ?, workspace_path = ? WHERE id = ?",
                ("business-b", str(tmp_path / "successor"), task_id),
            )
        assert kb.claim_task(conn, task_id, claimer=f"successor:{action}")
        before = _snapshot(conn, task_id)
        if action == "heartbeat":
            changed = kbd.heartbeat_worker(conn, task_id, note="stale", **old)
        elif action == "reclaim":
            changed = kb.reclaim_task(conn, task_id, reason="stale", **old)
        else:
            changed = kb.complete_task(
                conn, task_id, summary="stale", created_cards=["t_deadbeefcafe"], **old,
            )
        assert changed is False
        assert _snapshot(conn, task_id) == before
    finally:
        conn.close()


@pytest.mark.parametrize("action", ("heartbeat", "reclaim", "complete"))
def test_partial_new_lease_identity_is_rejected_without_mutation(tmp_path, action):
    conn, task_id, guards = _running_task(tmp_path, f"partial-{action}")
    try:
        before = _snapshot(conn, task_id)
        partial = {
            "expected_run_id": guards["expected_run_id"],
            "expected_claim_lock": guards["expected_claim_lock"],
        }
        with pytest.raises(ValueError, match="require expected_run_id"):
            if action == "heartbeat":
                kbd.heartbeat_worker(conn, task_id, **partial)
            elif action == "reclaim":
                kb.reclaim_task(conn, task_id, **partial)
            else:
                kb.complete_task(conn, task_id, summary="partial", **partial)
        assert _snapshot(conn, task_id) == before
    finally:
        conn.close()


def test_worker_tool_uses_all_environment_lease_fields(tmp_path, monkeypatch):
    conn, task_id, guards = _running_task(tmp_path, "worker-tool")
    path = tmp_path / "worker-tool.db"
    try:
        before = _snapshot(conn, task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(guards["expected_run_id"]))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", guards["expected_claim_lock"])
    monkeypatch.setenv("HERMES_KANBAN_TENANT", "stale-business")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", guards["expected_workspace_path"])
    monkeypatch.setattr(kanban_tools, "_is_dispatcher_owned_worker", lambda: True)

    assert "error" in json.loads(kanban_tools._handle_complete({"summary": "stale"}))
    explicit_run_only = SimpleNamespace(
        expected_run_id=guards["expected_run_id"], expected_claim_lock=None,
        expected_tenant=None, expected_workspace_path=None,
    )
    resolved = kc._lease_guard_kwargs(explicit_run_only, task_id)
    assert set(resolved) == {
        "expected_run_id", "expected_claim_lock", "expected_tenant",
        "expected_workspace_path",
    }
    with pytest.raises(ValueError, match="worker is scoped"):
        kc._lease_guard_kwargs(explicit_run_only, "foreign-task")
    with kbc.connect_closing(db_path=path) as check:
        assert _snapshot(check, task_id) == before


def test_failed_terminal_readback_rolls_back_to_the_guarded_running_lease(tmp_path):
    conn, task_id, guards = _running_task(tmp_path, "terminal-readback")
    try:
        conn.execute(
            "CREATE TRIGGER corrupt_terminal AFTER UPDATE OF status ON tasks "
            f"WHEN NEW.id = '{task_id}' AND NEW.status = 'done' BEGIN "
            "UPDATE tasks SET status = 'running', claim_lock = NULL WHERE id = NEW.id; END"
        )
        conn.commit()
        before = _snapshot(conn, task_id)
        with pytest.raises(kb.TerminalReadbackError):
            kb.complete_task(conn, task_id, summary="done", **guards)
        assert _snapshot(conn, task_id) == before
    finally:
        conn.close()

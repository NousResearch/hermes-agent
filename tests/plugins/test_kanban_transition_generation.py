"""Dashboard transitions reserve the observed run before requesting termination."""

from __future__ import annotations

import hermes_cli.kanban_claims as _owner_kanban_claims

import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.mark.parametrize("bulk", [False, True])
def test_transition_rejects_changed_generation_before_stop(tmp_path, monkeypatch, bulk):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_claims
    from hermes_cli import kanban_worker_identity

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    path = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    name = "kanban_transition_generation_test_api"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    app = FastAPI()
    app.include_router(module.router)
    client = TestClient(app)
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="dashboard generation", assignee="worker")
        assert _owner_kanban_claims.claim_task(conn, tid)

    stopped = []
    monkeypatch.setattr(kanban_worker_identity, "_terminate_reclaimed_worker",
                        lambda *a, **kw: stopped.append((a, kw)) or {"terminated": True})
    reserve = kanban_claims.reserve_reclaim
    successor = []

    def replace_then_reserve(conn, task_id, snapshot, **kwargs):
        conn.execute("UPDATE task_runs SET ended_at = 1 WHERE id = ?", (snapshot["current_run_id"],))
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, "
            "current_run_id = NULL WHERE id = ?", (task_id,))
        conn.commit()
        assert _owner_kanban_claims.claim_task(conn, task_id)
        successor.append(kb.get_task(conn, task_id).current_run_id)
        return reserve(conn, task_id, snapshot, **kwargs)

    monkeypatch.setattr(kanban_claims, "reserve_reclaim", replace_then_reserve)

    def move():
        if bulk:
            return client.post("/tasks/bulk", json={"ids": [tid], "status": "ready"})
        return client.patch(f"/tasks/{tid}", json={"status": "ready"})

    response = move()
    if bulk:
        assert response.status_code == 200
        assert response.json()["results"][0]["ok"] is False
    else:
        assert response.status_code == 409
    assert not stopped
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.current_run_id == successor[0]

    # The same request succeeds once it reserves the unchanged current run.
    monkeypatch.setattr(kanban_claims, "reserve_reclaim", reserve)
    response = move()
    assert response.status_code == 200
    if bulk:
        assert response.json()["results"][0]["ok"] is True
    assert stopped[0][1]["run_id"] == successor[0]
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "ready"
        row = conn.execute("SELECT reclaim_reserved_at, last_heartbeat_at FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["reclaim_reserved_at"] is None
        assert row["last_heartbeat_at"] is None

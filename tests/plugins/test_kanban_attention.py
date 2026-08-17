"""Human attention receipts stay durable and independent of Kanban workflow."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _load_router():
    path = Path(__file__).resolve().parents[2] / "plugins/kanban/dashboard/plugin_api.py"
    spec = importlib.util.spec_from_file_location("kanban_attention_test_plugin", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.router


@pytest.fixture
def client(kanban_home):
    app = FastAPI()
    app.include_router(_load_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def create(client, status="ready"):
    response = client.post("/api/plugins/kanban/tasks", json={"title": "receipt", "assignee": "worker", "status": status})
    assert response.status_code == 200, response.text
    return response.json()["task"]


def act(client, task_id, action, key, revision=0, wake_at=None):
    body = {"action": action, "actor": "captain", "source": "test", "expected_revision": revision, "idempotency_key": key}
    if wake_at is not None:
        body["wake_at"] = wake_at
    return client.post(f"/api/plugins/kanban/tasks/{task_id}/attention", json=body)


def test_schema_is_additive_and_upgrade_safe(kanban_home):
    conn = kb.connect()
    names = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"tasks", "attention_receipts", "attention_receipt_events"} <= names
    conn.close()
    # Reopening (the supported additive migration/rollback boundary) is idempotent.
    kb.init_db()


def test_settle_wake_and_snooze_never_mutate_workflow(client):
    task = create(client, status="ready")
    settled = act(client, task["id"], "settle", "settle-1")
    assert settled.status_code == 200, settled.text
    assert settled.json()["attention"]["state"] == "settled"
    assert client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["status"] == "ready"

    snoozed = act(client, task["id"], "snooze", "snooze-1", revision=1, wake_at=int(time.time()) + 3600)
    assert snoozed.status_code == 200, snoozed.text
    assert snoozed.json()["attention"]["state"] == "snoozed"
    woke = act(client, task["id"], "wake", "wake-1", revision=2)
    assert woke.status_code == 200, woke.text
    assert woke.json()["attention"]["state"] == "active"
    assert client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["status"] == "ready"


def test_idempotency_and_concurrent_revision_conflicts(client):
    task = create(client)
    first = act(client, task["id"], "settle", "same")
    assert first.status_code == 200
    duplicate = act(client, task["id"], "settle", "same")
    assert duplicate.status_code == 200 and duplicate.json()["idempotent"] is True
    conflict = act(client, task["id"], "wake", "same", revision=1)
    assert conflict.status_code == 409
    stale = act(client, task["id"], "wake", "other", revision=0)
    assert stale.status_code == 409


def test_activity_resurfaces_settled_and_snoozed(client):
    task = create(client)
    assert act(client, task["id"], "settle", "settle").status_code == 200
    # A canonical task comment appends consequential activity.
    comment = client.post(f"/api/plugins/kanban/tasks/{task['id']}/comments", json={"author": "worker", "body": "new evidence"})
    assert comment.status_code == 200
    projected = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["attention"]
    assert projected["state"] == "active" and projected["reason"] == "activity"


def test_expiry_recovers_after_restart_and_invalid_inputs_fail_closed(client, kanban_home):
    task = create(client)
    past = act(client, task["id"], "snooze", "past", wake_at=int(time.time()) - 1)
    assert past.status_code == 422
    unknown = act(client, "t_missing", "settle", "missing")
    assert unknown.status_code == 404

    future = int(time.time()) + 3600
    assert act(client, task["id"], "snooze", "future", wake_at=future).status_code == 200
    # Simulate restart/offline passage by moving the durable deadline backward.
    conn = kb.connect()
    conn.execute("UPDATE attention_receipts SET wake_at = ? WHERE subject_id = ?", (int(time.time()) - 1, task["id"]))
    conn.commit()
    conn.close()
    projected = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["attention"]
    assert projected["state"] == "active" and projected["reason"] == "expired"


def test_receipt_audit_is_append_only(client, kanban_home):
    task = create(client, status="review")
    status_before = client.get(f"/api/plugins/kanban/tasks/{task['id']}").json()["task"]["status"]
    assert act(client, task["id"], "settle", "audit").status_code == 200
    conn = kb.connect()
    audit = conn.execute("SELECT action, actor, source, revision FROM attention_receipt_events WHERE subject_id = ?", (task["id"],)).fetchall()
    status = conn.execute("SELECT status FROM tasks WHERE id = ?", (task["id"],)).fetchone()["status"]
    conn.close()
    assert [tuple(row) for row in audit] == [("settle", "captain", "test", 1)]
    assert status == status_before

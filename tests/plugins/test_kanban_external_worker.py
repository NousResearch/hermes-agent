"""Dashboard lifecycle guards for externally owned Kanban runs."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_external_worker as xw


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_external_worker_test", plugin_file
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.router


@pytest.fixture
def client(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/kanban")
    return TestClient(app)


def _create_external_run(*, board: str = "default") -> tuple[str, int, xw.Lease]:
    with xw.connect(board=board) as conn:
        task_id = kb.create_task(
            conn, title="external", assignee="mas-runner", triage=True
        )
        raw = json.dumps(
            {
                "schema_version": xw.SPEC_SCHEMA_VERSION,
                "board": board,
                "repo_key": "fixture",
                "objective": "test dashboard conflicts",
                "acceptance_criteria": ["conflict is HTTP 409"],
                "scope": {"include": ["src/**"], "exclude": [".env"]},
                "base_sha": "a" * 40,
                "risk": "small",
                "workflow": "implement-verify",
                "execution": {
                    "timeout_seconds": 60,
                    "max_attempts": 1,
                    "max_tokens": 1000,
                    "max_cost_usd": None,
                },
                "verification": {
                    "check_ids": ["focused"],
                    "fresh_reviewer": True,
                    "security_review": False,
                },
                "delivery": {
                    "mode": "review-only",
                    "push": False,
                    "deploy": False,
                },
            },
            sort_keys=True,
        ).encode()
        attachment_id = kb.store_attachment_bytes(
            conn,
            task_id,
            filename=xw.SPEC_ATTACHMENT_NAME,
            data=raw,
        )
        spec = xw.submit(
            conn, task_id=task_id, spec_attachment_id=attachment_id
        )
        lease = xw.claim_external(
            conn,
            task_id=task_id,
            expected_spec=spec,
            lease_token="dashboard-test",
            lease_expires_at=int(time.time()) + 600,
        )
        return task_id, attachment_id, lease


@pytest.mark.parametrize(
    "status",
    ["ready", "todo", "triage", "scheduled", "blocked", "done", "archived"],
)
def test_status_mutations_return_typed_409(client, status):
    task_id, _attachment_id, lease = _create_external_run()
    response = client.patch(
        f"/api/plugins/kanban/tasks/{task_id}",
        json={"status": status, "block_reason": "manual"},
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "external_run_active"
    assert detail["task_id"] == task_id
    assert detail["run_id"] == lease.run_id
    with xw.connect() as conn:
        assert kb.get_task(conn, task_id).status == "running"
        assert kb.get_run(conn, lease.run_id).ended_at is None


def test_dashboard_actions_return_typed_409(client):
    task_id, _attachment_id, lease = _create_external_run()
    responses = [
        client.patch(
            f"/api/plugins/kanban/tasks/{task_id}", json={"assignee": "other"}
        ),
        client.post(f"/api/plugins/kanban/tasks/{task_id}/reclaim", json={}),
        client.post(
            f"/api/plugins/kanban/tasks/{task_id}/reassign",
            json={"profile": "other", "reclaim_first": True},
        ),
        client.post(
            f"/api/plugins/kanban/runs/{lease.run_id}/terminate", json={}
        ),
        client.delete(f"/api/plugins/kanban/tasks/{task_id}"),
    ]
    for response in responses:
        assert response.status_code == 409, response.text
        detail = response.json()["detail"]
        assert detail["code"] == "external_run_active"
        assert detail["run_id"] == lease.run_id


def test_spec_attachment_delete_returns_typed_409(client):
    task_id, attachment_id, _lease = _create_external_run()
    response = client.delete(f"/api/plugins/kanban/attachments/{attachment_id}")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "code": "external_spec_locked",
        "task_id": task_id,
        "operation": "delete_attachment",
        "run_id": None,
    }


def test_bulk_mutation_reports_typed_conflict(client):
    task_id, _attachment_id, lease = _create_external_run()
    response = client.post(
        "/api/plugins/kanban/tasks/bulk",
        json={"ids": [task_id], "status": "done"},
    )
    assert response.status_code == 200
    assert response.json()["results"] == [
        {
            "id": task_id,
            "ok": False,
            "error": (
                f"complete_task refused for external task {task_id}: "
                f"external_run_active (run_id={lease.run_id})"
            ),
            "code": "external_run_active",
            "run_id": lease.run_id,
        }
    ]


def test_board_delete_returns_typed_409_for_active_external_run(client):
    kb.create_board("named")
    task_id, _attachment_id, lease = _create_external_run(board="named")
    response = client.delete("/api/plugins/kanban/boards/named")
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "code": "external_run_active",
        "task_id": task_id,
        "operation": "remove_board",
        "run_id": lease.run_id,
    }
    assert kb.board_exists("named")


def test_reopening_parent_cannot_collateral_demote_external_child(client):
    with xw.connect() as conn:
        parent_id = kb.create_task(conn, title="parent", body="p", triage=True)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
                (int(time.time()), parent_id),
            )
        child_id = kb.create_task(conn, title="child", body="c", triage=True)
        raw = json.dumps(
            {
                "schema_version": xw.SPEC_SCHEMA_VERSION,
                "board": "default",
                "repo_key": "fixture",
                "objective": "stay externally owned",
                "acceptance_criteria": ["parent reopen conflicts"],
                "scope": {"include": ["src/**"], "exclude": [".env"]},
                "base_sha": "a" * 40,
                "risk": "small",
                "workflow": "implement-verify",
                "execution": {
                    "timeout_seconds": 60,
                    "max_attempts": 1,
                    "max_tokens": 1000,
                    "max_cost_usd": None,
                },
                "verification": {
                    "check_ids": ["focused"],
                    "fresh_reviewer": True,
                    "security_review": False,
                },
                "delivery": {
                    "mode": "review-only",
                    "push": False,
                    "deploy": False,
                },
            },
            sort_keys=True,
        ).encode()
        attachment_id = kb.store_attachment_bytes(
            conn,
            child_id,
            filename=xw.SPEC_ATTACHMENT_NAME,
            data=raw,
        )
        kb.link_tasks(conn, parent_id, child_id)
        spec = xw.submit(
            conn, task_id=child_id, spec_attachment_id=attachment_id
        )
        lease = xw.claim_external(
            conn,
            task_id=child_id,
            expected_spec=spec,
            lease_token="child",
            lease_expires_at=int(time.time()) + 600,
        )

    unlink_response = client.delete(
        "/api/plugins/kanban/links",
        params={"parent_id": parent_id, "child_id": child_id},
    )
    assert unlink_response.status_code == 409, unlink_response.text
    assert unlink_response.json()["detail"]["run_id"] == lease.run_id

    response = client.patch(
        f"/api/plugins/kanban/tasks/{parent_id}", json={"status": "todo"}
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["run_id"] == lease.run_id
    with xw.connect() as conn:
        assert kb.get_task(conn, parent_id).status == "done"
        assert kb.get_task(conn, child_id).status == "running"

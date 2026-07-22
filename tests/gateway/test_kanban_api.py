"""Behavioral contract tests for the API-server Kanban surface."""

from __future__ import annotations

import pytest
import json

from gateway.kanban_api import KanbanApiError, create_task, get_task, mutate_task
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli import kanban_db


class _Request:
    def __init__(self, *, authorization: str = ""):
        self.headers = {"Authorization": authorization} if authorization else {}
        self.query = {}
        self.match_info = {}
        self.transport = None
        self.remote = "127.0.0.1"
        self.method = "GET"
        self.path_qs = "/v1/kanban/boards"


def _create_payload(key: str = "create-key-0001") -> dict:
    return {
        "title": "Ship revisioned Kanban",
        "body": "Keep the browser away from dashboard tokens.",
        "assignee": None,
        "workspace_kind": "scratch",
        "priority": 3,
        "idempotency_key": key,
    }


def test_create_replays_only_the_exact_idempotent_request():
    created = create_task("default", _create_payload())
    task_id = created["task"]["id"]
    assert created["replayed"] is False
    assert created["revision"].startswith("kanbanrev_")

    replayed = create_task("default", _create_payload())
    assert replayed["replayed"] is True
    assert replayed["task"]["id"] == task_id

    changed = _create_payload()
    changed["title"] = "Different task"
    with pytest.raises(KanbanApiError, match="Idempotency key") as exc_info:
        create_task("default", changed)
    assert exc_info.value.status == 409
    assert exc_info.value.code == "idempotency_conflict"


def test_revision_action_is_atomic_stale_safe_and_replay_safe():
    created = create_task("default", _create_payload("create-key-0002"))
    task_id, revision = created["task"]["id"], created["revision"]
    action = {
        "action": "comment",
        "author": "mentat",
        "body": "Please verify the post-action read-back.",
        "expected_revision": revision,
        "idempotency_key": "comment-key-0002",
    }
    mutated = mutate_task("default", task_id, action)
    assert mutated["replayed"] is False
    assert mutated["comments"][-1]["body"] == action["body"]
    assert mutated["revision"] != revision

    replayed = mutate_task("default", task_id, action)
    assert replayed["replayed"] is True
    assert len(replayed["comments"]) == 1

    stale = dict(action, idempotency_key="comment-key-0003")
    with pytest.raises(KanbanApiError, match="refresh") as exc_info:
        mutate_task("default", task_id, stale)
    assert exc_info.value.status == 409
    assert exc_info.value.code == "stale_revision"


def test_snapshot_omits_event_payload_and_local_execution_details():
    created = create_task("default", _create_payload("create-key-0003"))
    task_id = created["task"]["id"]
    with kanban_db.connect_closing(board="default") as conn, kanban_db.write_txn(conn):
        kanban_db._append_event(
            conn,
            task_id,
            "private_worker_detail",
            {"path": "/Users/operator/private", "token": "secret"},
        )
    snapshot = get_task("default", task_id)
    assert snapshot["events"][-1] == {
        "id": snapshot["events"][-1]["id"],
        "kind": "private_worker_detail",
        "created_at": snapshot["events"][-1]["created_at"],
        "run_id": None,
    }
    assert "/Users/operator/private" not in str(snapshot)
    assert "secret" not in str(snapshot)
    assert "workspace_path" not in str(snapshot)
    assert "worker_pid" not in str(snapshot)


def test_outer_write_transaction_supports_existing_mutation_helpers():
    with kanban_db.connect_closing(board="default") as conn, kanban_db.write_txn(conn):
        task_id = kanban_db.create_task(conn, title="nested transaction")
        assert kanban_db.assign_task(conn, task_id, "builder") is True
    with kanban_db.connect_closing(board="default") as conn:
        assert kanban_db.get_task(conn, task_id).assignee == "builder"


@pytest.mark.asyncio
async def test_http_route_requires_configured_api_key_before_kanban_access():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    response = await adapter._handle_kanban_boards(_Request())
    assert response.status == 403
    assert json.loads(response.body)["error"]["code"] == "kanban_api_auth_required"


@pytest.mark.asyncio
async def test_http_route_uses_normal_bearer_auth_and_advertises_contract():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    denied = await adapter._handle_kanban_boards(_Request())
    assert denied.status == 401
    allowed = await adapter._handle_kanban_boards(_Request(authorization="Bearer sk-secret"))
    assert allowed.status == 200
    capabilities = await adapter._handle_capabilities(_Request(authorization="Bearer sk-secret"))
    data = json.loads(capabilities.body)
    assert data["features"]["kanban_api"] is True
    assert data["features"]["kanban_api_revisioned"] is True
    assert data["endpoints"]["kanban_task_action"]["path"].startswith("/v1/kanban/")


def test_http_route_table_includes_every_kanban_contract_path():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    routes = {(method, path) for method, path, _handler in adapter._http_route_table()}
    assert {
        ("GET", "/v1/kanban/boards"),
        ("GET", "/v1/kanban/profiles"),
        ("GET", "/v1/kanban/tasks"),
        ("POST", "/v1/kanban/tasks"),
        ("GET", "/v1/kanban/tasks/{task_id}"),
        ("POST", "/v1/kanban/tasks/{task_id}/actions"),
    } <= routes

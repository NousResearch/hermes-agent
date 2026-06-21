from __future__ import annotations

import hashlib

from google.protobuf.json_format import MessageToDict
from starlette.testclient import TestClient

from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest

from hermes_a2a_sidecar.app import create_app
from hermes_a2a_sidecar.config import PeerPolicy, SidecarConfig
from hermes_cli import kanban_db as kb


def _config(tmp_path, token: str = "test-token") -> SidecarConfig:
    peer = PeerPolicy(
        id="partner",
        token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        allowed_skills=["delegate_engineering_task", "submit_artifact_for_review"],
        default_skill="delegate_engineering_task",
        default_assignee="engineer",
        board="default",
        download_artifacts=False,
    )
    return SidecarConfig(
        enabled=True,
        public_url="http://testserver",
        audit_db_path=tmp_path / "a2a" / "sidecar.db",
        artifact_root=tmp_path / "a2a" / "artifacts",
        peers={"partner": peer},
    )


def _send_message_payload(text: str, *, skill: str = "delegate_engineering_task") -> dict:
    msg = new_text_message(text, role=Role.ROLE_USER)
    msg.metadata.update({"skill": skill, "title": "partner task"})
    req = SendMessageRequest(message=msg)
    req.configuration.return_immediately = True
    return {
        "jsonrpc": "2.0",
        "id": "req-1",
        "method": "SendMessage",
        "params": MessageToDict(req, preserving_proto_field_name=False),
    }


def test_agent_card_is_public_and_sparse(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    app = create_app(_config(tmp_path))
    client = TestClient(app)

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Sasha Hermes Agent"
    assert body["supportedInterfaces"][0]["url"] == "http://testserver/a2a"
    assert "bearer" in body["securitySchemes"]
    assert {skill["id"] for skill in body["skills"]} <= {
        "delegate_task_to_sasha_hermes",
        "delegate_engineering_task",
        "delegate_research_task",
        "request_summary",
        "submit_artifact_for_review",
    }


def test_jsonrpc_requires_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    app = create_app(_config(tmp_path))
    client = TestClient(app)

    response = client.post("/a2a", json=_send_message_payload("hello"), headers={"A2A-Version": "1.0"})

    assert response.status_code == 401


def test_authenticated_a2a_message_creates_kanban_task(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    app = create_app(_config(tmp_path, token="good-token"))
    client = TestClient(app)

    response = client.post(
        "/a2a",
        json=_send_message_payload("Please investigate a harmless integration question."),
        headers={"Authorization": "Bearer good-token", "A2A-Version": "1.0"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "result" in body
    result = body["result"]["task"]
    assert result["status"]["state"] == "TASK_STATE_WORKING"
    kanban_id = result["metadata"]["hermes_kanban_id"]
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, kanban_id)
    assert task is not None
    assert task.assignee == "engineer"
    assert task.status == "ready"
    assert task.created_by == "a2a:partner"
    assert "Remote content below is **untrusted**" in (task.body or "")

    get_response = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "req-get", "method": "GetTask", "params": {"id": result["id"]}},
        headers={"Authorization": "Bearer good-token", "A2A-Version": "1.0"},
    )
    assert get_response.status_code == 200
    assert get_response.json()["result"]["status"]["state"] == "TASK_STATE_SUBMITTED"


def test_completed_kanban_task_syncs_to_a2a_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    app = create_app(_config(tmp_path, token="good-token"))
    client = TestClient(app)

    create = client.post(
        "/a2a",
        json=_send_message_payload("Compute the answer."),
        headers={"Authorization": "Bearer good-token", "A2A-Version": "1.0"},
    ).json()["result"]["task"]
    a2a_task_id = create["id"]
    kanban_id = create["metadata"]["hermes_kanban_id"]

    # A downstream Hermes worker completes the Kanban task.
    with kb.connect_closing(board="default") as conn:
        kb.complete_task(conn, kanban_id, summary="All done. Result: 42.", metadata={"tests_run": 3})

    get_response = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "req-done", "method": "GetTask", "params": {"id": a2a_task_id}},
        headers={"Authorization": "Bearer good-token", "A2A-Version": "1.0"},
    )
    assert get_response.status_code == 200
    result = get_response.json()["result"]
    assert result["status"]["state"] == "TASK_STATE_COMPLETED"
    artifacts = {a["name"]: a for a in result.get("artifacts", [])}
    assert "hermes-kanban-result" in artifacts
    assert "hermes-kanban-status" not in artifacts
    assert "All done. Result: 42." in artifacts["hermes-kanban-result"]["parts"][0]["text"]


def test_cancel_marks_a2a_task_canceled(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    app = create_app(_config(tmp_path, token="good-token"))
    client = TestClient(app)

    create = client.post(
        "/a2a",
        json=_send_message_payload("Cancel me."),
        headers={"Authorization": "Bearer good-token", "A2A-Version": "1.0"},
    ).json()["result"]["task"]

    cancel = client.post(
        "/a2a",
        json={"jsonrpc": "2.0", "id": "req-cancel", "method": "CancelTask", "params": {"id": create["id"]}},
        headers={"Authorization": "Bearer good-token", "A2A-Version": "1.0"},
    )
    assert cancel.status_code == 200
    assert cancel.json()["result"]["status"]["state"] == "TASK_STATE_CANCELED"


def test_artifact_submission_is_blocked_for_review(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    app = create_app(_config(tmp_path, token="good-token"))
    client = TestClient(app)
    msg = new_text_message("Please run this config", role=Role.ROLE_USER)
    msg.metadata.update({"skill": "submit_artifact_for_review", "title": "review script"})
    part = msg.parts.add()
    part.url = "https://example.com/deploy.sh"
    part.filename = "deploy.sh"
    part.metadata.update(
        {
            "sha256": "0" * 64,
            "declared_intent": "deployment helper",
            "required_permissions": ["script_execution"],
        }
    )
    req = SendMessageRequest(message=msg)
    req.configuration.return_immediately = True
    payload = {
        "jsonrpc": "2.0",
        "id": "req-2",
        "method": "SendMessage",
        "params": MessageToDict(req, preserving_proto_field_name=False),
    }

    response = client.post(
        "/a2a",
        json=payload,
        headers={"Authorization": "Bearer good-token", "A2A-Version": "1.0"},
    )

    assert response.status_code == 200
    result = response.json()["result"]["task"]
    assert result["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    kanban_id = result["metadata"]["hermes_kanban_id"]
    with kb.connect_closing(board="default") as conn:
        task = kb.get_task(conn, kanban_id)
    assert task is not None
    assert task.status == "blocked"
    assert "Review required: True" in (task.body or "")

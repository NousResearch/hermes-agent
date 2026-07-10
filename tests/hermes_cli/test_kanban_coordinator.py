from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_coordinator import create_app


def test_coordinator_registers_and_claims_only_eligible_machine(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path)
    with kb.connect(db_path) as conn:
        mac_task = kb.create_task(
            conn,
            title="Xcode work",
            assignee="ios",
            required_capabilities=["macos", "xcode"],
        )
        linux_task = kb.create_task(conn, title="Linux work", assignee="ops")

    app = create_app(db_path=db_path, token="secret")
    client = TestClient(app)
    headers = {"Authorization": "Bearer secret"}
    linux_id = str(uuid.uuid4())
    mac_id = str(uuid.uuid4())

    assert client.get("/v1/health").status_code == 401
    assert client.post(
        "/v1/machines/register",
        headers=headers,
        json={
            "machine_id": linux_id,
            "hostname": "linux",
            "profiles": ["ops"],
            "capabilities": ["linux"],
        },
    ).status_code == 200

    assert client.post(
        "/v1/machines/register",
        headers=headers,
        json={
            "machine_id": mac_id,
            "hostname": "mac",
            "profiles": ["ios"],
            "capabilities": ["macos", "xcode"],
        },
    ).status_code == 200

    machines = client.get("/v1/machines", headers=headers).json()
    assert machines["count"] == 2
    assert {item["hostname"] for item in machines["machines"]} == {"linux", "mac"}
    mac = next(item for item in machines["machines"] if item["hostname"] == "mac")
    assert mac["online"] is True
    assert mac["profiles"] == ["ios"]
    assert mac["capabilities"] == ["macos", "xcode"]

    first = client.post(
        "/v1/tasks/claim-next", headers=headers, json={"machine_id": linux_id},
    ).json()["task"]
    assert first["id"] == linux_task
    assert first["machine_id"] == linux_id

    second = client.post(
        "/v1/tasks/claim-next", headers=headers, json={"machine_id": mac_id},
    ).json()["task"]
    assert second["id"] == mac_task
    assert second["required_capabilities"] == ["macos", "xcode"]

    assert client.post(
        f"/v1/tasks/{mac_task}/renew",
        headers=headers,
        json={"claim_lock": second["claim_lock"]},
    ).json() == {"renewed": True}
    assert client.post(
        f"/v1/tasks/{mac_task}/worker-started",
        headers=headers,
        json={"claim_lock": second["claim_lock"], "worker_pid": 4242},
    ).json() == {"recorded": True}
    with kb.connect(db_path) as conn:
        assert kb.get_task(conn, mac_task).worker_pid == 4242
        assert kb.latest_run(conn, mac_task).worker_pid == 4242
    assert client.post(
        f"/v1/tasks/{mac_task}/complete",
        headers=headers,
        json={"result": "wrong lock", "claim_lock": "not-the-owner"},
    ).json() == {"completed": False}
    assert client.post(
        f"/v1/tasks/{mac_task}/complete",
        headers=headers,
        json={"result": "finished", "claim_lock": second["claim_lock"]},
    ).json() == {"completed": True}


def test_coordinator_rejects_invalid_machine_identity_without_500(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path)
    client = TestClient(create_app(db_path=db_path, token="secret"))
    headers = {"Authorization": "Bearer secret"}

    response = client.post(
        "/v1/machines/register",
        headers=headers,
        json={"machine_id": "not-a-uuid"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "machine_id must be a UUID"

    response = client.post(
        "/v1/tasks/claim-next",
        headers=headers,
        json={"machine_id": "not-a-uuid"},
    )
    assert response.status_code == 422

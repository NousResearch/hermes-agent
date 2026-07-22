from __future__ import annotations

import time
import uuid

from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_coordinator import _maintenance_once, create_app
from hermes_cli.kanban_remote import RemoteConnection, RemoteKanban, RemoteKanbanError


def _remote_client(client: TestClient, headers: dict[str, str]) -> RemoteKanban:
    remote = RemoteKanban("http://coordinator", "secret")

    def request(method: str, path: str, body=None) -> dict:
        response = client.request(method, path, headers=headers, json=body)
        payload = response.json()
        if response.status_code >= 400:
            raise RemoteKanbanError(response.status_code, payload)
        return payload

    remote._request = request
    return remote


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
    assert (
        client.post(
            "/v1/machines/register",
            headers=headers,
            json={
                "machine_id": linux_id,
                "hostname": "linux",
                "profiles": ["ops"],
                "capabilities": ["linux"],
            },
        ).status_code
        == 200
    )

    assert (
        client.post(
            "/v1/machines/register",
            headers=headers,
            json={
                "machine_id": mac_id,
                "hostname": "mac",
                "profiles": ["ios"],
                "capabilities": ["macos", "xcode"],
            },
        ).status_code
        == 200
    )

    machines = client.get("/v1/machines", headers=headers).json()
    assert machines["count"] == 2
    assert {item["hostname"] for item in machines["machines"]} == {"linux", "mac"}
    mac = next(item for item in machines["machines"] if item["hostname"] == "mac")
    assert mac["online"] is True
    assert mac["profiles"] == ["ios"]
    assert mac["capabilities"] == ["macos", "xcode"]

    first = client.post(
        "/v1/tasks/claim-next",
        headers=headers,
        json={"machine_id": linux_id},
    ).json()["task"]
    assert first["id"] == linux_task
    assert first["machine_id"] == linux_id

    second = client.post(
        "/v1/tasks/claim-next",
        headers=headers,
        json={"machine_id": mac_id},
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


def test_coordinator_maintenance_recovers_remote_lease_without_pid_control(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path)
    machine_id = str(uuid.uuid4())
    with kb.connect(db_path) as conn:
        kb.register_machine(
            conn,
            machine_id,
            hostname="remote",
            profiles=["ios"],
            capabilities=["macos"],
        )
        task_id = kb.create_task(conn, title="Remote work", assignee="ios")
        claimed = kb.claim_task(
            conn,
            task_id,
            machine_id=machine_id,
            enforce_machine_routing=True,
            require_registered_profile=True,
        )
        assert claimed is not None
        expired = int(time.time()) - 1
        conn.execute(
            "UPDATE tasks SET claim_expires = ?, worker_pid = ? WHERE id = ?",
            (expired, 4242, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET claim_expires = ?, worker_pid = ? WHERE id = ?",
            (expired, 4242, claimed.current_run_id),
        )
        parent = kb.create_task(conn, title="Finished parent", assignee="ios")
        child = kb.create_task(
            conn,
            title="Waiting child",
            assignee="ios",
            parents=[parent],
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))

    def forbidden_kill(*_args):
        raise AssertionError("coordinator attempted remote PID control")

    monkeypatch.setattr(kb.os, "kill", forbidden_kill)

    assert _maintenance_once(db_path) == {"reclaimed": 1, "promoted": 1}

    with kb.connect(db_path) as conn:
        recovered = kb.get_task(conn, task_id)
        assert recovered.status == "ready"
        assert recovered.claim_lock is None
        assert kb.latest_run(conn, task_id).outcome == "reclaimed"
        assert kb.get_task(conn, child).status == "ready"
        claimed_again = kb.claim_task(
            conn,
            task_id,
            machine_id=machine_id,
            enforce_machine_routing=True,
            require_registered_profile=True,
        )
        assert claimed_again is not None


def test_remote_backend_supports_orchestrator_contract_and_structured_handoff(
    tmp_path,
):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path)
    client = TestClient(create_app(db_path=db_path, token="secret"))
    headers = {"Authorization": "Bearer secret"}
    remote = _remote_client(client, headers)
    conn = RemoteConnection()

    root = remote.create_task(
        conn,
        title="Root",
        assignee="ios",
        created_by="orchestrator",
    )
    created_card = remote.create_task(
        conn,
        title="Follow-up",
        assignee="ios",
        created_by="ios",
    )
    listed = remote.list_tasks(conn, assignee="ios", limit=10)
    assert {task.id for task in listed} == {root, created_card}

    assert remote.complete_task(
        conn,
        root,
        result="full result",
        summary="handoff summary",
        metadata={"tests": ["contract"]},
        created_cards=[created_card],
    )
    with kb.connect(db_path) as local_conn:
        run = kb.latest_run(local_conn, root)
        assert run.summary == "handoff summary"
        assert run.metadata == {"tests": ["contract"]}
        completed = [
            event
            for event in kb.list_events(local_conn, root)
            if event.kind == "completed"
        ][-1]
        assert completed.payload["verified_cards"] == [created_card]

    parent = remote.create_task(
        conn,
        title="Parent",
        assignee="ios",
    )
    child = remote.create_task(
        conn,
        title="Child",
        assignee="ios",
    )
    remote.link_tasks(conn, parent, child)
    assert remote.get_task(conn, child).status == "todo"
    assert remote.complete_task(conn, parent, result="done")
    assert remote.recompute_ready(conn) == 0
    assert remote.get_task(conn, child).status == "ready"

    assert remote.block_task(conn, child, reason="needs input", kind="needs_input")
    assert remote.unblock_task(conn, child)
    assert remote.get_task(conn, child).status == "ready"

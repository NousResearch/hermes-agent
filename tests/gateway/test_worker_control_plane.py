"""Behavioral tests for the isolated, test-only Worker Control Plane."""

from __future__ import annotations

import asyncio
import inspect
import os
import uuid

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from gateway.worker_control_plane.app import create_worker_control_plane_app
from gateway.worker_control_plane.config import WorkerControlPlaneSettings
from gateway.worker_control_plane.service import WorkerControlPlaneService
from tests.gateway.worker_control_plane_helpers import MockWorkerClient


@pytest_asyncio.fixture
async def control_plane(tmp_path):
    settings = WorkerControlPlaneSettings.for_test(
        tmp_path / "worker-control-plane.db", approved_test_root=tmp_path
    )
    service = WorkerControlPlaneService(settings)
    secret = service.seed_test_worker()
    app = create_worker_control_plane_app(settings, service)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        yield service, client, secret
    finally:
        await client.close()
        service.close()


def _register_direct_test_worker(service):
    secret = service.seed_test_worker()
    instance_id = str(uuid.uuid4())
    status, response = service.register_worker(
        {
            "protocol_version": "1.0",
            "worker_id": "server-a-worker",
            "instance_id": instance_id,
            "worker_name": "test worker",
            "worker_version": "0.1.0",
            "capabilities": ["system.echo"],
        },
        secret,
    )
    assert status == 201
    identity = {
        "worker_id": "server-a-worker",
        "instance_id": instance_id,
        "registration_id": response["registration_id"],
    }
    return identity, response["access_token"]


def _direct_poll(service, identity, token, key):
    return service.poll_one_task(
        identity
        | {
            "capabilities": ["system.echo"],
            "max_tasks": 1,
            "wait_seconds": 0,
        },
        token,
        key,
    )


def _direct_temporary_nack(service, identity, token, task, key):
    return service.ack_delivery(
        task["task_id"],
        identity
        | {
            "delivery_id": task["delivery_id"],
            "accepted": False,
            "reason": "temporary",
            "worker_time": "2026-01-01T00:00:00Z",
        },
        token,
        key,
    )


@pytest.mark.asyncio
async def test_full_echo_lifecycle_and_idempotent_result(control_plane):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    task_id = service.create_test_echo_task({"message": "你好 🌍"}, "create-1")
    assert (await worker.heartbeat())[0] == 200
    status, envelope = await worker.poll()
    assert status == 200
    task = envelope["task"]
    assert task["task_id"] == task_id
    assert (await worker.ack(task))[0] == 200
    status, body = await worker.result(task)
    assert status == 200 and body["task_state"] == "completed"
    status, replay = await worker.result(task)
    assert status == 200 and replay == body
    assert service.task_state(task_id) == "completed"
    assert service.result_count(task_id) == 1


@pytest.mark.asyncio
async def test_registration_auth_capability_and_instance_guards(control_plane):
    _, client, secret = control_plane
    bad = MockWorkerClient(client, "not-the-secret")
    assert (await bad.register())[0] == 401
    unknown = MockWorkerClient(client, secret, worker_id="unknown-worker")
    assert (await unknown.register())[0] == 401
    worker = MockWorkerClient(client, secret)
    assert (await worker.register(capabilities=["codex.task"]))[0] == 422
    assert (await worker.register(protocol_version="2.0"))[0] == 422
    assert (await worker.register())[0] == 201
    assert (await worker.register())[0] == 200
    second = MockWorkerClient(client, secret)
    assert (await second.register())[0] == 409


@pytest.mark.asyncio
async def test_poll_ack_result_validation_and_fifo(control_plane):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    await worker.register()
    assert (await worker.poll())[0] == 204
    first = service.create_test_echo_task({"message": "first"}, "fifo-1")
    service.create_test_echo_task({"message": "second"}, "fifo-2")
    status, envelope = await worker.poll("same-poll")
    assert status == 200 and envelope["task"]["task_id"] == first
    status2, replay = await worker.poll("same-poll")
    assert status2 == 200 and replay["task"]["delivery_id"] == envelope["task"]["delivery_id"]
    task = envelope["task"]
    assert (await worker.ack(task))[0] == 200
    wrong_hash = dict(task)
    wrong_hash["payload_hash"] = "0" * 64
    assert (await worker.result(wrong_hash))[0] == 422
    assert (await worker.result(task, stdout="not echo", result_key="bad-result", request_key="bad-request"))[0] == 422
    assert (await worker.result(task))[0] == 200


@pytest.mark.asyncio
async def test_negative_ack_expiry_redelivery_and_dead_letter(control_plane):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    await worker.register()
    task_id = service.create_test_echo_task({"message": "retry"}, "retry-1")
    _, env = await worker.poll("p1")
    task = env["task"]
    assert (await worker.ack(task, accepted=False, reason="temporary", key="a1"))[0] == 200
    _, env = await worker.poll("p2")
    task = env["task"]
    service.advance_for_test(120)
    service.reap_expired_deliveries()
    assert service.task_state(task_id) == "queued"
    for index in range(3):
        _, env = await worker.poll(f"expire-{index}")
        if env is None:
            break
        service.advance_for_test(120)
        service.reap_expired_deliveries()
    assert service.task_state(task_id) == "dead_letter"


@pytest.mark.asyncio
async def test_wrong_registration_revocation_and_size_limits_are_safe(control_plane):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    await worker.register()
    task_id = service.create_test_echo_task({"message": "x"}, "size-1")
    _, env = await worker.poll()
    task = env["task"]
    original = worker.registration_id
    worker.registration_id = str(uuid.uuid4())
    assert (await worker.heartbeat())[0] == 409
    worker.registration_id = original
    assert (await worker.ack(task))[0] == 200
    assert (await worker.result(task, stdout="x" * 5000))[0] == 413
    service.revoke_test_worker()
    assert (await worker.heartbeat())[0] == 403
    audit = service.audit_text()
    assert secret not in audit and (worker.access_token or "") not in audit
    assert service.task_state(task_id) == "running"


def test_closed_echo_schema_and_no_production_dependencies(tmp_path):
    from gateway.worker_control_plane.models import validate_system_echo_payload
    from gateway.worker_control_plane import app, auth, service, storage

    assert validate_system_echo_payload({"message": "x"}) == {"message": "x"}
    for invalid in ({}, {"message": ""}, {"message": ["x"]}, {"message": "x", "command": "id"}):
        with pytest.raises(ValueError):
            validate_system_echo_payload(invalid)
    settings = WorkerControlPlaneSettings.for_test(
        tmp_path / "x.db", approved_test_root=tmp_path
    )
    with pytest.raises(ValueError):
        create_worker_control_plane_app(
            WorkerControlPlaneSettings(
                False, True, tmp_path / "x2.db", approved_test_root=tmp_path
            )
        )
    source = "\n".join(inspect.getsource(module) for module in (app, auth, service, storage))
    for forbidden in ("subprocess", "os.system", "kanban.db", "SessionDB", "state.db", "C:\\\\HermesServerWorker", "/v1/runs"):
        assert forbidden not in source


@pytest.mark.asyncio
async def test_closed_schema_ack_deadline_and_idempotency_conflicts(control_plane):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    body = worker.base() | {"capabilities": ["system.echo"], "max_tasks": 1, "wait_seconds": 0, "extra": True}
    response = await client.post("/worker/v1/tasks/poll", headers=worker.headers("closed"), json=body)
    assert response.status == 400
    task_id = service.create_test_echo_task({"message": "deadline"}, "deadline-create")
    _, envelope = await worker.poll("deadline-poll")
    task = envelope["task"]
    service.advance_for_test(11)
    assert (await worker.ack(task, key="deadline-ack"))[0] == 410
    assert service.task_state(task_id) == "queued"
    _, envelope = await worker.poll("redelivery-poll")
    task = envelope["task"]
    assert (await worker.ack(task, key="same-key"))[0] == 200
    conflicting = worker.base() | {"delivery_id": task["delivery_id"], "accepted": False, "reason": "temporary", "worker_time": "2026-01-01T00:00:00Z"}
    response = await client.post(f"/worker/v1/tasks/{task_id}/ack", headers=worker.headers("same-key"), json=conflicting)
    assert response.status == 409
    wrong_body = worker.base() | {"task_id": str(uuid.uuid4()), "delivery_id": task["delivery_id"], "task_type": "system.echo", "status": "completed", "stdout": "deadline", "stderr": "", "exit_code": 0, "started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:00Z", "duration_ms": 0, "result_idempotency_key": "wrong-task", "payload_hash": task["payload_hash"], "trace_id": task["trace_id"]}
    response = await client.post(f"/worker/v1/tasks/{task_id}/result", headers=worker.headers("wrong-task-request"), json=wrong_body)
    assert response.status == 422


def test_test_settings_reject_non_temporary_database_path():
    with pytest.raises(ValueError):
        WorkerControlPlaneSettings.for_test(
            __import__("pathlib").Path("/home/boonl/not-test-worker.db"),
            approved_test_root=__import__("pathlib").Path("/tmp"),
        )


def test_sqlite_path_is_confined_to_explicit_approved_root(tmp_path):
    approved = tmp_path / "approved"
    approved.mkdir()
    valid = approved / "worker-control-plane.db"
    settings = WorkerControlPlaneSettings.for_test(valid, approved_test_root=approved)
    assert settings.db_path == valid.resolve()
    assert settings.approved_test_root == approved.resolve()

    outside = tmp_path / "outside.db"
    with pytest.raises(ValueError):
        WorkerControlPlaneSettings.for_test(outside, approved_test_root=approved)
    assert not outside.exists()

    traversal = approved / "nested" / ".." / "escape.db"
    with pytest.raises(ValueError):
        WorkerControlPlaneSettings.for_test(traversal, approved_test_root=approved)
    assert not traversal.resolve().exists()


def test_sqlite_path_rejects_symlink_escape_and_production_like_names(tmp_path):
    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = approved / "link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    escaped = link / "worker-control-plane.db"
    with pytest.raises(ValueError):
        WorkerControlPlaneSettings.for_test(escaped, approved_test_root=approved)
    assert not (outside / "worker-control-plane.db").exists()

    production = approved / ".hermes" / "state.db"
    with pytest.raises(ValueError):
        WorkerControlPlaneSettings.for_test(production, approved_test_root=approved)
    assert not production.exists()
    assert not production.parent.exists()


def test_store_revalidates_path_after_delayed_symlink_escape(tmp_path):
    from gateway.worker_control_plane.storage import WorkerControlPlaneStore

    approved = tmp_path / "approved"
    approved.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    child = approved / "child"
    outside_db = outside / "worker-control-plane.db"
    settings = WorkerControlPlaneSettings.for_test(
        child / "worker-control-plane.db", approved_test_root=approved
    )

    try:
        os.symlink(outside, child, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError):
        WorkerControlPlaneStore(settings)
    assert not outside_db.exists()


def test_store_connect_time_revalidation_accepts_unchanged_path(tmp_path):
    from gateway.worker_control_plane.storage import WorkerControlPlaneStore

    approved = tmp_path / "approved"
    approved.mkdir()
    child = approved / "child"
    db_path = child / "worker-control-plane.db"
    settings = WorkerControlPlaneSettings.for_test(
        db_path, approved_test_root=approved
    )
    child.mkdir()

    store = WorkerControlPlaneStore(settings)
    try:
        assert db_path.exists()
    finally:
        store.close()


def test_store_and_service_require_test_mode(tmp_path):
    from gateway.worker_control_plane.storage import WorkerControlPlaneStore

    with pytest.raises(ValueError):
        settings = WorkerControlPlaneSettings(
            enabled=True,
            test_mode=False,
            db_path=tmp_path / "disabled.db",
            approved_test_root=tmp_path,
        )
        WorkerControlPlaneService(settings)
    assert not (tmp_path / "disabled.db").exists()

    with pytest.raises((TypeError, ValueError)):
        WorkerControlPlaneStore(tmp_path / "direct.db")
    assert not (tmp_path / "direct.db").exists()


@pytest.mark.asyncio
async def test_idempotency_key_binds_endpoint_and_empty_poll(control_plane):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201

    assert (await worker.poll("empty-replay"))[0] == 204
    service.create_test_echo_task({"message": "later"}, "empty-later")
    assert (await worker.poll("empty-replay"))[0] == 204

    _, envelope = await worker.poll("lease-later")
    task = envelope["task"]
    status, body = await worker.ack(task, key="empty-replay")
    assert status == 409
    assert body["error"]["code"] == "idempotency_conflict"


@pytest.mark.asyncio
async def test_idempotency_key_binds_body_registration_and_task(control_plane):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    assert (await worker.poll("identity-key"))[0] == 204

    changed_registration = worker.base() | {
        "registration_id": str(uuid.uuid4()),
        "capabilities": ["system.echo"],
        "max_tasks": 1,
        "wait_seconds": 0,
    }
    response = await client.post(
        "/worker/v1/tasks/poll",
        headers=worker.headers("identity-key"),
        json=changed_registration,
    )
    body = await response.json()
    assert response.status == 409
    assert body["error"]["code"] == "idempotency_conflict"

    first = service.create_test_echo_task({"message": "first"}, "task-key-first")
    _, first_envelope = await worker.poll("first-lease")
    assert first_envelope["task"]["task_id"] == first
    assert (await worker.ack(first_envelope["task"], accepted=False, reason="permanent", key="task-bound"))[0] == 200

    second = service.create_test_echo_task({"message": "second"}, "task-key-second")
    _, second_envelope = await worker.poll("second-lease")
    assert second_envelope["task"]["task_id"] == second
    status, body = await worker.ack(second_envelope["task"], accepted=False, reason="permanent", key="task-bound")
    assert status == 409
    assert body["error"]["code"] == "idempotency_conflict"


@pytest.mark.asyncio
async def test_result_http_idempotency_replays_and_conflicts_before_mutation(control_plane):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    service.create_test_echo_task({"message": "one"}, "result-one")
    _, envelope = await worker.poll("result-poll-one")
    task = envelope["task"]
    assert (await worker.ack(task, key="result-ack-one"))[0] == 200

    status, accepted = await worker.result(task, result_key="body-result-one", request_key="http-result-key")
    assert status == 200 and accepted["duplicate"] is False
    status, replay = await worker.result(task, result_key="body-result-one", request_key="http-result-key")
    assert status == 200 and replay == accepted

    changed = worker.base() | {
        "task_id": task["task_id"],
        "delivery_id": task["delivery_id"],
        "task_type": "system.echo",
        "status": "failed",
        "stdout": "changed",
        "stderr": "",
        "exit_code": 1,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:00Z",
        "duration_ms": 1,
        "result_idempotency_key": "body-result-changed",
        "payload_hash": task["payload_hash"],
        "trace_id": task["trace_id"],
    }
    response = await client.post(
        f'/worker/v1/tasks/{task["task_id"]}/result',
        headers=worker.headers("http-result-key"),
        json=changed,
    )
    body = await response.json()
    assert response.status == 409
    assert body["error"]["code"] == "idempotency_conflict"

    service.create_test_echo_task({"message": "two"}, "result-two")
    _, envelope = await worker.poll("result-poll-two")
    second = envelope["task"]
    assert (await worker.ack(second, key="result-ack-two"))[0] == 200
    status, body = await worker.result(second, result_key="body-result-two", request_key="http-result-key")
    assert status == 409
    assert body["error"]["code"] == "idempotency_conflict"
    assert service.task_state(second["task_id"]) == "running"


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "value"), (("max_tasks", True), ("wait_seconds", False)))
async def test_poll_rejects_boolean_integer_fields(control_plane, field, value):
    _, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    body = worker.base() | {"capabilities": ["system.echo"], "max_tasks": 1, "wait_seconds": 0}
    body[field] = value
    response = await client.post("/worker/v1/tasks/poll", headers=worker.headers(f"bool-{field}"), json=body)
    assert response.status == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("current_task_id", ("not-a-uuid", {"task_id": str(uuid.uuid4()), "extra": True}))
async def test_heartbeat_rejects_invalid_or_nested_current_task_id(control_plane, current_task_id):
    _, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    status, _ = await worker.heartbeat(current_task_id=current_task_id)
    assert status == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(("field", "value"), (("exit_code", True), ("duration_ms", False)))
async def test_result_rejects_boolean_integer_fields(control_plane, field, value):
    service, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    service.create_test_echo_task({"message": "types"}, f"types-{field}")
    _, envelope = await worker.poll(f"types-poll-{field}")
    task = envelope["task"]
    assert (await worker.ack(task, key=f"types-ack-{field}"))[0] == 200
    body = worker.base() | {
        "task_id": task["task_id"],
        "delivery_id": task["delivery_id"],
        "task_type": "system.echo",
        "status": "failed",
        "stdout": "",
        "stderr": "",
        "exit_code": 1,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:00Z",
        "duration_ms": 1,
        "result_idempotency_key": f"types-result-{field}",
        "payload_hash": task["payload_hash"],
        "trace_id": task["trace_id"],
    }
    body[field] = value
    response = await client.post(
        f'/worker/v1/tasks/{task["task_id"]}/result',
        headers=worker.headers(f"types-http-{field}"),
        json=body,
    )
    assert response.status == 422


@pytest.mark.asyncio
async def test_missing_null_and_unknown_fields_are_deterministic(control_plane):
    _, client, secret = control_plane
    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    base = worker.base() | {"status": "idle", "current_task_id": None, "worker_time": "2026-01-01T00:00:00Z"}
    missing = dict(base)
    missing.pop("status")
    extra = base | {"extra": {"nested": True}}
    wrong_null = base | {"status": None}
    for body in (missing, extra, wrong_null):
        response = await client.post("/worker/v1/heartbeat", headers=worker.headers(), json=body)
        assert response.status == 400


@pytest.mark.asyncio
async def test_temporary_nack_cannot_exceed_retry_limit(control_plane):
    service, _, secret = control_plane
    worker = MockWorkerClient(control_plane[1], secret)
    assert (await worker.register())[0] == 201
    task_id = service.create_test_echo_task({"message": "bounded"}, "bounded-retry")
    for attempt in range(1, service.settings.max_attempts + 1):
        status, envelope = await worker.poll(f"bounded-poll-{attempt}")
        assert status == 200
        status, body = await worker.ack(
            envelope["task"],
            accepted=False,
            reason="temporary",
            key=f"bounded-ack-{attempt}",
        )
        assert status == 200
        expected = "queued" if attempt < service.settings.max_attempts else "dead_letter"
        assert body["task_state"] == expected
        assert service.task_state(task_id) == expected
    assert (await worker.poll("bounded-after"))[0] == 204
    assert service.store.conn.execute(
        "SELECT max(attempt) FROM worker_deliveries WHERE task_id=?", (task_id,)
    ).fetchone()[0] == service.settings.max_attempts
    assert "task_dead_lettered" in service.audit_text()


def test_temporary_nack_uses_persisted_limit_after_config_increase(tmp_path):
    db_path = tmp_path / "persisted-increase.db"
    initial_settings = WorkerControlPlaneSettings(
        enabled=True,
        test_mode=True,
        db_path=db_path,
        approved_test_root=tmp_path,
        max_attempts=2,
    )
    service = WorkerControlPlaneService(initial_settings)
    try:
        identity, token = _register_direct_test_worker(service)
        task_id = service.create_test_echo_task(
            {"message": "persisted increase"}, "persisted-increase-task"
        )
        first = _direct_poll(service, identity, token, "persisted-increase-poll-1")["task"]
        assert _direct_temporary_nack(
            service, identity, token, first, "persisted-increase-ack-1"
        )["task_state"] == "queued"
        second = _direct_poll(service, identity, token, "persisted-increase-poll-2")["task"]
        assert second["attempt"] == 2
    finally:
        service.close()

    reopened_settings = WorkerControlPlaneSettings(
        enabled=True,
        test_mode=True,
        db_path=db_path,
        approved_test_root=tmp_path,
        max_attempts=5,
    )
    reopened = WorkerControlPlaneService(reopened_settings)
    try:
        response = _direct_temporary_nack(
            reopened, identity, token, second, "persisted-increase-ack-2"
        )
        assert response["task_state"] == "dead_letter"
        assert reopened.task_state(task_id) == "dead_letter"
        assert "task_dead_lettered" in reopened.audit_text()
        assert _direct_poll(
            reopened, identity, token, "persisted-increase-after"
        ) is None
    finally:
        reopened.close()


def test_temporary_nack_uses_persisted_limit_after_config_decrease(tmp_path):
    db_path = tmp_path / "persisted-decrease.db"
    initial_settings = WorkerControlPlaneSettings(
        enabled=True,
        test_mode=True,
        db_path=db_path,
        approved_test_root=tmp_path,
        max_attempts=5,
    )
    service = WorkerControlPlaneService(initial_settings)
    try:
        identity, token = _register_direct_test_worker(service)
        task_id = service.create_test_echo_task(
            {"message": "persisted decrease"}, "persisted-decrease-task"
        )
        first = _direct_poll(service, identity, token, "persisted-decrease-poll-1")["task"]
        assert _direct_temporary_nack(
            service, identity, token, first, "persisted-decrease-ack-1"
        )["task_state"] == "queued"
        second = _direct_poll(service, identity, token, "persisted-decrease-poll-2")["task"]
        assert second["attempt"] == 2
    finally:
        service.close()

    reopened_settings = WorkerControlPlaneSettings(
        enabled=True,
        test_mode=True,
        db_path=db_path,
        approved_test_root=tmp_path,
        max_attempts=2,
    )
    reopened = WorkerControlPlaneService(reopened_settings)
    try:
        response = _direct_temporary_nack(
            reopened, identity, token, second, "persisted-decrease-ack-2"
        )
        assert response["task_state"] == "queued"
        assert reopened.task_state(task_id) == "queued"
    finally:
        reopened.close()


def test_access_token_verifier_uses_constant_time_digest_comparison(monkeypatch):
    from gateway.worker_control_plane import auth

    calls = []
    original = auth.hmac.compare_digest

    def tracked(left, right):
        calls.append((left, right))
        return original(left, right)

    monkeypatch.setattr(auth.hmac, "compare_digest", tracked)
    digest = auth.token_hash("opaque-access-token")
    assert auth.verify_access_token("opaque-access-token", digest) is True
    assert auth.verify_access_token("wrong-access-token", digest) is False
    assert len(calls) == 2
    assert all(len(left) == len(right) == 64 for left, right in calls)


@pytest.mark.asyncio
async def test_rejection_audits_persist_without_sensitive_values(control_plane):
    service, client, secret = control_plane
    bad = MockWorkerClient(client, "wrong-bootstrap-secret")
    assert (await bad.register())[0] == 401

    worker = MockWorkerClient(client, secret)
    assert (await worker.register())[0] == 201
    original_registration = worker.registration_id
    worker.registration_id = str(uuid.uuid4())
    assert (await worker.heartbeat())[0] == 409
    worker.registration_id = original_registration

    assert (await worker.poll("audit-conflict"))[0] == 204
    changed = worker.base() | {
        "registration_id": str(uuid.uuid4()),
        "capabilities": ["system.echo"],
        "max_tasks": 1,
        "wait_seconds": 0,
    }
    response = await client.post(
        "/worker/v1/tasks/poll",
        headers=worker.headers("audit-conflict"),
        json=changed,
    )
    assert response.status == 409

    task_id = service.create_test_echo_task({"message": "audit"}, "audit-task")
    _, envelope = await worker.poll("audit-lease")
    task = envelope["task"]
    service.advance_for_test(service.settings.ack_deadline_seconds)
    assert (await worker.ack(task, key="audit-late-ack"))[0] == 410

    _, envelope = await worker.poll("audit-redelivery")
    task = envelope["task"]
    assert (await worker.ack(task, key="audit-ack"))[0] == 200
    wrong_hash = dict(task)
    wrong_hash["payload_hash"] = "0" * 64
    assert (await worker.result(wrong_hash, request_key="audit-result"))[0] == 422
    service.revoke_test_worker()

    audit = service.audit_text()
    required = {
        "worker_revoked",
        "credential_failed",
        "registration_rejected",
        "idempotency_conflict",
        "heartbeat_rejected",
        "poll_rejected",
        "ack_rejected",
        "result_rejected",
    }
    assert all(event in audit for event in required)
    credential_rows = service.store.conn.execute(
        "SELECT token_hash,salt FROM worker_credentials"
    ).fetchall()
    forbidden = [secret, worker.access_token, "wrong-bootstrap-secret"]
    forbidden.extend(value for row in credential_rows for value in row if value)
    assert all(value not in audit for value in forbidden)
    assert "audit" not in audit
    assert service.task_state(task_id) == "running"


@pytest.mark.asyncio
async def test_ack_is_rejected_exactly_at_deadline(control_plane):
    service, _, secret = control_plane
    worker = MockWorkerClient(control_plane[1], secret)
    assert (await worker.register())[0] == 201
    task_id = service.create_test_echo_task({"message": "boundary"}, "ack-boundary")
    _, envelope = await worker.poll("ack-boundary-poll")
    service.advance_for_test(service.settings.ack_deadline_seconds)
    status, _ = await worker.ack(envelope["task"], key="ack-boundary-key")
    assert status == 410
    assert service.task_state(task_id) == "queued"

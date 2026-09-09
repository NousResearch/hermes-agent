"""Focused HTTP tests for scoped RoomLink output artifacts."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway import hosted_rooms
from gateway.hosted_room_artifacts import (
    RoomArtifactOutbox,
    RoomArtifactScope,
    terminal_artifact_manifest,
)
from gateway.hosted_room_peer import issue_room_grant
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms import api_server_room_artifacts


@pytest.fixture
def artifact_api(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    scope = RoomArtifactScope.from_mapping({
        "room_id": "room-1",
        "task_id": "task-1",
        "execution_generation": 1,
        "member_id": "member-reviewer",
        "target_profile": "default",
        "home_install_id": "install-home",
        "target_install_id": "install-target",
        "authority_gateway_id": "gateway-home",
        "authority_epoch": 1,
    })
    source = tmp_path / "handoff.md"
    source.write_text("# Cross-gateway handoff\n", encoding="utf-8")
    outbox = RoomArtifactOutbox(home / "state.db")
    item = outbox.put_path(scope=scope, path=source)
    status = {
        "run_id": "run-1",
        "status": "completed",
        "room_artifact_scope": scope.as_mapping(),
        "artifacts": terminal_artifact_manifest(home / "state.db", scope),
    }
    adapter._request_owns_run = lambda _request, _run_id: True
    adapter._durable_run_status = lambda _request, _run_id: status
    app = web.Application()
    for method, path, handler in api_server_room_artifacts._http_routes(adapter):
        app.router.add_route(method, path, handler)
    app.router.add_post(
        "/v1/runs/{run_id}/stop",
        adapter._handle_stop_run,
    )
    grant = issue_room_grant(
        adapter._room_grant_secret(),
        grant_id="grant-artifact",
        room_id=scope.room_id,
        home_install_id=scope.home_install_id,
        authority_gateway_id=scope.authority_gateway_id,
        authority_epoch=scope.authority_epoch,
        member_id=scope.member_id,
        target_install_id=scope.target_install_id,
        target_profile=scope.target_profile,
        permissions=("artifact.ack", "artifact.read", "stop"),
    )
    hosted_rooms.reserve_peer_room(
        hosted_rooms.default_db_path(),
        claims={
            "room_id": scope.room_id,
            "member_id": scope.member_id,
            "target_profile": scope.target_profile,
            "authority_gateway_id": scope.authority_gateway_id,
            "authority_epoch": scope.authority_epoch,
        },
        expires_at=time.time() + 3600,
    )
    return adapter, app, scope, item, status, grant, outbox


@pytest.mark.asyncio
async def test_scoped_download_then_idempotent_ack(artifact_api):
    _adapter, app, scope, item, status, grant, outbox = artifact_api
    headers = {"Authorization": f"HermesRoom {grant}"}
    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            f"/v1/runs/run-1/artifacts/{item['artifact_id']}",
            headers=headers,
        )
        assert response.status == 200
        assert await response.read() == b"# Cross-gateway handoff\n"
        assert response.headers["X-Hermes-Artifact-SHA256"] == item["sha256"]

        body = {
            "artifact_ids": [item["artifact_id"]],
            "manifest_digest": status["artifacts"]["manifest_digest"],
            "message_event_id": "dmessage:task-1",
        }
        first = await client.post(
            "/v1/runs/run-1/artifacts/ack",
            headers=headers,
            json=body,
        )
        assert first.status == 200
        assert await first.json() == {"acknowledged": True, "changed": 1}
        replay = await client.post(
            "/v1/runs/run-1/artifacts/ack",
            headers=headers,
            json=body,
        )
        assert replay.status == 200
        assert (await replay.json())["changed"] == 0
    assert outbox.list(scope) == []


@pytest.mark.asyncio
async def test_cross_scope_and_tampered_ack_fail_closed(artifact_api):
    adapter, app, scope, item, status, _grant, _outbox = artifact_api
    wrong_grant = issue_room_grant(
        adapter._room_grant_secret(),
        grant_id="grant-wrong-member",
        room_id=scope.room_id,
        home_install_id=scope.home_install_id,
        authority_gateway_id=scope.authority_gateway_id,
        authority_epoch=scope.authority_epoch,
        member_id="member-other",
        target_install_id=scope.target_install_id,
        target_profile=scope.target_profile,
        permissions=("artifact.ack", "artifact.read"),
    )
    headers = {"Authorization": f"HermesRoom {wrong_grant}"}
    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            f"/v1/runs/run-1/artifacts/{item['artifact_id']}",
            headers=headers,
        )
        assert response.status == 404
        rejected = await client.post(
            "/v1/runs/run-1/artifacts/ack",
            headers={"Authorization": f"HermesRoom {artifact_api[5]}"},
            json={
                "artifact_ids": [item["artifact_id"]],
                "manifest_digest": "0" * 64,
                "message_event_id": "dmessage:task-1",
            },
        )
        assert rejected.status == 409


@pytest.mark.asyncio
async def test_ack_rejects_wrong_or_changed_message_commitment(artifact_api):
    _adapter, app, scope, item, status, grant, outbox = artifact_api
    headers = {"Authorization": f"HermesRoom {grant}"}
    base = {
        "artifact_ids": [item["artifact_id"]],
        "manifest_digest": status["artifacts"]["manifest_digest"],
    }
    async with TestClient(TestServer(app)) as client:
        wrong = await client.post(
            "/v1/runs/run-1/artifacts/ack",
            headers=headers,
            json={**base, "message_event_id": "dmessage:wrong"},
        )
        assert wrong.status == 409
        assert len(outbox.list(scope)) == 1
        accepted = await client.post(
            "/v1/runs/run-1/artifacts/ack",
            headers=headers,
            json={**base, "message_event_id": "dmessage:task-1"},
        )
        assert accepted.status == 200
        changed_replay = await client.post(
            "/v1/runs/run-1/artifacts/ack",
            headers=headers,
            json={**base, "message_event_id": "dmessage:another"},
        )
        assert changed_replay.status == 409


def test_adapter_registers_output_artifact_routes(artifact_api):
    adapter = artifact_api[0]
    routes = {(method, path) for method, path, _handler in adapter._http_route_table()}
    assert ("GET", "/v1/runs/{run_id}/artifacts/{artifact_id}") in routes
    assert ("POST", "/v1/runs/{run_id}/artifacts/ack") in routes
    assert ("POST", "/v1/runs/{run_id}/artifacts/discard") in routes


@pytest.mark.asyncio
async def test_permanent_verification_failure_discards_exact_run_bytes(
    artifact_api,
):
    _adapter, app, scope, _item, status, grant, outbox = artifact_api
    status["artifacts"] = {"invalid": "home must still be able to retire bytes"}
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/runs/run-1/artifacts/discard",
            headers={"Authorization": f"HermesRoom {grant}"},
            json={"reason": "verification_failed"},
        )
        assert response.status == 200
        assert await response.json() == {"discarded": True, "removed": 1}
    assert outbox.list(scope) == []


@pytest.mark.asyncio
async def test_artifact_discard_reports_transient_cleanup_failure_as_retryable(
    artifact_api,
    monkeypatch,
):
    _adapter, app, scope, _item, _status, grant, outbox = artifact_api

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("temporary unlink fault")

    monkeypatch.setattr(RoomArtifactOutbox, "discard_durably", fail_cleanup)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/runs/run-1/artifacts/discard",
            headers={"Authorization": f"HermesRoom {grant}"},
            json={"reason": "verification_failed"},
        )
        assert response.status == 503
        assert response.headers["Retry-After"] == "1"
        assert (await response.json())["error"]["code"] == (
            "artifact_retirement_unavailable"
        )
    assert len(outbox.list(scope)) == 1


@pytest.mark.asyncio
async def test_terminal_remote_stop_discards_crash_stranded_output(artifact_api):
    _adapter, app, scope, _item, status, grant, outbox = artifact_api
    status["status"] = "interrupted"
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/runs/run-1/stop",
            headers={"Authorization": f"HermesRoom {grant}"},
        )
        assert response.status == 200
    assert outbox.list(scope) == []

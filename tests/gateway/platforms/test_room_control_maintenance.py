"""Real reciprocal-control HTTP routes respect maintenance admission."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_rooms
from tests.gateway.platforms.test_api_server_room_controls import control_api


def paused():
    return web.json_response({"error": {"code": "gateway_draining"}}, status=503)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["send", "retry"])
async def test_control_cannot_start_work_while_draining(control_api, monkeypatch, action):
    adapter, app, service, headers = control_api
    monkeypatch.setattr(adapter, "_draining_response", paused)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/room-controls/room-1", headers=headers,
            json={"action": action, "command_id": "paused-command", "text": "hello"},
        )
        assert response.status == 503
    events = hosted_rooms.read_events(
        service.db_path, room_id="room-1", since_seq=0, limit=20
    )["events"]
    assert not any(event["kind"] == "message.user" for event in events)
    assert service.retried == []
    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_control_body_is_counted_and_rechecks_drain(control_api, monkeypatch):
    adapter, app, _service, headers = control_api
    entered, release = asyncio.Event(), asyncio.Event()
    original = adapter._read_json_body

    async def held_body(request):
        entered.set()
        await release.wait()
        return await original(request)

    monkeypatch.setattr(adapter, "_read_json_body", held_body)
    async with TestClient(TestServer(app)) as client:
        task = asyncio.create_task(client.post(
            "/v1/room-controls/room-1", headers=headers,
            json={"action": "send", "command_id": "held-body", "text": "hello"},
        ))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            assert adapter.active_agent_work_count() == 1
            monkeypatch.setattr(adapter, "_draining_response", paused)
            release.set()
            response = await task
            assert response.status == 503
            assert adapter.active_agent_work_count() == 0
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_and_auth_denial_remain_available(control_api, monkeypatch):
    adapter, app, _service, headers = control_api
    monkeypatch.setattr(adapter, "_draining_response", paused)
    async with TestClient(TestServer(app)) as client:
        denied = await client.post(
            "/v1/room-controls/room-1", json={"action": "send", "command_id": "denied"}
        )
        assert denied.status == 401
        stopped = await client.post(
            "/v1/room-controls/room-1", headers=headers,
            json={"action": "stop", "command_id": "stop-existing"},
        )
        assert stopped.status == 200
    assert adapter.active_agent_work_count() == 0


@pytest.mark.asyncio
async def test_control_rechecks_authorization_after_body_wait(control_api, monkeypatch):
    adapter, app, service, headers = control_api
    entered, release = asyncio.Event(), asyncio.Event()
    original = adapter._read_json_body

    async def held_body(request):
        entered.set()
        await release.wait()
        return await original(request)

    monkeypatch.setattr(adapter, "_read_json_body", held_body)
    async with TestClient(TestServer(app)) as client:
        task = asyncio.create_task(client.post(
            "/v1/room-controls/room-1", headers=headers,
            json={"action": "send", "command_id": "revoked-body", "text": "hello"},
        ))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            revoked = await client.delete("/v1/room-controls/room-1", headers=headers)
            assert revoked.status == 200
            release.set()
            response = await task
            assert response.status == 401
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    events = hosted_rooms.read_events(
        service.db_path, room_id="room-1", since_seq=0, limit=20
    )["events"]
    assert not any(event["kind"] == "message.user" for event in events)
    assert adapter.active_agent_work_count() == 0

"""Scoped reciprocal API for remote Group Chat control."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_controls, hosted_rooms
from gateway.config import PlatformConfig
from gateway.platforms import api_server_room_controls
from gateway.platforms.api_server import APIServerAdapter


HOME = "install:home"


class FakeService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.retried: list[str] = []

    def status(self, room_id: str):
        pending = [] if self.retried else [{"kind": "retry", "task_id": "task-1"}]
        return {
            "working": False,
            "blocked": bool(pending),
            "counts": {"deferred": len(pending)},
            "pending_actions": pending,
        }

    def send_server_owned(self, *, room_id, event_id, payload, actor, expected_authority=None):
        room = hosted_rooms.room_state(self.db_path, room_id=room_id)
        authority = expected_authority
        if authority is None:
            authority = (str(room["authority_gateway_id"]), int(room["authority_epoch"]))
        return hosted_rooms.append_event(
            self.db_path,
            room_id=room_id,
            event_id=event_id,
            kind="message.user",
            actor=actor,
            payload=payload,
            authority_gateway_id=authority[0],
            authority_epoch=authority[1],
        )

    def stop_room(self, room_id, *, cancel_id):
        room = hosted_rooms.room_state(self.db_path, room_id=room_id)
        hosted_rooms.request_room_stop(
            self.db_path,
            room_id=room_id,
            cancel_id=cancel_id,
            expected_gateway_id=str(room["authority_gateway_id"]),
            expected_epoch=int(room["authority_epoch"]),
        )
        return 1

    def retry_room_task(self, room_id, *, task_id):
        assert room_id == "room-1"
        self.retried.append(task_id)
        return {"status": "queued"}


@pytest.fixture
def control_api(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = home / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[
            {"member_id": "member-peer", "profile": "reviewer", "handle": "reviewer"},
            {"member_id": "local", "profile": "local", "handle": "local"},
        ],
        authority_gateway_id=HOME,
    )
    issued = hosted_room_controls.issue_home_control_token(
        db,
        room_id="room-1",
        member_id="member-peer",
        authority_gateway_id=HOME,
        authority_epoch=1,
        expires_at=10_000_000_000,
    )
    service = FakeService(db)
    monkeypatch.setattr(
        "tui_gateway.methods_groups.get_hosted_room_service",
        lambda: service,
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-secret"}))
    app = web.Application()
    for method, path, handler in api_server_room_controls._http_routes(adapter):
        app.router.add_route(method, path, handler)
    headers = {
        "Authorization": f"HermesRoomControl {issued.control_token}",
        "X-Hermes-Room-Member": "member-peer",
    }
    return adapter, app, service, headers


def test_api_server_registers_reciprocal_control_routes(control_api):
    adapter, _app, _service, _headers = control_api
    routes = {(method, path) for method, path, _handler in adapter._http_route_table()}
    assert ("GET", "/v1/room-controls/{room_id}") in routes
    assert ("POST", "/v1/room-controls/{room_id}") in routes
    assert ("DELETE", "/v1/room-controls/{room_id}") in routes


@pytest.mark.asyncio
@pytest.mark.parametrize("attached_service", [False, True])
@pytest.mark.parametrize("late_append", [False, True])
async def test_bound_send_rejects_authority_drift_but_typed_send_keeps_current_term(
    tmp_path, monkeypatch, attached_service, late_append,
):
    from gateway import hosted_room_messaging as rooms
    from tests.gateway.test_hosted_room_messaging import _TestHostedRoomService, _event, _runner

    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: HOME)
    service = _TestHostedRoomService(tmp_path / "fenced-send.db")
    service.create_room(room_id="room-1", name="Release room", members=[
        {"member_id": "default", "profile": "default", "handle": "hermes"},
        {"member_id": "ops", "profile": "ops", "handle": "ops"},
    ])
    backend = rooms.MessagingRoomBackend(db_path=service.db_path, service=service if attached_service else None)
    room = rooms.resolve_room(rooms.list_messaging_rooms(backend), "1")
    bound = _event("/group 1 send Bound draft", message_id="bound-send")

    def advance():
        hosted_rooms.claim_authority(
            service.db_path, room_id="room-1", expected_gateway_id=HOME,
            expected_epoch=1, new_gateway_id=HOME, event_id="advance-before-append",
        )

    with monkeypatch.context() as patch:
        if late_append:
            original = hosted_rooms.append_event

            def advance_at_append(*args, **kwargs):
                if kwargs.get("kind") == "message.user":
                    assert (kwargs["authority_gateway_id"], kwargs["authority_epoch"]) == (HOME, 1)
                    advance()
                return original(*args, **kwargs)

            patch.setattr(hosted_rooms, "append_event", advance_at_append)
        else:
            advance()
        with pytest.raises(hosted_rooms.AuthorityConflictError):
            rooms.send_to_room(backend, room, bound, "Bound draft", expected_authority=(HOME, 1))
    assert not [event for event in hosted_rooms.read_events(service.db_path, room_id="room-1")["events"]
                if event["kind"] == "message.user"]

    monkeypatch.setattr(rooms, "current_room_backend", lambda: backend)
    runner = _runner()
    typed = _event("/group 1 send Normal typed draft", message_id="typed-send")
    result = await runner._handle_room_command(typed)
    assert result.startswith("Queued in Release room")
    assert await runner._handle_room_command(typed) == result
    events = [event for event in hosted_rooms.read_events(service.db_path, room_id="room-1")["events"]
              if event["kind"] == "message.user"]
    assert len(events) == 1 and events[0]["authority_epoch"] == 2
    assert events[0]["payload"]["text"] == "Normal typed draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("late_append", [False, True])
async def test_remote_control_authority_reaches_real_service_append(control_api, monkeypatch, late_append):
    from types import ModuleType
    from tui_gateway.hosted_room_service import HostedRoomService

    _adapter, app, fixture_service, headers = control_api
    monkeypatch.setattr(hosted_rooms, "local_authority_gateway_id", lambda: HOME)
    service = HostedRoomService(ModuleType("test_core_control_service"), db_path=fixture_service.db_path)
    monkeypatch.setattr("tui_gateway.methods_groups.get_hosted_room_service", lambda: service)
    fences = []

    def advance():
        hosted_rooms.claim_authority(
            service.db_path, room_id="room-1", expected_gateway_id=HOME,
            expected_epoch=1, new_gateway_id=HOME, event_id="advance-after-token-authorization",
        )

    if late_append:
        original = hosted_rooms.append_event

        def advance_before_append(*args, **kwargs):
            if kwargs.get("kind") == "message.user":
                fences.append((kwargs["authority_gateway_id"], kwargs["authority_epoch"]))
                advance()
            return original(*args, **kwargs)

        monkeypatch.setattr(hosted_rooms, "append_event", advance_before_append)
    else:
        original = service.send_server_owned

        def advance_before_service_read(**kwargs):
            fences.append(kwargs.get("expected_authority"))
            advance()
            return original(**kwargs)

        monkeypatch.setattr(service, "send_server_owned", advance_before_service_read)
    async with TestClient(TestServer(app)) as client:
        denied = await client.post(
            "/v1/room-controls/room-1", headers=headers,
            json={"action": "send", "command_id": "late-control-send", "text": "Do not cross terms"},
        )
        assert denied.status == 400
        assert (await denied.json())["error"]["message"] == "stale hosted room authority"
        old_token = await client.post(
            "/v1/room-controls/room-1", headers=headers,
            json={"action": "send", "command_id": "old-token-retry", "text": "Still bound to the old term"},
        )
        assert old_token.status == 401
    assert fences == [(HOME, 1)]
    assert not [event for event in hosted_rooms.read_events(service.db_path, room_id="room-1")["events"]
                if event["kind"] == "message.user"]


@pytest.mark.asyncio
async def test_read_send_stop_and_retry_are_scoped_and_replay_safe(control_api):
    _adapter, app, service, headers = control_api
    async with TestClient(TestServer(app)) as client:
        denied = await client.get("/v1/room-controls/room-1")
        assert denied.status == 401

        initial = await client.get("/v1/room-controls/room-1", headers=headers)
        assert initial.status == 200
        initial_payload = await initial.json()
        assert initial_payload["room"]["name"] == "Release room"
        assert all(
            set(member) <= {"member_id", "handle", "display_name"}
            for member in initial_payload["room"]["members"]
        )

        send_body = {
            "action": "send",
            "command_id": "remote-send-1",
            "actor_display_name": "Signal",
            "text": "Review the release",
        }
        sent = await client.post(
            "/v1/room-controls/room-1",
            json=send_body,
            headers=headers,
        )
        replayed = await client.post(
            "/v1/room-controls/room-1",
            json=send_body,
            headers=headers,
        )
        assert sent.status == replayed.status == 200
        events = hosted_rooms.read_events(
            service.db_path,
            room_id="room-1",
            since_seq=0,
            limit=20,
        )["events"]
        user_events = [event for event in events if event["kind"] == "message.user"]
        assert len(user_events) == 1
        assert user_events[0]["actor"] == {
            "kind": "user",
            "id": "peer:member-peer",
            "display_name": "Signal",
        }

        retried = await client.post(
            "/v1/room-controls/room-1",
            json={"action": "retry", "command_id": "remote-retry-1"},
            headers=headers,
        )
        retry_replay = await client.post(
            "/v1/room-controls/room-1",
            json={"action": "retry", "command_id": "remote-retry-1"},
            headers=headers,
        )
        assert retried.status == retry_replay.status == 200
        assert service.retried == []
        assert (
            len(
                hosted_room_controls.load_pending_control_retries(
                    service.db_path,
                    room_id="room-1",
                )
            )
            == 1
        )

        stopped = await client.post(
            "/v1/room-controls/room-1",
            json={"action": "stop", "command_id": "remote-stop-1"},
            headers=headers,
        )
        assert stopped.status == 200
        stopped_events = hosted_rooms.read_events(
            service.db_path,
            room_id="room-1",
            since_seq=0,
            limit=20,
        )["events"]
        assert any(event["kind"] == "room.stop_requested" for event in stopped_events)

        revoked = await client.delete(
            "/v1/room-controls/room-1",
            headers=headers,
        )
        assert revoked.status == 200
        revoke_replay = await client.delete(
            "/v1/room-controls/room-1",
            headers=headers,
        )
        assert revoke_replay.status == 200
        denied_after_revoke = await client.get(
            "/v1/room-controls/room-1",
            headers=headers,
        )
        assert denied_after_revoke.status == 401


@pytest.mark.asyncio
async def test_control_token_is_member_and_room_scoped(control_api):
    _adapter, app, _service, headers = control_api
    async with TestClient(TestServer(app)) as client:
        wrong_member = await client.get(
            "/v1/room-controls/room-1",
            headers={**headers, "X-Hermes-Room-Member": "local"},
        )
        wrong_room = await client.get(
            "/v1/room-controls/other-room",
            headers=headers,
        )
        assert wrong_member.status == 401
        assert wrong_room.status == 401


@pytest.mark.asyncio
async def test_retry_is_queued_for_the_process_that_owns_the_room_lease(
    control_api,
):
    _adapter, app, service, headers = control_api
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/room-controls/room-1",
            json={"action": "retry", "command_id": "remote-retry-worker"},
            headers=headers,
        )
        assert response.status == 200
        payload = await response.json()
        assert payload["queued"] is True
        assert payload["retried"] == 1
    pending = hosted_room_controls.load_pending_control_retries(
        service.db_path,
        room_id="room-1",
    )
    assert [(item.command_id, item.task_ids) for item in pending] == [
        ("remote-retry-worker", ("task-1",))
    ]


@pytest.mark.asyncio
async def test_empty_remote_retry_returns_the_shared_actionable_error(control_api):
    _adapter, app, service, headers = control_api
    service.retried.append("already-settled")

    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/room-controls/room-1",
            json={"action": "retry", "command_id": "remote-retry-empty"},
            headers=headers,
        )
        payload = await response.json()

    assert response.status == 400
    assert payload["error"]["message"] == (
        "This Group Chat has no failed work to retry."
    )

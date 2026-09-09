"""Real loopback HTTP grants, transport and passive replica persistence."""

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_rooms as rooms
from gateway import hosted_room_replicas as replicas
from gateway.config import PlatformConfig
from gateway.platforms import api_server_room_grants as grants
from gateway.platforms import api_server_room_replicas as ingress
from gateway.platforms.api_server import APIServerAdapter
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


HOME = "install:home"
TARGET = "install:target"
KEY = "disposable-api-key-for-loopback-only"
MEMBERS = [{"member_id": "writer", "profile": "default", "handle": "writer", "target": {
    "kind": "local", "profile": "default",
}}, {"member_id": "reviewer", "profile": "default", "handle": "reviewer", "target": {
    "kind": "peer", "peer_id": "peer-reviewer", "installation_id": TARGET,
    "profile": "default", "capability_digest": "b" * 64,
}}]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    source, target = tmp_path / "source.db", tmp_path / "target.db"
    monkeypatch.setattr(rooms, "default_db_path", lambda: target)
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: TARGET)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))
    app = web.Application()
    for method, path, handler in grants._http_routes(adapter):
        app.router.add_route(method, path, handler)
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(
        source, room_id="room", event_id="hello", kind="message.user",
        actor={"kind": "user", "id": "owner"}, payload={"text": "Caf\u00e9 \u65e5\u672c\u8a9e"},
        authority_gateway_id=HOME, authority_epoch=1,
    )
    return source, target, app


async def invite(http, *, replication=None):
    body = {
        "room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME,
        "authority_epoch": 1, "member_id": "reviewer",
    }
    if replication is not None:
        body["replication"] = replication
    response = await http.post(
        "/v1/room-members/invitations", json=body, headers={"Authorization": f"Bearer {KEY}"},
    )
    assert response.status == 201, await response.text()
    return (await response.json())["grant"]


@pytest.mark.asyncio
async def test_real_http_invitation_transport_and_sqlite_replica(setup):
    source, target, app = setup
    async with TestClient(TestServer(app)) as http:
        token = await invite(http, replication=True)
        sender = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=2)
        result = await asyncio.to_thread(
            sender.replicate_page, grant=token, target_profile="default", room_id="room",
            room_name="Workshop", members=MEMBERS, page=rooms.read_events(source, room_id="room"),
        )
        assert result["stored_seq"] == 1
        assert result["authority"]["gateway_id"] == HOME
        assert replicas.replica_state(target, room_id="room")["safety_status"] == "passive"
        with pytest.raises(rooms.RoomNotFoundError):
            rooms.room_state(target, room_id="room")


@pytest.mark.asyncio
async def test_normal_invitation_cannot_be_used_for_replication(setup):
    source, target, app = setup
    async with TestClient(TestServer(app)) as http:
        token = await invite(http)
        sender = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=2)
        with pytest.raises(PeerRunsHTTPError) as error:
            await asyncio.to_thread(
                sender.replicate_page, grant=token, target_profile="default", room_id="room",
                room_name="Workshop", members=MEMBERS, page=rooms.read_events(source, room_id="room"),
            )
        assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_replica_endpoint_rejects_broad_bearer_auth(setup):
    _, _, app = setup
    async with TestClient(TestServer(app)) as http:
        response = await http.post(
            "/v1/room-members/replica", json={}, headers={"Authorization": f"Bearer {KEY}"},
        )
        assert response.status == 401


@pytest.mark.asyncio
async def test_replication_http_body_is_bounded(setup, monkeypatch):
    _, target, app = setup
    monkeypatch.setattr(ingress, "MAX_REPLICA_HTTP_BYTES", 256)
    async with TestClient(TestServer(app)) as http:
        token = await invite(http, replication=True)
        response = await http.post(
            "/v1/room-members/replica", json={"room_id": "x" * 1000},
            headers={"Authorization": f"HermesRoom {token}"},
        )
        assert response.status in {400, 413}
        with pytest.raises(replicas.ReplicaError, match="not found"):
            replicas.replica_state(target, room_id="room")


@pytest.mark.asyncio
async def test_near_limit_unicode_page_passes_through_real_replica_transport(setup):
    source, target, app = setup
    for i in range(8):
        rooms.append_event(
            source, room_id="room", event_id=f"large-{i}", kind="message.user",
            actor={"kind": "user", "id": "owner"}, payload={"text": "\u00e9" * 130000},
            authority_gateway_id=HOME, authority_epoch=1,
        )
    page = rooms.read_events(source, room_id="room")
    assert page["has_more"] is False
    body = {"room_id": "room", "room_name": "Workshop", "members": MEMBERS, "page": page}
    assert len(json.dumps(body, ensure_ascii=True).encode()) > ingress.MAX_REPLICA_HTTP_BYTES
    assert len(json.dumps(body, ensure_ascii=False).encode()) < ingress.MAX_REPLICA_HTTP_BYTES
    async with TestClient(TestServer(app)) as http:
        token = await invite(http, replication=True)
        sender = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=5)
        result = await asyncio.to_thread(sender.replicate_page, grant=token, target_profile="default", **body)
        assert result["stored_seq"] == page["latest_seq"]
        assert replicas.replica_state(target, room_id="room")["last_seq"] == page["latest_seq"]


@pytest.mark.asyncio
@pytest.mark.parametrize("base_suffix", ["", "/p/reviewer"])
async def test_replica_transport_addresses_named_profile_exactly_once(base_suffix):
    app = web.Application()

    async def accept(request):
        assert request.headers["Authorization"] == "HermesRoom disposable-scoped-token"
        return web.json_response({"path": request.path})

    app.router.add_post("/p/reviewer/v1/room-members/replica", accept)
    async with TestClient(TestServer(app)) as http:
        client = PeerRunsHTTPClient(base_url=str(http.make_url(base_suffix or "/")), api_key="")
        result = await asyncio.to_thread(
            client.replicate_page, grant="disposable-scoped-token", target_profile="reviewer",
            room_id="room", room_name="Workshop", members=[], page={},
        )
        assert result["path"] == "/p/reviewer/v1/room-members/replica"


def test_replica_transport_refuses_conflicting_saved_profile_before_network():
    client = PeerRunsHTTPClient(base_url="http://127.0.0.1:9/p/another", api_key="")
    with pytest.raises(ValueError, match="profile"):
        client.replicate_page(
            grant="disposable-scoped-token", target_profile="reviewer",
            room_id="room", room_name="Workshop", members=[], page={},
        )

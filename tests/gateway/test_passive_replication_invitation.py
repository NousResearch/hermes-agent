"""Restricted owner invitations reach the real HTTP/RPC permission consumers."""

import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_runs import _http_routes
from tests.gateway.test_api_server_room_replicas import (
    HOME, TARGET, KEY, MEMBERS, setup,  # noqa: F401
)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["http", "rpc"])
async def test_passive_invitation_can_copy_but_cannot_execute(setup, surface):
    source, target, app = setup
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))
    for method, path, handler in _http_routes(adapter):
        app.router.add_route(method, path, handler)
    body = dict(room_id="room", home_install_id=HOME, authority_gateway_id=HOME,
                authority_epoch=1, member_id="reviewer", replication=True, passive_only=True)
    async with TestClient(TestServer(app)) as http:
        if surface == "http":
            response = await http.post("/v1/room-members/invitations", json=body,
                                       headers={"Authorization": f"Bearer {KEY}"})
            assert response.status == 201, await response.text()
            result = await response.json()
        else:
            import tui_gateway.server as srv
            result = srv._methods["groups.peer.invite"](1, body)
            assert "error" not in result, result
            result = result["result"]
            capability = srv._methods["groups.capabilities"](2, {})["result"]
            assert capability["room_link"]["passive_replication"] == result["passive_replication"]
        claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), result["grant"], permission="status")
        assert set(claims["permissions"]) == {"status", "replicate"}
        versions = result["passive_replication"]
        assert 2 in versions["history_versions"] and 2 in versions["retirement_versions"]
        assert versions["work_record_versions"] == []
        assert peer.GatewayRoomCatalog.from_mapping(result["catalog"]).catalog_digest == result["catalog"]["catalog_digest"]
        sender = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=2)
        probe = await asyncio.to_thread(sender.probe, grant=result["grant"])
        assert probe["passive_replication"] == versions
        await asyncio.to_thread(sender.replicate_page, grant=result["grant"], target_profile="default",
            room_id="room", room_name="Workshop", members=MEMBERS, page=rooms.read_events(source, room_id="room"))
        assert replicas.replica_state(target, room_id="room")["last_seq"] == 1
        for path, payload in [("/v1/runs", {"input": "must not run"}),
                              ("/v1/runs/retained-run/approval", {"decision": "allow"}),
                              ("/v1/runs/retained-run/stop", {}),
                              ("/v1/room-members/grants/refresh", {})]:
            denied = await http.post(path, json=payload, headers={"Authorization": f"HermesRoom {result['grant']}"})
            assert denied.status == 401, (path, await denied.text())
            assert (await denied.json())["error"]["code"] == "invalid_room_grant"
        assert not adapter._active_run_tasks and not adapter._run_statuses


@pytest.mark.asyncio
@pytest.mark.parametrize("passive,replication", [(True, False), ("true", True), (1, True), (None, True)])
async def test_passive_mode_requires_explicit_booleans(setup, passive, replication):
    _, _, app = setup
    async with TestClient(TestServer(app)) as http:
        response = await http.post("/v1/room-members/invitations", json={
            "room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME,
            "authority_epoch": 1, "member_id": "reviewer", "replication": replication, "passive_only": passive,
        }, headers={"Authorization": f"Bearer {KEY}"})
        assert response.status == 400

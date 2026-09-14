"""Registered disband RPC closes a real peer route and drains copy retirement."""

import asyncio
import copy
from contextvars import ContextVar

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_driver as driver
from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from gateway.config import PlatformConfig
from gateway.platforms import api_server, api_server_room_grants
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.gateway.test_api_server_room_replicas import HOME, TARGET, KEY, MEMBERS, invite
from tests.gateway.test_api_replica_retirement import HOME_SECRET, home_row, retired_row
from tests.tui_gateway.test_hosted_room_service import _server
from tui_gateway import hosted_room_replication as publisher
from tui_gateway import server as rpc
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.mark.asyncio
async def test_disband_rpc_freezes_before_stop_and_retires_through_live_worker(tmp_path, monkeypatch):
    source_home, target_home = tmp_path / "home", tmp_path / "target"
    source_home.mkdir()
    target_home.mkdir()
    source, target = source_home / "state.db", target_home / "state.db"
    identity = ContextVar("retirement_rpc_identity", default=HOME)
    database = ContextVar("retirement_rpc_database", default=source)
    monkeypatch.setenv("HERMES_HOME", str(source_home))
    monkeypatch.setattr(rooms, "local_authority_gateway_id", identity.get)
    monkeypatch.setattr(rooms, "default_db_path", database.get)
    monkeypatch.setattr(publisher, "gateway_room_grant_secret", lambda: HOME_SECRET)
    observed, secret_reads, model_calls = [], [], []

    def forbidden_secret():
        secret_reads.append(True)
        return HOME_SECRET

    @web.middleware
    async def target_context(request, handler):
        identity_token, db_token = identity.set(TARGET), database.set(target)
        home_token = set_hermes_home_override(target_home)
        profile_token = api_server._api_request_profile.set(request.match_info.get("profile") or "default")
        try:
            if request.path.endswith("/grants/revoke"):
                row = home_row(source)
                blocked = False
                try:
                    retirement.materialize_notice(
                        source, enrollment_id=row["enrollment_id"], local_gateway_id=HOME,
                        secret_loader=forbidden_secret,
                    )
                except retirement.RetirementConflictError:
                    blocked = True
                observed.append(("before_revoke", row["state"], row["frozen_at"] is not None, blocked))
            response = await handler(request)
            if request.path == "/v1/group-replicas/retire":
                observed.append(("retire_http", response.status))
            return response
        finally:
            api_server._api_request_profile.reset(profile_token)
            reset_hermes_home_override(home_token)
            database.reset(db_token)
            identity.reset(identity_token)

    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))
    app = web.Application(middlewares=[target_context])
    for method, path, handler in api_server_room_grants._http_routes(adapter):
        app.router.add_route(method, path, handler)
        if path.startswith("/v1/room-members/"):
            app.router.add_route(method, "/p/{profile}" + path, handler)
    service = HostedRoomService(_server(), db_path=source)
    monkeypatch.setattr(rpc, "get_hosted_room_service", lambda: service)

    def no_submit(**kwargs):
        model_calls.append(True)
        raise AssertionError("seeded room must not invoke a model")

    monkeypatch.setattr(service.rpc, "submit", no_submit)
    original_stop = service.stop_room

    def observed_stop(*args, **kwargs):
        row = home_row(source)
        observed.append(("before_stop", row["state"], row["frozen_at"] is not None))
        return original_stop(*args, **kwargs)

    monkeypatch.setattr(service, "stop_room", observed_stop)

    async def wait_for(predicate):
        deadline = asyncio.get_running_loop().time() + 12
        while asyncio.get_running_loop().time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.05)
        raise AssertionError("owned retirement workflow did not settle")

    try:
        async with TestClient(TestServer(app)) as http:
            token = await invite(http, replication=True)
            client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=2)
            proof = await asyncio.to_thread(client.probe, grant=token)
            catalog = peer.GatewayRoomCatalog.from_mapping(proof["catalog"])
            members = copy.deepcopy(MEMBERS)
            members[1]["target"]["capability_digest"] = catalog.catalog_digest
            service.create_room(room_id="room", name="Original name", members=members)
            rooms.rename_room(source, room_id="room", event_id="rename-only", name="Renamed room")
            route = PeerMemberRoute(
                home_install_id=HOME, member_id="reviewer", target_install_id=TARGET, target_profile="default",
                capability_digest=catalog.catalog_digest, execution_policy_digest=catalog.execution_policy.policy_digest,
                cancellation_scope_id="rpc-retirement", trace_id="rpc-retirement", grant=token,
            )
            service.register_peer_route(
                room_id="room", member_id="reviewer", route=route, client=client,
                target_url=str(http.make_url("/")), catalog=catalog,
            )
            enrollment = retirement.prepare_home_enrollment(
                source, room_id="room", target_install_id=TARGET, endpoint=str(http.make_url("/")),
                local_gateway_id=HOME, secret=HOME_SECRET, enrollment_id="rpc-enrollment",
            )
            response = await http.post(
                "/v1/group-replicas/enroll", json={"enrollment": enrollment},
                headers={"Authorization": "Bearer " + KEY},
            )
            assert response.status == 200
            worker = service.replication
            service.start()
            await wait_for(lambda: home_row(source)["state"] == "enrolled")
            await wait_for(lambda: worker.status("room")["routes"][0]["acked_seq"] == 1)
            result = await asyncio.to_thread(rpc._methods["groups.disband"], 701, {"room_id": "room"})
            assert "error" not in result, result.get("error", {}).get("message")
            assert result["result"]["tombstone"]["disbanded_at"] is not None
            assert not links.load_room_links_tolerant(source)[0]
            with pytest.raises(PeerRunsHTTPError):
                await asyncio.to_thread(client.probe, grant=token)
            await wait_for(lambda: home_row(source)["state"] == "acknowledged")
            assert service.replication is worker
            assert retired_row(target)["enrollment_id"] == enrollment["enrollment_id"]
            assert replicas.replica_state(target, room_id="room")["disbanded_at"] is None
            with rooms._transaction(target) as conn:
                kinds = [row[0] for row in conn.execute(
                    "SELECT kind FROM hosted_room_replica_events WHERE room_id='room'")]
            assert "room.disbanded" not in kinds
            assert driver.list_tasks(source, room_id="room") == []
            assert model_calls == [] and secret_reads == []
            assert ("before_revoke", "closing", True, True) in observed
            assert ("retire_http", 200) in observed
            assert ("before_stop", "closing", True) in observed
    finally:
        assert await asyncio.to_thread(service.stop, timeout=5)

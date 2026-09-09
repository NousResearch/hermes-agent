"""Real HTTP retirement boundaries with distinct home and target secrets."""

import asyncio
import json
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_link_records as records
from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_rooms as rooms
from tests.gateway.test_api_server_room_replicas import HOME, TARGET, KEY, invite, setup  # noqa: F401
from tests.tui_gateway.test_hosted_room_replication_http import make_publisher
from tui_gateway import hosted_room_replication as publisher
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

HOME_SECRET = b"separate-home-retirement-test-secret-only"
OWNER = {"Authorization": "Bearer " + KEY}
BASE = "/v1/group-replicas/"


@pytest.fixture
def http_setup(setup):
    source, target, app = setup
    for route in list(app.router.routes()):
        if route.resource.canonical.startswith("/v1/room-members/"):
            app.router.add_route(route.method, "/p/{profile}" + route.resource.canonical, route.handler)
    return source, target, app


def home_row(source):
    return retirement.home_status(source, room_id="room")[0]


def retired_row(target):
    with rooms._transaction(target) as conn:
        row = conn.execute(
            f"SELECT enrollment_id,retired_at,stored_seq FROM {retirement.RETIREMENT_TABLE} WHERE room_id='room'").fetchone()
        return dict(row) if row else None


def prepare(source, http):
    return retirement.prepare_home_enrollment(
        source, room_id="room", target_install_id=TARGET, endpoint=str(http.make_url("/")),
        local_gateway_id=HOME, secret=HOME_SECRET, enrollment_id="http-enrollment")


async def connected(source, http, monkeypatch):
    token = await invite(http, replication=True)
    client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=2)
    proof = await asyncio.to_thread(client.probe, grant=token)
    links.save_room_link(source, links.make_stored_link(
        room_id="room", member_id="reviewer", target_url=str(http.make_url("/")),
        target_profile="default", grant=token, catalog=peer.GatewayRoomCatalog.from_mapping(proof["catalog"]),
        cancellation_scope_id="retirement-test", trace_id="retirement-test"))
    monkeypatch.setattr(publisher, "gateway_room_grant_secret", lambda: HOME_SECRET)
    return token, client, make_publisher(source, monkeypatch)


async def enroll(http, enrollment):
    response = await http.post(BASE + "enroll", json={"enrollment": enrollment}, headers=OWNER)
    assert response.status == 200
    return await response.json()


async def close_source(source, client, token):
    scope = dict(room_id="room", authority_gateway_id=HOME, authority_epoch=1)
    records.begin_room_link_retirement(source, **scope)
    await asyncio.to_thread(client.revoke_grant, grant=token)
    records.complete_room_link_retirement(source, **scope)
    records.delete_room_link_records(source, room_id="room")
    rooms.disband_room(source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
    assert not links.load_room_links_tolerant(source)[0]


async def drain(pub):
    pub._scan(time.monotonic())
    work = [item for item in pub._routes if isinstance(item, publisher._RetirementWork)]
    assert len(work) == 1
    await asyncio.to_thread(pub._publish_retirement, work[0])


@pytest.mark.asyncio
async def test_owner_bearer_required_and_ordinary_grant_cannot_retire(http_setup, monkeypatch):
    source, target, app = http_setup
    async with TestClient(TestServer(app)) as http:
        token, client, _ = await connected(source, http, monkeypatch)
        enrollment = prepare(source, http)
        for headers in ({}, {"Authorization": "HermesRoom " + token}):
            response = await http.post(BASE + "enroll", json={"enrollment": enrollment}, headers=headers)
            assert response.status in {401, 403}
        assert (await enroll(http, enrollment))["state"] == "active"
        body = {"room_id": "room", "enrollment_id": enrollment["enrollment_id"]}
        for headers in ({}, {"Authorization": "HermesRoom " + token}):
            response = await http.post(BASE + "revoke-enrollment", json=body, headers=headers)
            assert response.status in {401, 403}
        response = await http.post(BASE + "retire", json=body, headers={"Authorization": "HermesRoom " + token})
        assert response.status == 403
        response = await http.post(BASE + "revoke-enrollment", json=body, headers=OWNER)
        assert response.status == 200
        await close_source(source, client, token)
        notice = retirement.materialize_notice(
            source, enrollment_id=enrollment["enrollment_id"], local_gateway_id=HOME, secret_loader=lambda: HOME_SECRET)
        with pytest.raises(PeerRunsHTTPError) as denied:
            await asyncio.to_thread(client.retire_replica, notice)
        assert denied.value.status_code == 403
        assert retired_row(target) is None


@pytest.mark.asyncio
async def test_probe_confirmation_precedes_copy_and_fence_cannot_reveal(http_setup, monkeypatch):
    source, target, app = http_setup
    async with TestClient(TestServer(app)) as http:
        _, _, pub = await connected(source, http, monkeypatch)
        enrollment = prepare(source, http)
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        assert home_row(source)["state"] == "prepared"
        with pytest.raises(replicas.ReplicaError):
            replicas.replica_state(target, room_id="room")
        await enroll(http, enrollment)
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        assert home_row(source)["state"] == "enrolled"
        assert replicas.replica_state(target, room_id="room")["last_seq"] == 1
        records.begin_room_link_retirement(source, room_id="room", authority_gateway_id=HOME, authority_epoch=1)
        calls = []
        with pytest.raises(retirement.RetirementConflictError):
            retirement.materialize_notice(
                source, enrollment_id=enrollment["enrollment_id"], local_gateway_id=HOME,
                secret_loader=lambda: calls.append(True),
            )
        assert calls == []
        assert retirement.pending_notice_ids(source, local_gateway_id=HOME) == []
        with pytest.raises(retirement.RetirementConflictError):
            prepare(source, http)


@pytest.mark.asyncio
async def test_closed_route_retires_after_lost_ack_and_publisher_restart(http_setup, monkeypatch):
    source, target, app = http_setup
    lose = True
    @web.middleware
    async def lose_retire_ack(request, handler):
        nonlocal lose
        response = await handler(request)
        if request.path == BASE + "retire" and response.status == 200 and lose:
            lose = False
            request.transport.close()
        return response
    app.middlewares.append(lose_retire_ack)
    async with TestClient(TestServer(app)) as http:
        token, client, pub = await connected(source, http, monkeypatch)
        enrollment = prepare(source, http)
        await enroll(http, enrollment)
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        await close_source(source, client, token)
        with pytest.raises(PeerRunsHTTPError):
            await asyncio.to_thread(client.probe, grant=token)
        await drain(pub)
        original = retired_row(target)
        assert original["stored_seq"] == 1
        assert home_row(source)["state"] == "ready"
        restarted = make_publisher(source, monkeypatch)
        await drain(restarted)
        assert retired_row(target) == original
        assert home_row(source)["state"] == "acknowledged"
        assert replicas.replica_state(target, room_id="room")["disbanded_at"] is None
        response = await http.post(BASE + "enroll", json={"enrollment": enrollment}, headers=OWNER)
        assert response.status == 409


@pytest.mark.asyncio
async def test_late_target_enrollment_closes_zero_history_without_new_home_setup(http_setup, monkeypatch):
    source, target, app = http_setup
    async with TestClient(TestServer(app)) as http:
        token, client, pub = await connected(source, http, monkeypatch)
        enrollment = prepare(source, http)
        await close_source(source, client, token)
        await drain(pub)
        assert home_row(source)["state"] == "ready"
        with pytest.raises(retirement.RetirementConflictError):
            prepare(source, http)
        await enroll(http, enrollment)
        await drain(make_publisher(source, monkeypatch))
        assert retired_row(target)["stored_seq"] == 0
        assert home_row(source)["state"] == "acknowledged"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"room_id": "other"}, {"target_install_id": "install:other"}, {"authority_gateway_id": "install:other"},
    {"enrollment_id": "other"}, {"authority_epoch": 2}, {"extra": True},
])
async def test_closing_capability_rejects_cross_scope_and_malformed_payload(http_setup, monkeypatch, change):
    source, target, app = http_setup
    async with TestClient(TestServer(app)) as http:
        token, client, _ = await connected(source, http, monkeypatch)
        enrollment = prepare(source, http)
        await enroll(http, enrollment)
        await close_source(source, client, token)
        notice = retirement.materialize_notice(
            source, enrollment_id=enrollment["enrollment_id"], local_gateway_id=HOME, secret_loader=lambda: HOME_SECRET)
        response = await http.post(BASE + "retire", json={**notice.payload(), **change},
                                   headers={"Authorization": "HermesReplicaRetirement " + notice.value})
        assert response.status in {400, 403, 409}
        clean = notice.value not in await response.text()
        assert clean, "closing material leaked"
        assert retired_row(target) is None


@pytest.mark.asyncio
async def test_retirement_capability_is_not_general_auth_and_endpoint_is_installation_only(http_setup, monkeypatch):
    source, _, app = http_setup
    for route in list(app.router.routes()):
        if route.method == "POST" and route.resource.canonical.startswith(BASE):
            app.router.add_post("/p/{profile}" + route.resource.canonical, route.handler)
    async with TestClient(TestServer(app)) as http:
        token, client, _ = await connected(source, http, monkeypatch)
        enrollment = prepare(source, http)
        await enroll(http, enrollment)
        await close_source(source, client, token)
        notice = retirement.materialize_notice(
            source, enrollment_id=enrollment["enrollment_id"], local_gateway_id=HOME, secret_loader=lambda: HOME_SECRET)
        headers = {"Authorization": "HermesReplicaRetirement " + notice.value}
        for path in (BASE + "enroll", BASE + "revoke-enrollment", "/v1/room-members/invitations", "/v1/room-members/replica"):
            response = await http.post(path, json={"enrollment": enrollment}, headers=headers)
            assert response.status in {401, 403}
        response = await http.get("/v1/room-members/capabilities", headers=headers)
        assert response.status in {401, 403}
        for operation in ("enroll", "revoke-enrollment", "retire"):
            response = await http.post(
                "/p/default" + BASE + operation, json={"enrollment": enrollment},
                headers=headers if operation == "retire" else OWNER,
            )
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "installation_endpoint_required"
        scoped = PeerRunsHTTPClient(base_url=str(http.make_url("/p/default")), api_key="")
        with pytest.raises(PeerRunsHTTPError, match="installation"):
            await asyncio.to_thread(scoped.retire_replica, notice)
        response = await http.post(BASE + "retire", json=notice.payload(),
                                   headers={"Authorization": "HermesReplicaRetirement " + "A" * 43})
        assert response.status == 403
        result = await asyncio.to_thread(client.retire_replica, notice)
        assert result["retired"] is True
        clean = notice.value not in json.dumps(result)
        assert clean, "closing material leaked"

"""The participant's real HTTP routes, the real client, and publisher threads and processes."""

import asyncio
import json
import multiprocessing
from contextlib import suppress

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from gateway.hosted_room_passive_protocol import passive_capabilities
from gateway.hosted_room_replication import HostedRoomReplicationPublisher
from gateway.platforms import api_server_room_replicas as ingress
from tests.gateway.fixtures.passive_copy import API_KEY, HOME, MEMBERS, api, append, invite  # noqa: F401
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


def sender(http, timeout=3):
    return PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=timeout)


async def replicate(http, token, source, **body):
    return await asyncio.to_thread(
        sender(http, 5).replicate_page, grant=token, room_id="room", room_name="Workshop", members=MEMBERS,
        page=body.pop("page", None) or rooms.read_events(source, room_id="room"), **body)


async def register(http, api, token):
    """Store the member route on the home, as ``groups.peer.register`` does after its probe."""
    probe = await asyncio.to_thread(sender(http).probe, grant=token)
    links.save_room_link(api.source, links.make_stored_link(
        room_id="room", member_id="reviewer", target_url=str(http.make_url("/")), target_profile="default",
        grant=token, catalog=peer.GatewayRoomCatalog.from_mapping(probe["catalog"]),
        cancellation_scope_id="test-cancel", trace_id="test-trace"))
    return probe


def home_publisher(source, monkeypatch):
    # One process plays both installations: the publisher keeps the home identity it was built with.
    with monkeypatch.context() as scoped:
        scoped.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
        return HostedRoomReplicationPublisher(source)


async def wait_until(predicate, *, seconds=12):
    deadline = asyncio.get_running_loop().time() + seconds
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the copy did not reach the expected durable state")


@pytest.mark.asyncio
async def test_an_opted_in_invitation_copies_history_into_the_participant_store(api):
    async with TestClient(TestServer(api.app)) as http:
        invitation = await invite(http, replication=True)
        assert invitation["passive_replication"] == passive_capabilities()
        probe = await asyncio.to_thread(sender(http).probe, grant=invitation["grant"])
        assert probe["passive_replication"] == invitation["passive_replication"]
        result = await replicate(http, invitation["grant"], api.source)
        assert (result["object"], result["stored_seq"], result["authority"]["gateway_id"]) == (
            "hermes.room_member.replica", 1, HOME)
        assert replicas.replica_state(api.target, room_id="room")["safety_status"] == "passive"
        with pytest.raises(rooms.RoomNotFoundError):
            rooms.room_state(api.target, room_id="room")


@pytest.mark.asyncio
async def test_an_ordinary_invitation_cannot_be_used_to_copy(api):
    async with TestClient(TestServer(api.app)) as http:
        invitation = await invite(http)
        with pytest.raises(PeerRunsHTTPError) as error:
            await replicate(http, invitation["grant"], api.source)
        assert error.value.status_code == 401
        with pytest.raises(replicas.ReplicaNotFoundError):
            replicas.replica_state(api.target, room_id="room")


@pytest.mark.asyncio
async def test_the_copy_route_refuses_broad_api_key_auth(api):
    async with TestClient(TestServer(api.app)) as http:
        response = await http.post("/v1/room-members/replica", json={},
                                   headers={"Authorization": f"Bearer {API_KEY}"})
        assert response.status == 401


@pytest.mark.asyncio
async def test_the_copy_request_body_is_bounded(api, monkeypatch):
    monkeypatch.setattr(ingress, "MAX_REPLICA_HTTP_BYTES", 256)
    async with TestClient(TestServer(api.app)) as http:
        invitation = await invite(http, replication=True)
        response = await http.post("/v1/room-members/replica", json={"room_id": "x" * 1000},
                                   headers={"Authorization": f"HermesRoom {invitation['grant']}"})
        assert response.status in {400, 413}
        with pytest.raises(replicas.ReplicaNotFoundError):
            replicas.replica_state(api.target, room_id="room")


@pytest.mark.asyncio
async def test_a_near_limit_unicode_page_fits_because_it_travels_as_utf8(api):
    for i in range(8):
        append(api.source, f"large-{i}", "é" * 130000)
    page = rooms.read_events(api.source, room_id="room")
    assert page["has_more"] is False
    body = {"room_id": "room", "room_name": "Workshop", "members": MEMBERS, "page": page}
    assert len(json.dumps(body, ensure_ascii=True).encode()) > ingress.MAX_REPLICA_HTTP_BYTES
    assert len(json.dumps(body, ensure_ascii=False).encode()) < ingress.MAX_REPLICA_HTTP_BYTES
    async with TestClient(TestServer(api.app)) as http:
        invitation = await invite(http, replication=True)
        result = await replicate(http, invitation["grant"], api.source, page=page)
        assert result["stored_seq"] == page["latest_seq"]
        assert replicas.replica_state(api.target, room_id="room")["last_seq"] == page["latest_seq"]


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [
    {"passive_only": True}, {"replication": "true"}, {"replication": 1, "passive_only": True},
    {"replication": True, "passive_only": None}])
async def test_opt_in_flags_must_be_explicit_booleans(api, flags):
    async with TestClient(TestServer(api.app)) as http:
        response = await http.post("/v1/room-members/invitations", json={
            "room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME,
            "authority_epoch": 1, "member_id": "reviewer", **flags,
        }, headers={"Authorization": f"Bearer {API_KEY}"})
        assert response.status == 400


@pytest.mark.asyncio
async def test_a_lost_http_reply_and_a_publisher_restart_keep_one_exact_copy(api, monkeypatch):
    accepted, lose_reply = asyncio.Event(), [True]

    @web.middleware
    async def lose_after_persistence(request, handler):
        response = await handler(request)
        if request.path.endswith("/replica") and response.status == 200 and lose_reply[0]:
            lose_reply[0] = False
            accepted.set()
            request.transport.close()
        return response

    api.app.middlewares.append(lose_after_persistence)
    publishers = []
    async with TestClient(TestServer(api.app)) as http:
        await register(http, api, (await invite(http, replication=True))["grant"])
        try:
            first = home_publisher(api.source, monkeypatch)
            publishers.append(first)
            first.start()
            await asyncio.wait_for(accepted.wait(), timeout=10)
            assert await asyncio.to_thread(first.stop, timeout=5)
            assert replicas.replica_state(api.target, room_id="room")["last_seq"] == 1
            append(api.source, "follow-up", "Follow up after restart")
            rooms.rename_room(api.source, room_id="room", event_id="rename", name="Updated workshop")
            second = home_publisher(api.source, monkeypatch)
            publishers.append(second)
            second.start()
            await wait_until(lambda: any(s["acked_seq"] == 3 and s["status"] == "acked"
                                         for s in second.status("room")["routes"]))
            state = replicas.replica_state(api.target, room_id="room")
            assert (state["last_seq"], state["name"], state["safety_status"]) == (3, "Updated workshop", "passive")
            with rooms._transaction(api.target) as conn:
                copied = conn.execute("SELECT event_id FROM hosted_room_replica_events ORDER BY seq").fetchall()
            assert [r[0] for r in copied] == ["hello", "follow-up", "rename"]
        finally:
            for pub in publishers:
                assert await asyncio.to_thread(pub.stop, timeout=5)


@pytest.mark.asyncio
async def test_a_full_participant_recovers_without_a_new_grant(api, monkeypatch):
    async with TestClient(TestServer(api.app)) as http:
        await register(http, api, (await invite(http, replication=True))["grant"])
        pub = home_publisher(api.source, monkeypatch)
        with monkeypatch.context() as limited:
            limited.setattr(replicas, "MAX_REPLICA_EVENT_BYTES", 1)
            await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        state = pub.status("room")["routes"][0]
        assert (state["status"], state["acked_seq"]) == ("unavailable", 0)
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        assert pub.status("room")["routes"][0]["status"] == "acked"
        assert replicas.replica_state(api.target, room_id="room")["last_seq"] == 1


def run_publisher_process(source, control):
    rooms.local_authority_gateway_id = lambda: HOME
    pub = HostedRoomReplicationPublisher(source)
    pub.start()
    try:
        if control.poll(20):
            control.recv()
    finally:
        control.close()
        if not pub.stop(timeout=5):
            raise RuntimeError("test publisher did not stop")


@pytest.mark.asyncio
async def test_a_publisher_killed_after_the_remote_write_replays_without_duplicates(api):
    accepted, release, hold_first = asyncio.Event(), asyncio.Event(), [True]

    @web.middleware
    async def hold_after_persistence(request, handler):
        response = await handler(request)
        if request.path.endswith("/replica") and response.status == 200 and hold_first[0]:
            hold_first[0] = False
            accepted.set()
            await release.wait()
        return response

    api.app.middlewares.append(hold_after_persistence)
    context, children = multiprocessing.get_context("spawn"), []
    async with TestClient(TestServer(api.app)) as http:
        await register(http, api, (await invite(http, replication=True))["grant"])
        try:
            receive, control = context.Pipe(duplex=False)
            first = context.Process(target=run_publisher_process, args=(api.source, receive))
            children.append((first, control))
            first.start()
            receive.close()
            await asyncio.wait_for(accepted.wait(), timeout=12)
            first.terminate()
            await asyncio.to_thread(first.join, 5)
            assert not first.is_alive() and first.exitcode != 0
            release.set()
            assert replicas.replica_state(api.target, room_id="room")["last_seq"] == 1
            with rooms._transaction(api.source) as conn:
                assert conn.execute("SELECT acked_seq FROM hosted_room_replication_targets").fetchone()[0] == 0
            append(api.source, "after-crash", "Resume after process death")
            receive, control = context.Pipe(duplex=False)
            second = context.Process(target=run_publisher_process, args=(api.source, receive))
            children.append((second, control))
            second.start()
            receive.close()
            await wait_until(lambda: replicas.replica_state(api.target, room_id="room")["last_seq"] == 2)
            with rooms._transaction(api.target) as conn:
                copied = conn.execute("SELECT event_id FROM hosted_room_replica_events ORDER BY seq").fetchall()
            assert [row[0] for row in copied] == ["hello", "after-crash"]
            control.send("stop")
            await asyncio.to_thread(second.join, 5)
            assert second.exitcode == 0
        finally:
            release.set()
            for child, pipe in children:
                if child.is_alive():
                    with suppress(BrokenPipeError, OSError):
                        pipe.send("stop")
                await asyncio.to_thread(child.join, 5)
                if child.is_alive():
                    child.terminate()
                    await asyncio.to_thread(child.join, 5)
                assert not child.is_alive()
                pipe.close()
                child.close()

"""Independent-review regressions: alternate credentials and legacy history audit."""

import asyncio
import json
import sqlite3
import time
from dataclasses import replace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_driver as driver
from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from gateway.platforms import api_server
from tests.gateway.test_api_server_room_replicas import HOME, TARGET, setup  # noqa: F401
from tests.gateway.test_api_server_room_work_records import TASK, seed, capture, invitation, history, deliver
from tui_gateway.hosted_room_replication import HostedRoomReplicationPublisher
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError


def make_publisher(source, monkeypatch):
    with monkeypatch.context() as scope:
        scope.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
        return HostedRoomReplicationPublisher(source)


@pytest.mark.asyncio
@pytest.mark.parametrize(("both_revoked", "boundary"), [
    (False, "normal"), (True, "normal"), (False, "history_unavailable"),
    (False, "work_unavailable"), (False, "route_replaced"), (False, "midpoint_migration"),
    (False, "both_unavailable"),
])
async def test_work_refusal_selects_alternate_and_remembers_each_generation(setup, monkeypatch, both_revoked, boundary):
    source, target, app = setup
    requests = []
    fail_alpha = False
    beta_recovered = False
    for route in list(app.router.routes()):
        if route.resource.canonical.startswith("/v1/room-members/"):
            app.router.add_route(route.method, "/p/{profile}" + route.resource.canonical, route.handler)
    @web.middleware
    async def profile_scope(request, handler):
        token = api_server._api_request_profile.set(request.match_info.get("profile"))
        try:
            retry_failure = fail_alpha and request.path.endswith("work-records") and (
                (boundary == "work_unavailable" and request.match_info.get("profile") == "alpha")
                or (boundary == "both_unavailable" and (request.match_info.get("profile") == "alpha" or not beta_recovered)))
            if retry_failure:
                response = web.json_response({"error": {"code": "unavailable"}}, status=503)
            else:
                response = await handler(request)
            if boundary == "route_replaced" and request.path.endswith("work-records") and response.status in {401, 403}:
                old = links.load_room_link(source, room_id="room", member_id="alpha")
                replacement = peer.issue_room_grant(secret, grant_id="replacement", room_id="room", home_install_id=HOME,
                    authority_gateway_id=HOME, authority_epoch=1, member_id="alpha", target_install_id=TARGET,
                    target_profile="alpha", execution_policy_digest=old.catalog.execution_policy.policy_digest,
                    permissions=peer.invitation_permissions(True, True), issued_at=time.time() + 1)
                claims = peer.decode_room_grant(secret, replacement, permission=records.PERMISSION)
                rooms.reserve_peer_room(target, claims=claims, expires_at=claims["status_expires_at"])
                links.save_room_link(source, replace(old, grant=replacement))
            requests.append((request.path, response.status))
            return response
        finally:
            api_server._api_request_profile.reset(token)
    app.middlewares.append(profile_scope)
    async with TestClient(TestServer(app)) as http:
        secret, grants, members = peer.gateway_room_grant_secret(), {}, []
        for profile in ("alpha", "beta"):
            catalog = peer.GatewayRoomCatalog.from_mapping(peer.catalog_mapping(
                installation_id=TARGET, target_profile=profile, persistent_process=True))
            grants[profile] = peer.issue_room_grant(
                secret, grant_id=f"grant-{profile}", room_id="room", home_install_id=HOME,
                authority_gateway_id=HOME, authority_epoch=1, member_id=profile,
                target_install_id=TARGET, target_profile=profile,
                execution_policy_digest=catalog.execution_policy.policy_digest,
                permissions=peer.invitation_permissions(True, True))
            claims = peer.decode_room_grant(secret, grants[profile], permission=records.PERMISSION)
            rooms.reserve_peer_room(target, claims=claims, expires_at=claims["status_expires_at"])
            links.save_room_link(source, links.make_stored_link(
                room_id="room", member_id=profile, target_url=str(http.make_url("/")), target_profile=profile,
                grant=grants[profile], catalog=catalog, cancellation_scope_id="cancel", trace_id="trace"))
            members.append({"member_id": profile, "profile": profile, "handle": profile, "target": {
                "kind": "peer", "peer_id": profile, "installation_id": TARGET, "profile": profile,
                "capability_digest": catalog.catalog_digest}})
        with rooms._transaction(source, immediate=True) as conn:
            conn.execute("UPDATE hosted_rooms SET members_json=? WHERE room_id='room'", (json.dumps(members),))
        driver.admit_task(source, TASK, payload={"prompt": "private", "target_profile": "alpha",
                          "target_member_id": "alpha", "source_event_seq": 1}, clock=lambda: 100)
        pub = make_publisher(source, monkeypatch)
        await asyncio.to_thread(pub._publish_one, ("room", "alpha"))
        await asyncio.to_thread(pub._publish_one, ("room", "alpha"))
        rejected = () if boundary in {"work_unavailable", "both_unavailable"} else ("alpha", "beta") if both_revoked else ("alpha",)
        for profile in rejected:
            claims = peer.decode_room_grant(secret, grants[profile], permission=records.PERMISSION)
            rooms.revoke_room_grant_scope(target, claims=claims, expires_at=claims["status_expires_at"])
        fail_alpha = True
        held = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
                                    process_generation="process", ttl_seconds=30, clock=lambda: 100)
        driver.start_task(source, TASK, held, expected_cancel_generation=0, clock=lambda: 100)
        await asyncio.to_thread(pub._publish_one, ("room", "alpha"))
        with rooms._transaction(source) as conn:
            pending = conn.execute(f"SELECT record_json FROM {records.PENDING_TABLE} WHERE room_id='room'").fetchone()[0]
        if boundary == "history_unavailable":
            with rooms._transaction(source, immediate=True) as conn:
                conn.execute("UPDATE hosted_room_replication_publishers SET status='unavailable' WHERE member_id='beta'")
        if boundary == "midpoint_migration":
            with sqlite3.connect(source) as conn:
                conn.execute("ALTER TABLE hosted_room_replication_publishers DROP COLUMN work_record_status")
            pub = make_publisher(source, monkeypatch)
        if boundary == "both_unavailable":
            await asyncio.to_thread(pub._publish_one, ("room", "beta"))
            assert {r["work_record_status"] for r in pub.status()["routes"]} == {"unavailable"}
            beta_recovered = True
        # Another task-only change cannot replace the unresolved whole record.
        driver.begin_task_cancel(source, TASK, cancel_id="stop", expected_cancel_generation=0, clock=lambda: 100)
        await asyncio.to_thread(pub._publish_one, ("room", "beta"))
        if both_revoked:
            refusals = [path for path, status in requests if path.endswith("work-records") and status in {401, 403}]
            assert refusals == ["/p/alpha/v1/room-members/work-records", "/p/beta/v1/room-members/work-records"]
            restarted = make_publisher(source, monkeypatch)
            before = len(requests)
            for profile in ("alpha", "beta", "alpha", "beta"):
                await asyncio.to_thread(restarted._publish_one, ("room", profile))
            assert len(requests) == before
            with rooms._transaction(source) as conn:
                assert conn.execute(f"SELECT record_json FROM {records.PENDING_TABLE} WHERE room_id='room'").fetchone()[0] == pending
        else:
            selected = "alpha" if boundary == "route_replaced" else "beta"
            assert requests[-1] == (f"/p/{selected}/v1/room-members/work-records", 200)
            with rooms._transaction(target) as conn:
                assert conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone()[0] == pending
            assert replicas.replica_state(target, room_id="room")["work_records"]["phases"] == {"running": 1}
        assert rooms.room_state(source, room_id="room")["latest_seq"] == 1


@pytest.mark.asyncio
async def test_metadata_admission_runs_existing_legacy_audit_before_ack(setup):
    source, target, app = setup
    seed(source)
    rooms.append_event(source, room_id="room", event_id="second", kind="message.user",
                       actor={"kind": "user", "id": "owner"}, payload={"text": "second"},
                       authority_gateway_id=HOME, authority_epoch=1)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        with sqlite3.connect(target) as old:
            old.execute("DELETE FROM hosted_room_replica_events WHERE room_id='room' AND seq=1")
            assert old.execute("SELECT last_seq,quarantine_reason FROM hosted_room_replicas WHERE room_id='room'").fetchone() == (2, None)
        with pytest.raises(PeerRunsHTTPError) as error:
            await deliver(client, grant, capture(source))
        assert error.value.status_code == 409
        # Do not call replica_state to repair the missing audit after admission.
        with sqlite3.connect(target) as conn:
            assert conn.execute("SELECT quarantine_reason FROM hosted_room_replicas WHERE room_id='room'").fetchone()[0] == "non_contiguous_history"
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (records.TARGET_TABLE,)).fetchone():
                assert conn.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 0

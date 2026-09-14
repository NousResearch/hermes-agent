"""Real HTTP, signatures, passive SQLite retention and publisher consumption."""

import asyncio
import copy
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
from tests.gateway.test_api_server_room_replicas import HOME, TARGET, KEY, MEMBERS, setup  # noqa: F401
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_replication import HostedRoomReplicationPublisher

TASK = driver.TaskIdentity("room", "task", "thread", "turn")


def seed(source):
    driver.admit_task(source, TASK, payload={"target_profile": "default", "target_member_id": "reviewer",
                      "prompt": "PRIVATE_PROMPT /private/workspace", "source_event_seq": 1}, clock=lambda: 100)


def capture(source):
    return records.capture(source, room_id="room", local_gateway_id=HOME)


async def invitation(http, **overrides):
    body = {"room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME,
            "authority_epoch": 1, "member_id": "reviewer", "replication": True, "work_records": True}
    body.update(overrides)
    response = await http.post("/v1/room-members/invitations", json=body, headers={"Authorization": f"Bearer {KEY}"})
    assert response.status == 201, await response.text()
    reply = await response.json()
    assert reply["work_records_version"] == records.VERSION
    return reply


async def history(client, token, source):
    return await asyncio.to_thread(client.replicate_page, grant=token, target_profile="default", room_id="room",
                                   room_name="Workshop", members=MEMBERS, page=rooms.read_events(source, room_id="room", include_disbanded=True))


async def deliver(client, token, record):
    return await asyncio.to_thread(client.replicate_work_records, grant=token, target_profile="default", record=record)


def publisher(source, http, reply, monkeypatch):
    links.save_room_link(source, links.make_stored_link(
        room_id="room", member_id="reviewer", target_url=str(http.make_url("/")), target_profile="default",
        grant=reply["grant"], catalog=peer.GatewayRoomCatalog.from_mapping(reply["catalog"]),
        cancellation_scope_id="cancel", trace_id="trace"))
    with monkeypatch.context() as scope:
        scope.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
        return HostedRoomReplicationPublisher(source)


@pytest.mark.asyncio
async def test_continuous_history_and_task_changes_deliver_over_http_after_lost_record_ack(setup, monkeypatch):
    source, target, app = setup
    seed(source)
    sent = []

    @web.middleware
    async def lose_first_record_ack(request, handler):
        response = await handler(request)
        if request.path.endswith("/work-records"):
            assert response.status == 200
            with sqlite3.connect(target) as conn:
                accepted = conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone()
            sent.append(json.loads(accepted[0]))
            if len(sent) == 1:
                return web.json_response({"error": {"code": "unavailable"}}, status=503)
        return response

    app.middlewares.append(lose_first_record_ack)
    async with TestClient(TestServer(app)) as http:
        invited = await invitation(http)
        pub = publisher(source, http, invited, monkeypatch)
        for turn in range(6):
            rooms.append_event(source, room_id="room", event_id=f"busy-{turn}", kind="message.user",
                               actor={"kind": "user", "id": "owner"}, payload={"text": "Continue"},
                               authority_gateway_id=HOME, authority_epoch=1)
            if turn == 1:
                held = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
                                            process_generation="process", ttl_seconds=30, clock=lambda: 100)
                driver.start_task(source, TASK, held, expected_cancel_generation=0, clock=lambda: 100)
            if turn == 2:
                pub = publisher(source, http, invited, monkeypatch)
            await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        assert len(sent) >= 3
        assert sent[0] == sent[1]
        assert sent[0]["tasks"][0]["phase"] == "queued"
        assert sent[-1]["tasks"][0]["phase"] == "running"
        latest = rooms.room_state(source, room_id="room")["latest_seq"]
        state = replicas.replica_state(target, room_id="room")
        assert state["last_seq"] == latest
        assert latest - state["work_records"]["history"]["seq"] <= 2
        assert state["work_records"]["source_loss_safe"] is False
        assert rooms.list_rooms(target) == []


@pytest.mark.asyncio
async def test_explicit_records_are_passively_retained_and_summarized(setup):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        first = capture(source)
        assert (await deliver(client, grant, first))["revision"] == first["revision"]
        assert await deliver(client, grant, first) == await deliver(client, grant, first)
        summary = replicas.replica_state(target, room_id="room")["work_records"]
        assert summary["tasks"][0]["task_id"] == TASK.task_id
        assert summary["phases"] == {"queued": 1}
        assert summary["source_loss_safe"] is False
        assert "PRIVATE_PROMPT" not in json.dumps(summary)
        assert rooms.list_rooms(target) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("auth", ["history_only", "bearer", "missing"])
async def test_old_or_broad_auth_cannot_deliver_records(setup, auth):
    source, _, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        token = (await invitation(http, work_records=False))["grant"]
        headers = {"history_only": {"Authorization": f"HermesRoom {token}"},
                   "bearer": {"Authorization": f"Bearer {KEY}"}, "missing": {}}[auth]
        response = await http.post("/v1/room-members/work-records", json={"record": capture(source)}, headers=headers)
        assert response.status == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["digest", "revision", "prefix", "extra", "authority", "roster"])
async def test_revision_scope_prefix_and_privacy_schema_reject_conflicts(setup, mutation):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        first = capture(source)
        await deliver(client, grant, first)
        bad = copy.deepcopy(first)
        if mutation == "digest":
            bad["digest"] = "a" * 64
        elif mutation == "revision":
            bad["tasks"][0]["phase"] = "running"
        elif mutation == "prefix":
            bad["history"]["event_sha256"] = "a" * 64
        elif mutation == "extra":
            bad["tasks"][0]["prompt"] = "MUST_NOT_STORE"
        elif mutation == "authority":
            bad["authority"]["epoch"] = 2
        else:
            bad["roster_sha256"] = "a" * 64
        if mutation != "digest":
            bad["digest"] = records.digest({k: v for k, v in bad.items() if k not in {"revision", "digest"}})
        with pytest.raises(PeerRunsHTTPError):
            await deliver(client, grant, bad)
        assert replicas.replica_state(target, room_id="room")["work_records"]["digest"] == first["digest"]


@pytest.mark.asyncio
async def test_revocation_is_rechecked_after_http_auth_before_target_write(setup, monkeypatch):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        record = capture(source)
        claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), grant, permission=records.PERMISSION)
        original = records.validate
        def revoke(value):
            checked = original(value)
            monkeypatch.setattr(records, "validate", original)
            rooms.revoke_room_grant_scope(target, claims=claims, expires_at=claims["status_expires_at"])
            return checked
        monkeypatch.setattr(records, "validate", revoke)
        with pytest.raises(PeerRunsHTTPError) as error:
            await deliver(client, grant, record)
        assert error.value.status_code == 401
        assert replicas.replica_state(target, room_id="room")["work_records"]["availability"] == "not_retained"


@pytest.mark.asyncio
async def test_canonical_disband_reclaims_records_and_rejects_late_delivery(setup):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        record = capture(source)
        await deliver(client, grant, record)
        rooms.disband_room(source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
        await history(client, grant, source)
        with pytest.raises(PeerRunsHTTPError):
            await deliver(client, grant, record)
        state = replicas.replica_state(target, room_id="room")
        assert state["work_records"]["availability"] == "not_retained"
        assert state["disbanded_at"] is not None


@pytest.mark.asyncio
async def test_publisher_loss_restart_and_task_only_change_use_same_pending_record(setup, monkeypatch):
    source, target, app = setup
    seed(source)
    received = []
    @web.middleware
    async def lose_ack(request, handler):
        response = await handler(request)
        if request.path.endswith("/work-records") and response.status == 200:
            with rooms._transaction(target) as conn:
                received.append(json.loads(conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone()[0]))
            if len(received) == 1:
                request.transport.close()
        return response
    app.middlewares.append(lose_ack)
    async with TestClient(TestServer(app)) as http:
        reply = await invitation(http)
        links.save_room_link(source, links.make_stored_link(
            room_id="room", member_id="reviewer", target_url=str(http.make_url("/")), target_profile="default",
            grant=reply["grant"], catalog=peer.GatewayRoomCatalog.from_mapping(reply["catalog"]),
            cancellation_scope_id="cancel", trace_id="trace"))
        with monkeypatch.context() as scope:
            scope.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
            first = HostedRoomReplicationPublisher(source)
        key = ("room", "reviewer")
        await asyncio.to_thread(first._publish_one, key)  # history
        await asyncio.to_thread(first._publish_one, key)  # records: accepted, reply lost
        assert first.status()["work_records"][0]["status"] == "unavailable"
        held = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
                                    process_generation="process", ttl_seconds=30, clock=lambda: 100)
        driver.start_task(source, TASK, held, expected_cancel_generation=0, clock=lambda: 100)
        with monkeypatch.context() as scope:
            scope.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
            second = HostedRoomReplicationPublisher(source)
        await asyncio.to_thread(second._publish_one, key)
        assert received[0] == received[1]
        await asyncio.to_thread(second._publish_one, key)
        assert received[2]["revision"] > received[1]["revision"]
        assert received[2]["history"] == received[1]["history"]
        assert replicas.replica_state(target, room_id="room")["work_records"]["phases"] == {"running": 1}
        await asyncio.to_thread(second._publish_one, key)
        assert len(received) == 3
        assert second.status()["source_loss_safe"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement_first", [False, True])
async def test_retirement_reclaims_records_and_fences_older_raw_writers(setup, retirement_first):
    from gateway import hosted_room_replica_retirement as retirement
    from tests.gateway.test_api_replica_retirement import prepare, enroll, close_source, HOME_SECRET
    source, target, app = setup
    seed(source)
    if retirement_first:
        with retirement._transaction(target):
            pass
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        record = capture(source)
        await deliver(client, grant, record)
        enrollment = prepare(source, http)
        await enroll(http, enrollment)
        await close_source(source, client, grant)
        notice = retirement.materialize_notice(source, enrollment_id=enrollment["enrollment_id"],
                                               local_gateway_id=HOME, secret_loader=lambda: HOME_SECRET)
        assert (await asyncio.to_thread(client.retire_replica, notice))["retired"]
        with sqlite3.connect(target) as old:
            old.execute("PRAGMA foreign_keys=OFF")
            assert old.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 0
            for verb in ("INSERT", "INSERT OR REPLACE"):
                with pytest.raises(sqlite3.IntegrityError):
                    old.execute(f"{verb} INTO {records.TARGET_TABLE} (room_id,revision,digest,record_json) VALUES (?,?,?,?)",
                                ("room", record["revision"], record["digest"], records.encode(record)))
            assert old.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "replica"
        with pytest.raises(PeerRunsHTTPError):
            await deliver(client, grant, record)


@pytest.mark.asyncio
async def test_quarantine_denies_work_record_writes(setup):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        record = capture(source)
        await deliver(client, grant, record)
        with sqlite3.connect(target) as conn:
            conn.execute("UPDATE hosted_room_replicas SET quarantine_reason='test' WHERE room_id='room'")
        with pytest.raises(PeerRunsHTTPError):
            await deliver(client, grant, record)
        assert replicas.replica_state(target, room_id="room")["work_records"]["availability"] == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["room", "target", "profile", "expired"])
async def test_wrong_or_expired_scoped_grant_never_writes(setup, scope):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), grant, permission=records.PERMISSION)
        fields = {k: claims[k] for k in ("grant_id", "room_id", "home_install_id", "authority_gateway_id",
                  "authority_epoch", "member_id", "target_install_id", "target_profile", "execution_policy_digest", "permissions")}
        changes = {"room": {"room_id": "wrong"}, "target": {"target_install_id": "wrong"},
                   "profile": {"target_profile": "wrong"}, "expired": {"issued_at": time.time() - 100, "ttl_seconds": 1}}
        bad = peer.issue_room_grant(peer.gateway_room_grant_secret(), **{**fields, **changes[scope]})
        with pytest.raises(PeerRunsHTTPError):
            await deliver(client, bad, capture(source))
        assert replicas.replica_state(target, room_id="room")["work_records"]["availability"] == "not_retained"


@pytest.mark.asyncio
async def test_stale_revision_cannot_replace_newer_record(setup):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        first = capture(source)
        await deliver(client, grant, first)
        held = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
                                    process_generation="process", ttl_seconds=30, clock=lambda: 100)
        driver.start_task(source, TASK, held, expected_cancel_generation=0, clock=lambda: 100)
        second = capture(source)
        await deliver(client, grant, second)
        with pytest.raises(PeerRunsHTTPError):
            await deliver(client, grant, first)
        assert replicas.replica_state(target, room_id="room")["work_records"]["revision"] == second["revision"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["disband", "delete"])
async def test_old_history_writer_reclaims_record_payload_without_removing_identity(setup, operation):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        record = capture(source)
        await deliver(client, grant, record)
        with sqlite3.connect(target) as old:
            old.execute("PRAGMA foreign_keys=OFF")
            sql = "UPDATE hosted_room_replicas SET disbanded_at=123 WHERE room_id='room'" if operation == "disband" else "DELETE FROM hosted_room_replicas WHERE room_id='room'"
            old.execute(sql)
            assert old.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 0
            assert old.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "replica"
            with pytest.raises(sqlite3.IntegrityError):
                old.execute(f"INSERT INTO {records.TARGET_TABLE} (room_id,revision,digest,record_json) VALUES (?,?,?,?)",
                            ("room", record["revision"], record["digest"], records.encode(record)))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["partial_ack", "unsupported", "route_race"])
async def test_publisher_requires_whole_ack_and_fences_route_changes_without_breaking_history(setup, monkeypatch, mode):
    source, target, app = setup
    seed(source)
    altered = []
    @web.middleware
    async def alter_response(request, handler):
        if mode == "unsupported" and request.path.endswith("/work-records"):
            altered.append(True)
            return web.json_response({"error": {"code": "unsupported"}}, status=404)
        response = await handler(request)
        if request.path.endswith("/work-records") and response.status == 200:
            altered.append(True)
            if mode == "partial_ack":
                body = json.loads(response.body)
                body.pop("digest")
                return web.json_response(body)
            if len(altered) == 1:
                stored = links.load_room_link(source, room_id="room", member_id="reviewer")
                links.save_room_link(source, replace(stored, trace_id="replacement-route"))
        return response
    app.middlewares.append(alter_response)
    async with TestClient(TestServer(app)) as http:
        pub = publisher(source, http, await invitation(http), monkeypatch)
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        expected = {"partial_ack": "invalid_ack", "unsupported": "rejected", "route_race": "pending"}[mode]
        assert pub.status()["work_records"][0]["status"] == expected
        assert pub.status()["routes"][0]["status"] == "acked"
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        if mode == "route_race":
            assert pub.status()["work_records"][0]["status"] == "acked"
            assert len(altered) == 2
        else:
            assert len(altered) == 1
        rooms.append_event(source, room_id="room", event_id="later", kind="message.user",
                           actor={"kind": "user", "id": "owner"}, payload={"text": "still copies"},
                           authority_gateway_id=HOME, authority_epoch=1)
        await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        assert replicas.replica_state(target, room_id="room")["last_seq"] == 2


@pytest.mark.asyncio
async def test_history_only_route_never_captures_or_sends_work_records(setup, monkeypatch):
    source, target, app = setup
    seed(source)
    async with TestClient(TestServer(app)) as http:
        pub = publisher(source, http, await invitation(http, work_records=False), monkeypatch)
        for _ in range(3):
            await asyncio.to_thread(pub._publish_one, ("room", "reviewer"))
        assert pub.status()["work_records"] == []
        assert replicas.replica_state(target, room_id="room")["work_records"]["availability"] == "not_retained"


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [{"work_records": "yes"}, {"work_records": 1}, {"work_records": True, "replication": False}])
async def test_http_invitation_rejects_implicit_or_malformed_opt_in(setup, options):
    _, _, app = setup
    async with TestClient(TestServer(app)) as http:
        body = {"room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME,
                "authority_epoch": 1, "member_id": "reviewer", "replication": True, **options}
        response = await http.post("/v1/room-members/invitations", json=body, headers={"Authorization": f"Bearer {KEY}"})
        assert response.status == 400


@pytest.mark.asyncio
async def test_work_record_http_body_is_bounded(setup, monkeypatch):
    _, _, app = setup
    monkeypatch.setattr(records, "MAX_BYTES", 256)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        response = await http.post("/v1/room-members/work-records", json={"record": "x" * 2000},
                                   headers={"Authorization": f"HermesRoom {grant}"})
        assert response.status in {400, 413}


@pytest.mark.asyncio
async def test_exact_run_receipt_and_stop_generations_are_retained_without_results(setup):
    source, target, app = setup
    seed(source)
    held = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
                                process_generation="process", ttl_seconds=30, clock=lambda: 100)
    driver.start_task(source, TASK, held, expected_cancel_generation=0, clock=lambda: 100)
    receipt = {"room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME, "authority_epoch": 1,
               "member_id": "reviewer", "target_install_id": TARGET, "target_profile": "default", "task_id": "task",
               "execution_generation": 1, "run_id": "accepted-run", "session_id": "accepted-session"}
    rooms.upsert_remote_run_receipt(source, record=receipt)
    async with TestClient(TestServer(app)) as http:
        grant = (await invitation(http))["grant"]
        client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=3)
        await history(client, grant, source)
        first = capture(source)
        await deliver(client, grant, first)
        assert replicas.replica_state(target, room_id="room")["work_records"]["receipts"] == [receipt]
        rooms.request_room_stop(source, room_id="room", cancel_id="stop", expected_gateway_id=HOME, expected_epoch=1)
        driver.begin_task_cancel(source, TASK, cancel_id="stop", expected_cancel_generation=0, clock=lambda: 100)
        await history(client, grant, source)
        stopping = capture(source)
        await deliver(client, grant, stopping)
        summary = replicas.replica_state(target, room_id="room")["work_records"]
        assert summary["phases"] == {"stopping": 1}
        assert summary["tasks"][0]["cancel_generation"] == 1
        assert summary["stop"]["cancel_id"] == "stop"
        assert summary["receipts"] == [receipt]
        driver.complete_task_cancel(source, TASK, cancel_id="stop", expected_cancel_generation=1, clock=lambda: 100)
        cancelled = capture(source)
        assert cancelled["history"] == stopping["history"]
        assert cancelled["revision"] > stopping["revision"]
        await deliver(client, grant, cancelled)
        assert replicas.replica_state(target, room_id="room")["work_records"]["phases"] == {"cancelled": 1}

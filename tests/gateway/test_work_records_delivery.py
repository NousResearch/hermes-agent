"""Task evidence travels with history: anchored, frozen until its exact ACK, and fair under load and loss."""

import asyncio
import json
import sqlite3
import threading
import time
from contextlib import closing
from dataclasses import replace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import hermes_state_wal
from gateway import hosted_room_driver as driver
from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replication as publisher
from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.fixtures.passive_copy import (  # noqa: F401
    API_KEY, EVIDENCE, HOME, KEY, MEMBERS, SECRET, TARGET, TASK, add_task, admit, api, append,
    both_routes_carry_evidence, copying, http_error, invite, pair, reserve, save_link, start)
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

OTHER = ("room", "z-other")


def pending(source):
    with rooms._transaction(source) as conn:
        row = conn.execute(f"SELECT record_json FROM {records.PENDING_TABLE} WHERE room_id='room'").fetchone()
    return json.loads(row[0]) if row else None


def delivery(pub, index=0):
    return pub.status("room")["work_records"][index]


@pytest.mark.parametrize("turns", [5, 20, 100])
def test_a_busy_conversation_delivers_current_task_snapshots_without_a_quiet_turn(copying, turns):
    for turn in range(turns):
        add_task(copying.source, str(turn))
        copying.pub._publish_one(KEY)
        if turn >= 1:
            # One page per turn: a healthy participant gets a snapshot within the next turn.
            latest = rooms.room_state(copying.source, room_id="room")["latest_seq"]
            assert copying.records and latest - copying.records[-1]["history"]["seq"] <= 2
    assert len(copying.records[-1]["tasks"]) >= turns - 2
    assert "PRIVATE_PROMPT" not in json.dumps(copying.records)


def test_a_frozen_record_keeps_its_anchor_across_new_events_a_lost_reply_and_a_restart(copying, monkeypatch):
    monkeypatch.setattr(publisher, "PAGE_LIMIT", 1)
    for turn in range(4):
        add_task(copying.source, str(turn))
    copying.http.lose_ack = True
    copying.pub._publish_one(KEY)
    frozen = pending(copying.source)
    assert frozen is not None
    copying.pub = publisher.HostedRoomReplicationPublisher(copying.source)
    for turn in range(frozen["history"]["seq"] + 2):
        add_task(copying.source, f"growing-{turn}")
        copying.pub._publish_one(KEY)
        if copying.records:
            break
    assert copying.records[0] == frozen
    assert copying.http.requests[0] == copying.http.requests[1]


def test_a_lost_record_reply_resends_the_identical_record_while_history_keeps_growing(copying):
    add_task(copying.source, "first")
    copying.pub._publish_one(KEY)
    copying.http.lose_record_ack = True
    add_task(copying.source, "second")
    copying.pub._publish_one(KEY)
    frozen = copying.records[-1]
    copying.pub = publisher.HostedRoomReplicationPublisher(copying.source)
    add_task(copying.source, "third")
    copying.pub._publish_one(KEY)
    assert copying.records[-1] == copying.records[-2] == frozen
    assert copying.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(copying.source, room_id="room")["latest_seq"]
    for suffix in ("fourth", "fifth"):
        add_task(copying.source, suffix)
        copying.pub._publish_one(KEY)
    assert copying.records[-1]["revision"] > frozen["revision"]


def test_a_quiet_room_sends_nothing_unchanged(copying):
    add_task(copying.source, "only")
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    delivered, history = list(copying.records), list(copying.http.requests)
    assert delivered
    for _ in range(3):
        copying.pub._publish_one(KEY)
    assert (copying.records, copying.http.requests) == (delivered, history)


def test_an_empty_room_copies_its_empty_history_before_the_first_record(copying):
    empty = copying.source.with_name("empty.db")
    rooms.create_room(empty, room_id="room", name="Empty", members=MEMBERS, authority_gateway_id=HOME)
    save_link(empty, permissions=EVIDENCE)
    pub = publisher.HostedRoomReplicationPublisher(empty)
    copying.http.source = empty
    pub._publish_one(KEY)
    assert copying.records == [] and copying.http.requests[-1][1]["page"]["cursor"] == 0
    pub._publish_one(KEY)
    assert copying.records[-1]["history"]["seq"] == 0
    assert delivery(pub)["status"] == "acked"


def test_an_unavailable_record_route_never_stops_history(copying):
    copying.http.record_error = TimeoutError("record endpoint unavailable")
    for turn in range(5):
        add_task(copying.source, str(turn))
        copying.pub._publish_one(KEY)
    assert copying.records and all(record == copying.records[0] for record in copying.records)
    assert copying.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(copying.source, room_id="room")["latest_seq"]


def test_a_route_without_the_evidence_opt_in_never_captures_or_sends_records(pair):
    admit(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    for _ in range(3):
        pub._publish_one(KEY)
    assert pub.status()["work_records"] == []
    assert replicas.copy_state(pair.target, room_id="room")["work_records"]["availability"] == "not_retained"


@pytest.mark.parametrize("busy", [False, True], ids=["quiet", "continuous-history"])
def test_a_recovered_alternate_route_gets_a_work_attempt(pair, busy):
    both_routes_carry_evidence(pair)
    recovered, attempts = [False], []

    def refuse_until_recovered(request, _body):
        member = peer.decode_room_grant(SECRET, request.get_header("Authorization").removeprefix("HermesRoom "),
                                        permission=records.PERMISSION)["member_id"]
        attempts.append(member)
        if member == "reviewer" or not recovered[0]:
            raise http_error(request, 503)

    pair.http.record_before = refuse_until_recovered
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    pub._publish_one(KEY)
    pub._publish_one(OTHER)
    assert attempts == ["reviewer", "z-other"]
    assert {row["work_record_status"] for row in pub.status()["routes"]} == {"unavailable"}
    frozen = pair.http.records[0]
    recovered[0] = True
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    for turn in range(6):
        if busy:
            append(pair.source, f"continuing-{turn}")
        pub._publish_one(OTHER if turn % 2 == 0 else KEY)
        if delivery(pub)["status"] == "acked":
            break
    assert all(body == frozen for body in pair.http.records)
    if busy:
        assert pair.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(pair.source, room_id="room")["latest_seq"]
    assert delivery(pub)["status"] == "acked"


@pytest.mark.parametrize("busy", [False, True], ids=["quiet", "continuous-history"])
def test_a_recovered_work_route_with_a_stale_history_failure_still_delivers(pair, busy):
    both_routes_carry_evidence(pair)
    admit(pair.source)
    recovered, work_attempts, history_failures = [False], [], []

    def member_of(request):
        return peer.decode_room_grant(SECRET, request.get_header("Authorization").removeprefix("HermesRoom "),
                                      permission="replicate")["member_id"]

    def history(request, _body):
        if member_of(request) == "z-other" and not recovered[0]:
            history_failures.append("z-other")
            raise http_error(request, 503)

    def work(request, _body):
        work_attempts.append(member_of(request))
        if member_of(request) == "reviewer" or not recovered[0]:
            raise http_error(request, 503)

    pair.http.before, pair.http.record_before = history, work
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    pub._publish_one(KEY)
    append(pair.source, "second-route-history-failure")
    pub._publish_one(OTHER)
    assert (work_attempts, history_failures) == (["reviewer", "z-other"], ["z-other"])
    pub._publish_one(KEY)
    status = {row["member_id"]: row for row in pub.status()["routes"]}
    assert (status["z-other"]["status"], status["reviewer"]["status"]) == ("unavailable", "acked")
    assert {row["work_record_status"] for row in status.values()} == {"unavailable"}
    frozen = pair.http.records[0]
    assert frozen["tasks"][0]["phase"] == "queued"
    recovered[0] = True
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    for turn in range(6):
        if busy:
            append(pair.source, f"busy-after-recovery-{turn}")
        pub._publish_one(OTHER if turn % 2 == 0 else KEY)
        if delivery(pub)["status"] == "acked":
            break
    assert all(body == frozen for body in pair.http.records)
    if busy:
        assert pair.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(pair.source, room_id="room")["latest_seq"]
    assert delivery(pub)["status"] == "acked"


def test_an_unacknowledged_anchor_still_prefers_a_healthy_history_route(pair, monkeypatch):
    both_routes_carry_evidence(pair)
    for turn in range(4):
        add_task(pair.source, str(turn))
    failures = {"reviewer": 1, "z-other": 1}

    def fail_once_each(request, _body):
        member = peer.decode_room_grant(SECRET, request.get_header("Authorization").removeprefix("HermesRoom "),
                                        permission="replicate")["member_id"]
        if failures[member]:
            failures[member] -= 1
            raise http_error(request, 503)

    pair.http.before = fail_once_each
    pair.http.record_before = lambda *_: pytest.fail("no record is anchored yet")
    monkeypatch.setattr(publisher, "PAGE_LIMIT", 1)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    pub._publish_one(OTHER)
    assert failures == {"reviewer": 0, "z-other": 0}
    pub._publish_one(KEY)
    assert pair.http.requests[-1][1]["page"]["cursor"] == 1
    assert pub._select_route(pub._load_route(OTHER)).key == KEY
    pub._publish_one(OTHER)
    assert pair.http.requests[-1][1]["page"]["cursor"] == 2


@pytest.mark.parametrize("bad_ack", [False, True], ids=["valid-ack", "refused-ack"])
def test_no_deliverable_work_keeps_healthy_history_first(pair, bad_ack):
    from tests.gateway.fixtures.passive_copy import add_route
    add_route(pair)  # A history-only alternate.
    pair.link = save_link(pair.source, permissions=EVIDENCE)
    reserve(pair.target, pair.link.grant)
    history_attempts, fail_first_primary = [], [True]

    def history(request, _body):
        member = peer.decode_room_grant(SECRET, request.get_header("Authorization").removeprefix("HermesRoom "),
                                        permission="replicate")["member_id"]
        history_attempts.append(member)
        failing = member == "z-other" or fail_first_primary[0]
        if member == "reviewer":
            fail_first_primary[0] = False
        if failing:
            raise http_error(request, 503)

    pair.http.before = history
    if bad_ack:
        pair.http.record_reply_transform = lambda reply: {**reply, "revision": reply["revision"] + 1}
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)  # The primary's history fails once.
    pub._publish_one(OTHER)  # The alternate's history failure is now durable.
    pub._publish_one(KEY)  # The primary catches up; the alternate stays unavailable.
    assert history_attempts == ["reviewer", "z-other", "reviewer"]
    pub._publish_one(KEY)  # A valid or refused record ACK, independently of healthy history.
    assert delivery(pub)["status"] == ("invalid_ack" if bad_ack else "acked")
    assert len(pair.http.records) == 1
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    for turn in range(4):
        append(pair.source, f"later-{turn}")
        pub._publish_one(OTHER if turn % 2 == 0 else KEY)
    if bad_ack:
        assert len(pair.http.records) == 1  # A refused generation is never retried.
    assert pair.http.requests[-1][1]["page"]["cursor"] == rooms.room_state(pair.source, room_id="room")["latest_seq"]


@pytest.mark.parametrize("failure", ["unauthorized", "rejected", "invalid_ack"])
def test_refusals_are_bounded_per_route_even_after_a_restart(pair, failure):
    both_routes_carry_evidence(pair)
    attempts = []

    def refuse(request, _body):
        attempts.append(peer.decode_room_grant(
            SECRET, request.get_header("Authorization").removeprefix("HermesRoom "),
            permission=records.PERMISSION)["member_id"])
        if failure != "invalid_ack":
            raise http_error(request, 403 if failure == "unauthorized" else 422, failure)

    pair.http.record_before = refuse
    pair.http.record_reply_transform = lambda reply: {}
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    for _ in range(8):
        pub._publish_one(KEY)
        pub._publish_one(OTHER)
    assert sorted(attempts) == ["reviewer", "z-other"]
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    for _ in range(4):
        pub._publish_one(KEY)
        pub._publish_one(OTHER)
    assert sorted(attempts) == ["reviewer", "z-other"]


def test_an_expired_source_anchor_stays_visible_after_a_restart_until_a_new_one_is_acked(copying):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    delivered = list(copying.records)
    with sqlite3.connect(copying.source) as conn:
        conn.execute("DELETE FROM hosted_room_events WHERE room_id='room'")
    copying.pub._publish_one(KEY)
    restarted = publisher.HostedRoomReplicationPublisher(copying.source)
    status = restarted.status("room")
    assert status["routes"][0]["work_record_status"] == status["work_records_error"] == "source_prefix_expired"
    assert copying.records == delivered
    append(copying.source, "new-retained-anchor")
    restarted._publish_one(KEY)
    restarted._publish_one(KEY)
    assert restarted.status("room")["work_records_error"] is None
    assert restarted.status("room")["routes"][0]["work_record_status"] == "acked"
    assert copying.records[-1]["history"]["seq"] > delivered[-1]["history"]["seq"]


@pytest.mark.parametrize("after_failure", ["unchanged", "history_pending", "refused"])
def test_a_successful_recapture_clears_an_error_only_for_acknowledged_work(copying, monkeypatch, after_failure):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    delivered = list(copying.records)
    with monkeypatch.context() as fault:
        def fail_capture(*args, **kwargs):
            raise sqlite3.OperationalError("temporary source read failure")
        fault.setattr(records, "capture_locked", fail_capture)
        copying.pub._publish_one(KEY)
    restarted = publisher.HostedRoomReplicationPublisher(copying.source)
    assert restarted.status("room")["work_records_error"] == "work_record_capture_unavailable"
    if after_failure == "history_pending":
        append(copying.source, "new-history-not-yet-acknowledged")
    elif after_failure == "refused":
        with sqlite3.connect(copying.source) as conn:
            conn.execute(f"UPDATE {records.PENDING_TABLE} SET status='rejected' WHERE room_id='room'")
    restarted._publish_one(KEY)
    status = publisher.HostedRoomReplicationPublisher(copying.source).status("room")
    assert copying.records == delivered
    if after_failure == "unchanged":
        assert status["work_records_error"] is None and status["routes"][0]["work_record_status"] == "acked"
    else:
        assert status["work_records_error"] == "work_record_capture_unavailable"
        assert status["routes"][0]["work_record_status"] != "acked" and status["work_records"][0]["status"] != "acked"


def test_invalid_pending_evidence_never_crosses_http_or_fakes_an_unchanged_ack(copying):
    copying.pub._publish_one(KEY)  # Freeze the record and copy its anchor.
    assert copying.records == []
    with sqlite3.connect(copying.source) as conn:
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET revision=revision+3,status='acked'")
        before = conn.execute(f"SELECT revision,digest,record_json,status,route_generation FROM {records.PENDING_TABLE}").fetchone()
    for _ in range(3):
        copying.pub._publish_one(KEY)
    assert copying.records == []
    assert copying.pub.status("room")["routes"][0]["work_record_status"] == "invalid_work_evidence"
    with closing(open_sqlite(copying.source)) as conn:
        assert not records.pending_delivery_is_anchored_locked(conn, room_id="room", target_install_id=TARGET, through_seq=99)
        assert tuple(conn.execute(f"SELECT revision,digest,record_json,status,route_generation FROM {records.PENDING_TABLE}").fetchone()) == before
        assert conn.execute(f"SELECT disposition FROM {records.PENDING_TABLE}").fetchone()[0] == "invalid"


def test_metadata_invalid_pending_evidence_fails_closed_before_http(copying):
    copying.pub._publish_one(KEY)
    with sqlite3.connect(copying.source) as raw:
        raw.execute(f"UPDATE {records.PENDING_TABLE} SET digest='wrong' WHERE room_id='room'")
    assert delivery(copying.pub)["disposition"] == "invalid"
    for _ in range(3):
        copying.pub._publish_one(KEY)
    assert copying.records == []


def test_unrelated_invalid_source_pending_and_history_never_block_delivery(copying):
    for room_id in ("invalid-current", "invalid-history"):
        rooms.create_room(copying.source, room_id=room_id, name=room_id, members=MEMBERS, authority_gateway_id=HOME)
        with rooms._transaction(copying.source, immediate=True) as conn:
            records.prepare_delivery_locked(conn, room_id=room_id, target_install_id="elsewhere",
                                            route_generation="frozen-route", local_gateway_id=HOME, through_seq=0)
    with rooms._transaction(copying.source, immediate=True) as conn:
        for room_id in ("invalid-current", "invalid-history"):
            for table in (records.SOURCE_TABLE, records.PENDING_TABLE):
                conn.execute(f"UPDATE {table} SET digest='damaged' WHERE room_id=?", (room_id,))
                if room_id == "invalid-history":
                    conn.execute(f"UPDATE {table} SET disposition='historical' WHERE room_id=?", (room_id,))

    def others(table):
        with closing(open_sqlite(copying.source)) as conn:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table} WHERE room_id!='room' ORDER BY room_id")]

    before = {table: others(table) for table in (records.SOURCE_TABLE, records.PENDING_TABLE)}
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    assert copying.records and delivery(copying.pub)["status"] == "acked"
    everything = copying.pub.status()
    assert everything["error"] is None
    assert {table: others(table) for table in before} == before
    assert next(r for r in everything["work_records"] if r["room_id"] == "room")["status"] == "acked"
    assert all(r["disposition"] == "invalid" for r in everything["work_records"] if r["room_id"].startswith("invalid-"))


def test_a_quarantined_copy_held_on_the_home_store_never_blocks_its_own_delivery(copying, tmp_path):
    from gateway import hosted_room_work_storage as storage
    other_home = tmp_path / "other-owner.db"
    rooms.create_room(other_home, room_id="bad-copy", name="Other owner", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(other_home, room_id="bad-copy", event_id="event", kind="message.user",
                       actor={"kind": "user", "id": "owner"}, payload={"text": "copied"}, authority_gateway_id=HOME,
                       authority_epoch=1)
    record = records.capture(other_home, room_id="bad-copy", local_gateway_id=HOME)
    replicas.ingest_page(copying.source, room_id="bad-copy", room_name="Other owner", members=MEMBERS,
                         page=rooms.read_events(other_home, room_id="bad-copy"))
    with rooms._transaction(copying.source, immediate=True) as conn:
        records.initialize(conn)
        storage.save_locked(conn, records.TARGET_TABLE, record)
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET digest='wrong' WHERE room_id='bad-copy'")
        conn.execute("UPDATE hosted_room_replica_events SET payload_json='not-json' WHERE room_id='bad-copy'")
    assert replicas.copy_state(copying.source, room_id="bad-copy")["safety_status"] == "quarantined"
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    assert copying.records and delivery(copying.pub)["status"] == "acked"
    assert replicas.copy_state(copying.source, room_id="bad-copy")["safety_status"] == "quarantined"


@pytest.mark.parametrize("journal_mode", ["wal", "delete"])
def test_an_unavailable_participant_never_blocks_a_healthy_one_or_holds_the_home_store(tmp_path, monkeypatch, journal_mode):
    monkeypatch.setattr(hermes_state_wal, "resolve_journal_mode", lambda: journal_mode)
    monkeypatch.setattr(PeerRunsHTTPClient, "_request", lambda *a, **k: pytest.fail("unexpected peer request"))
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
    from tests.gateway.fixtures.passive_copy import catalog, grant, member
    source, targets, members = tmp_path / "home.db", {}, []
    for index in range(2):
        member_id, installation, url = f"member-{index}", f"install:participant-{index}", f"http://127.0.0.1:{9800 + index}"
        members.append(member(member_id, target=installation))
        token = grant(member_id=member_id, target=installation, permissions=EVIDENCE)
        target = tmp_path / f"participant-{index}.db"
        reserve(target, token)
        targets[url] = (target, installation)
        links.save_room_link(source, links.make_stored_link(
            room_id="room", member_id=member_id, target_url=url, target_profile="default", grant=token,
            catalog=catalog(installation), cancellation_scope_id="cancel", trace_id="trace"))
    rooms.create_room(source, room_id="room", name="Workshop", members=members, authority_gateway_id=HOME)
    append(source, "hello")
    admit(source, member_id="member-0")

    def copy_history(client, *, grant, custody=None, **body):
        return replicas.ingest_page(targets[client.base_url][0], custody_report=custody, **body)

    monkeypatch.setattr(PeerRunsHTTPClient, "replicate_page", copy_history)
    pub = publisher.HostedRoomReplicationPublisher(source)
    for item in members:
        pub._publish_one(("room", item["member_id"]))
    with closing(sqlite3.connect(source)) as conn:
        expected = "delete" if hermes_state_wal.is_sqlite_wal_reset_vulnerable() else journal_mode
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == expected
    blocked, release, healthy = threading.Event(), threading.Event(), threading.Event()

    def copy_records(client, *, grant, record):
        target, installation = targets[client.base_url]
        if installation.endswith("-1"):
            assert blocked.wait(3)  # Progress while the other participant is out.
        with closing(sqlite3.connect(source, timeout=1)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")  # No home transaction spans the request.
        if installation.endswith("-0"):
            blocked.set()
            release.wait()
            raise PeerRunsHTTPError("unavailable", retryable=True, ambiguous=True)
        result = records.ingest(target, record=record, token=grant, secret=SECRET,
                                target_install_id=installation, target_profile="default")
        healthy.set()
        return result

    monkeypatch.setattr(PeerRunsHTTPClient, "replicate_work_records", copy_records)
    pub.start()
    try:
        assert blocked.wait(5) and healthy.wait(3), pub.status()
        assert pub.status()["workers"] == 2
    finally:
        release.set()
        assert pub.stop(timeout=5)
    assert {row["target_install_id"]: row["status"] for row in pub.status()["work_records"]} == {
        "install:participant-0": "unavailable", "install:participant-1": "acked"}


@pytest.mark.parametrize(("both_revoked", "boundary"), [
    (False, "normal"), (True, "normal"), (False, "history_unavailable"), (False, "work_unavailable"),
    (False, "route_replaced"), (False, "both_unavailable"),
])
def test_a_refused_record_moves_to_an_alternate_and_each_refusal_is_remembered(pair, both_revoked, boundary):
    both_routes_carry_evidence(pair)
    admit(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    pub._publish_one(KEY)
    grants = {"reviewer": pair.link.grant, "z-other": links.load_room_links(pair.source)[1].grant}
    rejected = () if boundary in {"work_unavailable", "both_unavailable"} else (
        ("reviewer", "z-other") if both_revoked else ("reviewer",))
    for member_id in rejected:
        claims = peer.decode_room_grant(SECRET, grants[member_id], permission=records.PERMISSION)
        rooms.revoke_room_grant_scope(pair.target, claims=claims, expires_at=claims["status_expires_at"])
    other_recovered, sent = [False], []

    def record_hook(request, _body):
        member_id = peer.decode_room_grant(SECRET, request.get_header("Authorization").removeprefix("HermesRoom "),
                                           permission=records.PERMISSION)["member_id"]
        sent.append(member_id)
        if (boundary == "work_unavailable" and member_id == "reviewer") or (
                boundary == "both_unavailable" and (member_id == "reviewer" or not other_recovered[0])):
            raise http_error(request, 503)
        if boundary == "route_replaced" and member_id == "reviewer" and len(sent) == 1:
            # The refused route is replaced: a fresh grant must get its own attempt.
            replacement = save_link(pair.source, grant_id="replacement", permissions=EVIDENCE, issued_at=time.time() + 1)
            reserve(pair.target, replacement.grant)

    pair.http.record_before = record_hook
    start(pair.source)
    pub._publish_one(KEY)
    frozen = pending(pair.source)
    if boundary == "history_unavailable":
        with rooms._transaction(pair.source, immediate=True) as conn:
            conn.execute(f"UPDATE {publisher.ROUTES_TABLE} SET status='unavailable' WHERE member_id='z-other'")
    if boundary == "both_unavailable":
        pub._publish_one(OTHER)
        assert {r["work_record_status"] for r in pub.status()["routes"]} == {"unavailable"}
        other_recovered[0] = True
    # Another task-only change cannot replace the unresolved whole record.
    driver.begin_task_cancel(pair.source, TASK, cancel_id="stop", expected_cancel_generation=0, clock=lambda: 100)
    # Asked for the other route, selection still prefers a deliverable replacement of the refused one.
    pub._publish_one(OTHER)
    if both_revoked:
        restarted = publisher.HostedRoomReplicationPublisher(pair.source)
        before = len(sent)
        for key in (KEY, OTHER, KEY, OTHER):
            restarted._publish_one(key)
        assert len(sent) == before and pending(pair.source) == frozen
    else:
        assert sent[-1] == ("reviewer" if boundary == "route_replaced" else "z-other")
        with rooms._transaction(pair.target) as conn:
            stored = conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone()[0]
        assert json.loads(stored) == frozen
        assert replicas.copy_state(pair.target, room_id="room")["work_records"]["phases"] == {"running": 1}


# -- over real HTTP ----------------------------------------------------------------------------


def http_sender(http, timeout=3):
    return PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=timeout)


def home_publisher(api, http, invitation, monkeypatch):
    links.save_room_link(api.source, links.make_stored_link(
        room_id="room", member_id="reviewer", target_url=str(http.make_url("/")), target_profile="default",
        grant=invitation["grant"], catalog=peer.GatewayRoomCatalog.from_mapping(invitation["catalog"]),
        cancellation_scope_id="cancel", trace_id="trace"))
    with monkeypatch.context() as scope:
        scope.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
        return publisher.HostedRoomReplicationPublisher(api.source)


@pytest.mark.asyncio
async def test_continuous_history_and_task_changes_arrive_over_http_after_a_lost_record_reply(api, monkeypatch):
    admit(api.source)
    received = []

    @web.middleware
    async def lose_first_record_reply(request, handler):
        response = await handler(request)
        if request.path.endswith("/work-records"):
            assert response.status == 200
            with sqlite3.connect(api.target) as conn:
                received.append(json.loads(conn.execute(
                    f"SELECT record_json FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone()[0]))
            if len(received) == 1:
                return web.json_response({"error": {"code": "unavailable"}}, status=503)
        return response

    api.app.middlewares.append(lose_first_record_reply)
    async with TestClient(TestServer(api.app)) as http:
        invitation = await invite(http, replication=True, work_records=True)
        pub = home_publisher(api, http, invitation, monkeypatch)
        for turn in range(6):
            append(api.source, f"busy-{turn}", "Continue")
            if turn == 1:
                start(api.source)
            if turn == 2:
                pub = home_publisher(api, http, invitation, monkeypatch)
            await asyncio.to_thread(pub._publish_one, KEY)
        assert len(received) >= 3 and received[0] == received[1]
        assert (received[0]["tasks"][0]["phase"], received[-1]["tasks"][0]["phase"]) == ("queued", "running")
        latest = rooms.room_state(api.source, room_id="room")["latest_seq"]
        state = replicas.copy_state(api.target, room_id="room")
        assert state["last_seq"] == latest and latest - state["work_records"]["history"]["seq"] <= 2
        assert state["work_records"]["source_loss_safe"] is False and rooms.list_rooms(api.target) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("auth", ["history_only", "bearer", "missing"])
async def test_history_only_or_broad_auth_can_never_deliver_records(api, auth):
    admit(api.source)
    async with TestClient(TestServer(api.app)) as http:
        token = (await invite(http, replication=True))["grant"]
        headers = {"history_only": {"Authorization": f"HermesRoom {token}"},
                   "bearer": {"Authorization": f"Bearer {API_KEY}"}, "missing": {}}[auth]
        record = records.capture(api.source, room_id="room", local_gateway_id=HOME)
        response = await http.post("/v1/room-members/work-records", json={"record": record}, headers=headers)
        assert response.status == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("flags", [{"work_records": "yes"}, {"work_records": 1}, {"work_records": True, "replication": False}])
async def test_the_evidence_opt_in_must_be_explicit_and_needs_the_copy(api, flags):
    async with TestClient(TestServer(api.app)) as http:
        response = await http.post("/v1/room-members/invitations", json={
            "room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME, "authority_epoch": 1,
            "member_id": "reviewer", "replication": True, **flags}, headers={"Authorization": f"Bearer {API_KEY}"})
        assert response.status == 400


@pytest.mark.asyncio
async def test_the_record_request_body_is_bounded(api, monkeypatch):
    monkeypatch.setattr(records, "MAX_BYTES", 256)
    async with TestClient(TestServer(api.app)) as http:
        token = (await invite(http, replication=True, work_records=True))["grant"]
        response = await http.post("/v1/room-members/work-records", json={"record": "x" * 2000},
                                   headers={"Authorization": f"HermesRoom {token}"})
        assert response.status in {400, 413}


@pytest.mark.asyncio
async def test_a_restart_and_a_task_only_change_reuse_the_same_pending_record(api, monkeypatch):
    admit(api.source)
    received = []

    @web.middleware
    async def lose_first_reply(request, handler):
        response = await handler(request)
        if request.path.endswith("/work-records") and response.status == 200:
            with rooms._transaction(api.target) as conn:
                received.append(json.loads(conn.execute(
                    f"SELECT record_json FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone()[0]))
            if len(received) == 1:
                request.transport.close()
        return response

    api.app.middlewares.append(lose_first_reply)
    async with TestClient(TestServer(api.app)) as http:
        invitation = await invite(http, replication=True, work_records=True)
        first = home_publisher(api, http, invitation, monkeypatch)
        await asyncio.to_thread(first._publish_one, KEY)  # history
        await asyncio.to_thread(first._publish_one, KEY)  # record: accepted, reply lost
        assert first.status()["work_records"][0]["status"] == "unavailable"
        start(api.source)
        second = home_publisher(api, http, invitation, monkeypatch)
        await asyncio.to_thread(second._publish_one, KEY)
        assert received[0] == received[1]
        await asyncio.to_thread(second._publish_one, KEY)
        assert received[2]["revision"] > received[1]["revision"] and received[2]["history"] == received[1]["history"]
        assert replicas.copy_state(api.target, room_id="room")["work_records"]["phases"] == {"running": 1}
        await asyncio.to_thread(second._publish_one, KEY)
        assert len(received) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["partial_ack", "unsupported", "route_race"])
async def test_only_the_whole_exact_ack_completes_a_record_and_route_changes_are_fenced(api, monkeypatch, mode):
    admit(api.source)
    altered = []

    @web.middleware
    async def alter(request, handler):
        if mode == "unsupported" and request.path.endswith("/work-records"):
            altered.append(True)
            from gateway.platforms.api_server_room_proof import wrap
            async def unsupported(verified):
                return web.json_response({"error": {"code": "unsupported"}}, status=404)
            return await wrap(api.adapter, unsupported)(request)
        response = await handler(request)
        if request.path.endswith("/work-records") and response.status == 200:
            altered.append(True)
            if mode == "partial_ack":
                from gateway import hosted_room_proof as proof
                from gateway.platforms.api_server_room_proof import _seal
                key = peer._split_token(request['verified_room_grant'])[1]
                envelope = json.loads(proof._b64decode(request.headers['Authorization'][len(proof.SCHEME):]))
                mac = envelope['mac']
                body = json.loads(proof.verify_response(key, mac, response.status, response.body,
                    response.headers[proof.RESPONSE_HEADER], response.headers[proof.RESPONSE_NONCE_HEADER]))
                body.pop("digest")
                return _seal(web.json_response(body), key, mac)
            if len(altered) == 1:
                stored = next(link for link in links.load_room_links(api.source) if link.member_id == "reviewer")
                links.save_room_link(api.source, replace(stored, trace_id="replacement-route"))
        return response

    api.app.middlewares.append(alter)
    async with TestClient(TestServer(api.app)) as http:
        pub = home_publisher(api, http, await invite(http, replication=True, work_records=True), monkeypatch)
        await asyncio.to_thread(pub._publish_one, KEY)
        await asyncio.to_thread(pub._publish_one, KEY)
        expected = {"partial_ack": "invalid_ack", "unsupported": "rejected", "route_race": "pending"}[mode]
        assert pub.status()["work_records"][0]["status"] == expected
        assert pub.status()["routes"][0]["status"] == "acked"
        await asyncio.to_thread(pub._publish_one, KEY)
        if mode == "route_race":
            assert pub.status()["work_records"][0]["status"] == "acked" and len(altered) == 2
        else:
            assert len(altered) == 1
        append(api.source, "later", "still copies")
        await asyncio.to_thread(pub._publish_one, KEY)
        assert replicas.copy_state(api.target, room_id="room")["last_seq"] == 2


@pytest.mark.asyncio
async def test_exact_run_receipts_and_stop_generations_are_kept_without_results(api):
    admit(api.source)
    start(api.source)
    receipt = {"room_id": "room", "home_install_id": HOME, "authority_gateway_id": HOME, "authority_epoch": 1,
               "member_id": "reviewer", "target_install_id": TARGET, "target_profile": "default", "task_id": "task",
               "execution_generation": 1, "run_id": "accepted-run", "session_id": "accepted-session"}
    rooms.upsert_remote_run_receipt(api.source, record=receipt)
    async with TestClient(TestServer(api.app)) as http:
        token = (await invite(http, replication=True, work_records=True))["grant"]
        client = http_sender(http)

        async def history():
            await asyncio.to_thread(client.replicate_page, grant=token, room_id="room", room_name="Workshop",
                                    members=MEMBERS, page=rooms.read_events(api.source, room_id="room"))

        async def deliver(record):
            return await asyncio.to_thread(client.replicate_work_records, grant=token, record=record)

        await history()
        await deliver(records.capture(api.source, room_id="room", local_gateway_id=HOME))
        assert replicas.copy_state(api.target, room_id="room")["work_records"]["receipts"] == [receipt]
        rooms.request_room_stop(api.source, room_id="room", cancel_id="stop", expected_gateway_id=HOME, expected_epoch=1)
        driver.begin_task_cancel(api.source, TASK, cancel_id="stop", expected_cancel_generation=0, clock=lambda: 100)
        await history()
        stopping = records.capture(api.source, room_id="room", local_gateway_id=HOME)
        await deliver(stopping)
        summary = replicas.copy_state(api.target, room_id="room")["work_records"]
        assert (summary["phases"], summary["tasks"][0]["cancel_generation"]) == ({"stopping": 1}, 1)
        assert (summary["stop"]["cancel_id"], summary["receipts"]) == ("stop", [receipt])
        driver.complete_task_cancel(api.source, TASK, cancel_id="stop", expected_cancel_generation=1, clock=lambda: 100)
        cancelled = records.capture(api.source, room_id="room", local_gateway_id=HOME)
        assert cancelled["history"] == stopping["history"] and cancelled["revision"] > stopping["revision"]
        await deliver(cancelled)
        assert replicas.copy_state(api.target, room_id="room")["work_records"]["phases"] == {"cancelled": 1}

"""The home's history publisher against real home and participant stores; only urllib is replaced."""

import base64
import json
import multiprocessing
import sqlite3
import threading
import time
from dataclasses import replace

import pytest

from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replication as publisher
from gateway import hosted_rooms as rooms
from tests.gateway.fixtures.passive_copy import (  # noqa: F401
    HOME, HTTP, KEY, SECRET, TARGET, add_route, append, pair, reserve, save_link)


def state(pub, index=0):
    return pub.status("room")["routes"][index]


def remove_routes(db):
    """What Disband does once each participant revoked its grant."""
    rooms.delete_room_link_records(db, room_id="room")


def _child_publish(source, target, entered, release, results):
    """A second gateway process sharing the home store: its own OS lock, its own SQLite handles."""
    from tui_gateway import hosted_room_peer_http
    rooms.local_authority_gateway_id = lambda: HOME
    publisher.PAGE_LIMIT = 1
    http = HTTP(target)

    def wait_for_parent(*_):
        entered.set()
        assert release.wait(3)

    http.before, http.lose_ack = wait_for_parent, True
    hosted_room_peer_http._open_roomlink_url = http
    pub = publisher.HostedRoomReplicationPublisher(source)
    pub._publish_one(KEY)
    results.put({"requests": http.requests, "status": pub.status()})


def test_restart_checkpoint_suppresses_unchanged_http_and_stays_passive(pair):
    first = publisher.HostedRoomReplicationPublisher(pair.source)
    assert first._publish_one(KEY) is False
    assert state(first)["acked_seq"] == 1
    second = publisher.HostedRoomReplicationPublisher(pair.source)
    for _ in range(3):
        assert second._publish_one(KEY) is False
    assert len(pair.http.requests) == 1
    append(pair.source, "later")
    second._publish_one(KEY)
    assert pair.http.requests[-1][1]["page"]["events"][0]["seq"] == 2
    assert replicas.replica_state(pair.target, room_id="room")["safety_status"] == "passive"
    assert rooms.list_rooms(pair.target) == []
    assert second.status()["source_loss_safe"] is False
    assert pair.link.grant not in repr(second.status())
    assert pair.link.grant not in repr(second._load_route(KEY))


def test_lost_reply_resends_the_identical_page_despite_new_source_events(pair):
    pair.http.lose_ack = True
    first = publisher.HostedRoomReplicationPublisher(pair.source)
    first._publish_one(KEY)
    assert (state(first)["acked_seq"], state(first)["status"]) == (0, "unavailable")
    assert replicas.replica_state(pair.target, room_id="room")["last_seq"] == 1
    append(pair.source, "after-lost-reply")
    restarted = publisher.HostedRoomReplicationPublisher(pair.source)
    restarted._publish_one(KEY)
    assert pair.http.requests[0] == pair.http.requests[1]
    restarted._publish_one(KEY)
    assert state(restarted)["acked_seq"] == 2


def test_typed_gap_resets_the_cursor_but_a_rejection_stays_blocked(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    append(pair.source, "second")
    pair.http.error = (409, "room_replica_gap")
    pub._publish_one(KEY)
    assert (state(pub)["acked_seq"], state(pub)["status"]) == (0, "replica_gap")
    pair.http.error = None
    pub._publish_one(KEY)
    assert [e["seq"] for e in pair.http.requests[-1][1]["page"]["events"]] == [1, 2]
    append(pair.source, "third")
    pair.http.error = (409, "invalid_room_replica")
    pub._publish_one(KEY)
    assert (state(pub)["acked_seq"], state(pub)["status"]) == (2, "replica_rejected")
    before = len(pair.http.requests)
    publisher.HostedRoomReplicationPublisher(pair.source)._publish_one(KEY)
    assert len(pair.http.requests) == before


@pytest.mark.parametrize("error", [(401, "invalid_room_grant"), (403, "room_reauthorization_required")])
def test_auth_rejection_stops_until_a_new_durable_grant(pair, error):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.error = error
    pub._publish_one(KEY)
    assert state(pub)["status"] == "needs_reauthorization"
    pub._publish_one(KEY)
    assert len(pair.http.requests) == 1
    reserve(pair.target, save_link(pair.source, grant_id="replacement").grant)
    pair.http.error = None
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 1


def test_link_health_changes_cannot_reactivate_the_same_rejected_grant(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.error = (401, "invalid_room_grant")
    pub._publish_one(KEY)
    links.mark_room_link_status(pair.source, room_id="room", member_id="reviewer", status="needs_reauthorization")
    pub._publish_one(KEY)
    links.mark_room_link_status(pair.source, room_id="room", member_id="reviewer", status="ready")
    pub._publish_one(KEY)
    assert len(pair.http.requests) == 1
    assert state(pub)["status"] == "needs_reauthorization"


@pytest.mark.parametrize("overrides", [
    {"permissions": ("dispatch", "status", "stop")}, {"member_id": "absent"},
    {"authority_gateway_id": "install:other"}, {"authority_epoch": 2},
    {"home_install_id": "install:other"}, {"target": "install:other"},
])
def test_ineligible_grants_never_send(pair, overrides):
    room = rooms.room_state(pair.source, room_id="room")
    pair.source = pair.source.with_name("ineligible.db")
    rooms.create_room(pair.source, room_id="room", name=room["name"], members=room["members"],
                      authority_gateway_id=HOME)
    append(pair.source, "hello")
    link = save_link(pair.source, **overrides)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    assert not pub._publish_one((link.room_id, link.member_id))
    assert pair.http.requests == []


def test_sending_hint_never_trusts_unsigned_timing(pair):
    encoded, signature = peer._split_token(pair.link.grant)
    hint = json.loads(encoded)
    hint.update(issued_at=0, expires_at=1, status_expires_at=2)
    token = base64.urlsafe_b64encode(json.dumps(hint).encode()).decode().rstrip("=")
    token += "." + base64.urlsafe_b64encode(signature).decode().rstrip("=")
    assert publisher._eligible(replace(pair.link, grant=token), rooms.room_state(pair.source, room_id="room"), HOME)


@pytest.mark.parametrize("change", ["grant", "url", "authority", "remove"])
def test_a_route_that_changes_during_http_cannot_advance(pair, change):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)

    def change_route(_request, _body):
        pair.http.before = None
        if change == "grant":
            reserve(pair.target, save_link(pair.source, grant_id="new-generation").grant)
        elif change == "url":
            links.save_room_link(pair.source, replace(pair.link, target_url="http://127.0.0.1:9877"))
        elif change == "authority":
            with sqlite3.connect(pair.source) as conn:
                conn.execute("UPDATE hosted_rooms SET authority_gateway_id='install:other' WHERE room_id='room'")
        else:
            remove_routes(pair.source)

    pair.http.before = change_route
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 0
    pub._publish_one(KEY)
    if change in {"grant", "url"}:
        assert state(pub)["acked_seq"] == 1
        assert pair.http.requests[-1][1]["page"]["events"][0]["seq"] == 1
    else:
        assert len(pair.http.requests) == 1


def test_disband_after_route_removal_reports_stopped_not_delivered(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    remove_routes(pair.source)
    rooms.disband_room(pair.source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
    pub._scan(time.monotonic())
    assert state(pub)["status"] == "stopped_route_removed"
    assert (state(pub)["acked_seq"], state(pub)["source_latest_seq"]) == (1, 2)
    assert state(pub)["delivery_unconfirmed"] is True
    assert replicas.replica_state(pair.target, room_id="room")["disbanded_at"] is None
    assert len(pair.http.requests) == 1


def test_a_live_route_delivers_the_real_terminal_event(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    rooms.disband_room(pair.source, room_id="room", expected_gateway_id=HOME, expected_epoch=1)
    pub._publish_one(KEY)
    assert pair.http.requests[-1][1]["page"]["events"][-1]["kind"] == "room.disbanded"
    assert replicas.replica_state(pair.target, room_id="room")["disbanded_at"] is not None


def test_two_workers_isolate_an_unavailable_participant_and_stop_without_thread_growth(pair):
    other = add_route(pair, target="install:independent-participant")
    links.save_room_link(pair.source, replace(other, target_url="http://127.0.0.1:9877"))
    blocked, release, healthy = threading.Event(), threading.Event(), threading.Event()

    def unavailable(request, _body):
        if ":9876/" in request.full_url:
            blocked.set()
            assert release.wait(3)
            raise TimeoutError("offline")
        healthy.set()

    pair.http.before = unavailable
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub.start()
    try:
        assert blocked.wait(3) and healthy.wait(3)
        assert pub.status()["workers"] == 2
        assert not pub.stop(timeout=0)
        original = tuple(pub._threads)
        pub.start()
        assert tuple(pub._threads) == original
    finally:
        release.set()
        assert pub.stop(timeout=5)
    assert {row["target_install_id"] for row in pub.status()["routes"]} == {
        TARGET, "install:independent-participant"}


def test_fixed_page_budget_and_round_robin_rotate_a_backlog(pair, monkeypatch):
    monkeypatch.setattr(publisher, "PAGE_LIMIT", 1)
    append(pair.source, "second")
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    assert pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 1
    assert not pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 2
    assert all(len(body["page"]["events"]) == 1 for _, body in pair.http.requests)


def test_a_participant_copy_never_becomes_publishable_authority(pair):
    publisher.HostedRoomReplicationPublisher(pair.source)._publish_one(KEY)
    links.save_room_link(pair.target, pair.link)
    assert not publisher.HostedRoomReplicationPublisher(pair.target)._publish_one(KEY)
    assert len(pair.http.requests) == 1
    assert rooms.list_rooms(pair.target) == []


def test_an_empty_page_whose_reply_was_lost_is_resent_empty(pair):
    with sqlite3.connect(pair.source) as conn:
        conn.execute("DELETE FROM hosted_room_events WHERE room_id='room'")
        conn.execute("UPDATE hosted_rooms SET next_seq=1, event_bytes=0 WHERE room_id='room'")
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.lose_ack = True
    pub._publish_one(KEY)
    append(pair.source, "first-after-empty-page")
    publisher.HostedRoomReplicationPublisher(pair.source)._publish_one(KEY)
    assert pair.http.requests[0] == pair.http.requests[1]
    pub._publish_one(KEY)
    assert state(pub)["acked_seq"] == 1


@pytest.mark.parametrize("changes", [
    {"room_id": "another"}, {"authority": {"gateway_id": "install:other", "epoch": 1}},
    {"stored_seq": True}, {"stored_seq": 999}, {"stored_seq": 0},
])
def test_a_malformed_reply_never_advances_or_retries_forever(pair, changes):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.reply_transform = lambda result: {**result, **changes}
    pub._publish_one(KEY)
    assert (state(pub)["acked_seq"], state(pub)["status"]) == (0, "invalid_ack")
    pub._publish_one(KEY)
    assert len(pair.http.requests) == 1


def test_a_quarantined_source_room_is_never_published(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    with sqlite3.connect(pair.source) as conn:
        conn.execute("""INSERT INTO hosted_room_quarantine(room_id, reason, detected_at)
            VALUES ('room', 'test-unverified-lineage', ?)""", (time.time(),))
    assert not pub._publish_one(KEY)
    assert pair.http.requests == []


def test_a_route_that_is_not_stored_is_never_published(pair):
    remove_routes(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._scan(time.monotonic())
    assert list(pub._routes) == []
    assert not pub._publish_one(KEY)
    assert pair.http.requests == []


def test_the_removed_route_diagnostic_tail_stays_bounded(pair):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    with sqlite3.connect(pair.source) as conn:
        conn.executemany(f"""INSERT INTO {publisher.ROUTES_TABLE}
            (room_id,member_id,generation,target_install_id,target_profile,
             authority_gateway_id,authority_epoch,status,updated_at)
            VALUES (?, 'gone', 'old', 'target', 'default', 'home', 1, 'pending', 0)""",
                         [(f"removed-{i}",) for i in range(links.MAX_LINKS + 10)])
    pub._scan(time.monotonic())
    assert len(pub.status()["routes"]) == links.MAX_LINKS + 1
    assert state(pub)["status"] == "acked"


def test_the_scheduler_honors_due_times_between_scans_and_rotates(pair, monkeypatch):
    other = add_route(pair, member_id="second")
    other_key = (other.room_id, other.member_id)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    now, waits = [5.0], []
    pub._scan(now[0])
    pub._due = {KEY: 5.2, other_key: 5.2}
    monkeypatch.setattr(publisher.time, "monotonic", lambda: now[0])

    def wait(*, timeout):
        waits.append(timeout)
        now[0] += timeout

    monkeypatch.setattr(pub._condition, "wait", wait)
    assert pub._take() == KEY
    assert waits == [pytest.approx(0.2)]
    pub._inflight.remove(KEY)
    assert pub._take() == other_key


def test_grants_without_the_opt_in_are_filtered_at_scan_without_room_reads(pair, monkeypatch):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    save_link(pair.source, permissions=("status", "dispatch"))
    reads = []
    monkeypatch.setattr(rooms, "room_state", lambda *a, **k: reads.append(k))
    pub._scan(time.monotonic())
    assert list(pub._routes) == []
    assert pub._load_route(KEY) is None
    assert reads == []
    remove_routes(pair.source)
    pub._scan(time.monotonic())
    assert state(pub)["status"] == "stopped_route_removed"


def test_one_participant_copy_is_shared_by_two_processes_and_a_replacement_route(pair, monkeypatch):
    other = add_route(pair)
    other_key = (other.room_id, other.member_id)
    monkeypatch.setattr(publisher, "PAGE_LIMIT", 1)
    second = publisher.HostedRoomReplicationPublisher(pair.source)
    context = multiprocessing.get_context("spawn")
    entered, release, results = context.Event(), context.Event(), context.Queue()
    child = context.Process(target=_child_publish, args=(pair.source, pair.target, entered, release, results))
    child.start()
    try:
        assert entered.wait(8)
        append(pair.source, "source-grows-during-http")
        assert not second._publish_one(other_key)
        assert pair.http.requests == []  # the other process holds this participant's lock
        release.set()
        first = results.get(timeout=8)
        child.join(8)
        assert child.exitcode == 0
    finally:
        release.set()
        if child.is_alive():
            child.terminate()
            child.join(5)
        results.close()
        results.join_thread()
    assert first["status"]["routes"][0]["status"] == "unavailable"
    second._publish_one(other_key)
    assert pair.http.requests[-1][1] == first["requests"][0][1]  # the alternate resends the same page
    rooms.rename_room(pair.source, room_id="room", event_id="rename", name="Current name")
    pair.http.lose_ack = True
    second._publish_one(other_key)
    old_pending = pair.http.requests[-1][1]
    append(pair.source, "after-second-lost-reply")
    reserve(pair.target, save_link(pair.source, grant_id="replacement-primary").grant)
    restarted = publisher.HostedRoomReplicationPublisher(pair.source)
    restarted._publish_one(KEY)
    assert pair.http.requests[-1][1] == old_pending
    for _ in range(4):
        restarted._publish_one(other_key)
    replica = replicas.replica_state(pair.target, room_id="room")
    assert replica["last_seq"] == rooms.room_state(pair.source, room_id="room")["latest_seq"] == 4
    assert replica["name"] == "Current name"
    assert replica["safety_status"] == "passive"
    rows = restarted.status()["routes"]
    assert all(row["status"] not in publisher._BLOCKED for row in rows)
    assert {row["target_acked_seq"] for row in rows} == {4}
    assert {row["selected_member_id"] for row in rows} == {KEY[1]}
    assert rows[1]["role"] == "alternate"
    before = len(pair.http.requests)
    second._publish_one(other_key)
    restarted._publish_one(KEY)
    assert len(pair.http.requests) == before


def test_a_revoked_route_hands_its_pending_page_to_an_authorized_alternate(pair):
    other = add_route(pair)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.lose_ack = True
    pub._publish_one(KEY)
    original = pair.http.requests[-1][1]
    claims = peer.decode_room_grant(SECRET, pair.link.grant, permission="replicate")
    rooms.revoke_room_grant_scope(pair.target, claims=claims, expires_at=claims["status_expires_at"])
    links.mark_room_link_status(pair.source, room_id="room", member_id="reviewer", status="needs_reauthorization")
    append(pair.source, "after-revoke")
    pub._publish_one((other.room_id, other.member_id))
    assert pair.http.requests[-1][1] == original
    pub._publish_one((other.room_id, other.member_id))
    assert replicas.replica_state(pair.target, room_id="room")["last_seq"] == 2
    assert pub.status()["routes"][1]["role"] == "selected"


@pytest.mark.parametrize("code", [401, 409])
def test_a_blocked_route_can_only_hand_over_to_another_authorized_route(pair, code):
    other = add_route(pair)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.error = (code, "invalid_room_grant" if code == 401 else "invalid_room_replica")
    pub._publish_one(KEY)
    original = pair.http.requests[-1][1]
    pair.http.error = None
    pub._publish_one((other.room_id, other.member_id))
    assert pair.http.requests[-1][1] == original
    assert pair.http.requests[-1][1]["page"]["events"][0]["seq"] == 1
    rows = pub.status()["routes"]
    assert (rows[0]["role"], rows[0]["acked_seq"]) == ("blocked", 0)
    assert (rows[1]["role"], rows[1]["target_acked_seq"]) == ("selected", 1)


def test_status_reports_an_unreadable_store_explicitly(pair, monkeypatch):
    pub = publisher.HostedRoomReplicationPublisher(pair.source)

    def fail(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(publisher, "open_sqlite", fail)
    result = pub.status()
    assert (result["error"], result["routes"], result["source_loss_safe"]) == (
        "publisher_status_unavailable", None, False)

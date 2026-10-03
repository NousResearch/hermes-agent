"""Voters, majority protection and the lease layer's hook points (automatic takeover).

The host and its always-on successors vote, at most seven, in the owner's order. Protection needs
a majority of them, counting the host. Voters change one at a time, each change stored on a
majority of both the old and the new voters before the next. Real stores; acknowledgments are
recorded exactly as the publisher records them.
"""

import time
from contextlib import closing
from types import SimpleNamespace

import pytest

from gateway import hosted_room_custody as custody
from gateway import hosted_room_driver as driver
from gateway import hosted_room_identity as identity
from gateway import hosted_room_replication as publisher
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.fixtures.passive_copy import HOME, KEY, MEMBERS, TARGET, TASK, admit, append, pair  # noqa: F401

ROOM = "room"
A, B, C, D = (f"install:{name}" for name in "abcd")
KEYS = {install_id: identity.local_public_key(secret=f"{install_id}-room-identity-secret-32b".encode())
        for install_id in (A, B, C, D, TARGET)}


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
    db = tmp_path / "home.db"
    rooms.create_room(db, room_id=ROOM, name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    append(db, "hello")
    return db


@pytest.fixture
def hooks():
    """Register lease hooks for one test, and restore the defaults after it."""
    replaced = []
    yield lambda **registered: replaced.append(custody.register_lease_hooks(**registered))
    for previous in reversed(replaced):
        custody.register_lease_hooks(**previous)


@pytest.fixture(autouse=True)
def lease_layer(monkeypatch):
    """The lease layer is installed and lets the host serve, unless a test says otherwise."""
    monkeypatch.setattr(custody, "lease_layer_installed", lambda: True)
    replaced = custody.register_lease_hooks(serving_provider=lambda room_id: True)
    yield
    custody.register_lease_hooks(**replaced)


def enroll(db, install_id, *, successor=True, always_on=True, now=None):
    custody.enroll_custodian(db, room_id=ROOM, install_id=install_id, public_key=KEYS[install_id],
                             endpoint=f"https://{install_id[8:]}.example", name=install_id[8:].title(),
                             role="custodian", active=True, allowed=successor, designated=successor,
                             always_on=always_on, now=now)


def configure(db):
    return custody.maintain_configuration(db, room_id=ROOM, local_gateway_id=HOME, public_key=identity.local_public_key(),
                                          endpoint=None, name="Home", owner_name="Dana", always_on=True)


def ack(db, install_id, seq=None):
    """``install_id`` durably holds the host's log through ``seq`` (by default all of it)."""
    with closing(open_sqlite(db)) as conn:
        seq = int(conn.execute("SELECT next_seq - 1 FROM hosted_rooms WHERE room_id=?", (ROOM,)).fetchone()[0]) \
            if seq is None else seq
        mark = {"epoch": 1, "seq": seq, "event_hash": custody.chain_hash_locked(conn, ROOM, seq, store=False)}
    assert custody.record_acknowledgment(db, room_id=ROOM, install_id=install_id, watermark=mark) == "acknowledged"
    return seq


def settle(db, *voters):
    """Configure step by step, every listed custodian acknowledging each step, until nothing changes."""
    while configure(db) is not None:
        for voter in voters:
            ack(db, voter)


def status(db):
    return custody.custody_status(db, ROOM)


def latest(db):
    return rooms.room_state(db, room_id=ROOM)["latest_seq"]


def configurations(db):
    with closing(open_sqlite(db)) as conn:
        return custody.configurations_locked(conn, ROOM)


def test_voters_are_the_host_and_its_always_on_successors_in_the_owners_order(host):
    enroll(host, B, now=1)
    enroll(host, A, now=2)
    enroll(host, C, always_on=False, now=3)  # a laptop: it may continue when asked, never by itself
    enroll(host, D, successor=False, now=4)
    settle(host, A, B)
    current = status(host)
    assert (current["voters"], current["mode"], current["automatic"]) == ([HOME, B, A], "majority", True)
    entries = {entry["install_id"]: entry for entry in current["configuration"]["custodians"]}
    assert [(entries[i]["successor"], entries[i]["always_on"], entries[i]["voter"]) for i in (C, D)] == [
        (True, False, False), (False, True, False)]
    assert {c["install_id"]: c["voter"] for c in current["custodians"]} == {A: True, B: True, C: False, D: False}
    # Every configuration changed at most one voter.
    sets = [{HOME}] + [set(configuration["voters"]) for configuration in configurations(host)]
    assert all(len(before ^ after) <= 1 for before, after in zip(sets, sets[1:]))


def test_at_most_seven_voters_in_the_owners_order(host):
    many = [f"install:v{index}" for index in range(8)]
    for index, install_id in enumerate(many):
        KEYS[install_id] = identity.local_public_key(secret=f"{install_id}-room-identity-secret-32b".encode())
        enroll(host, install_id, now=index)
    settle(host, *many)
    assert status(host)["voters"] == [HOME, *many[:6]]


def test_protection_needs_a_majority_of_voters_and_survives_one_down(host):
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    assert status(host)["mode"] == "majority"
    append(host, "m1")
    seq = append(host, "m2")["seq"]
    assert status(host)["protected_seq"] < seq
    ack(host, A)  # B is down: the host and A are a majority of three
    assert status(host)["protected_seq"] == seq
    assert custody.wait_protected(host, ROOM, seq, timeout=0)
    later = append(host, "m3")["seq"]  # A is down too: only the host holds it
    assert not custody.wait_protected(host, ROOM, later, timeout=0.05)
    assert status(host)["protected_seq"] == seq


def test_voters_change_one_at_a_time_each_stored_on_both_majorities(host):
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    # The owner replaces B with C: B leaves first, then C joins, each settled before the next.
    custody.designate_successor(host, room_id=ROOM, install_id=B, successor=False)
    enroll(host, C, now=3)
    assert configure(host)["voters"] == [HOME, A]
    assert {frozenset(voters) for voters in status(host)["voter_sets"]} == {
        frozenset({HOME, A}), frozenset({HOME, A, B})}
    assert configure(host) is None  # C waits until the change is stored on a majority of both
    ack(host, B)  # B alone is not enough: the new voters {host, A} both need it
    assert len(status(host)["voter_sets"]) == 2
    ack(host, A)
    assert status(host)["voter_sets"] == [[HOME, A]]
    assert configure(host)["voters"] == [HOME, A, C]
    ack(host, C)  # the old voters {host, A} still need A
    assert len(status(host)["voter_sets"]) == 2 and configure(host) is None
    ack(host, A)
    assert status(host)["voter_sets"] == [[HOME, A, C]]
    # Switching automatic moves off is a change like any other, and leaves the voters as they are.
    custody.set_automatic(host, room_id=ROOM, enabled=False)
    switched = configure(host)
    assert (switched["automatic"], switched["voters"], status(host)["mode"]) == (False, [HOME, A, C], "ask")
    assert len(status(host)["voter_sets"]) == 2  # stored on a majority of the voters before it counts


def test_a_stale_partitioned_voter_is_not_counted_under_a_newer_configuration(host):
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    ack(host, B)  # B's last word before it is cut off
    custody.designate_successor(host, room_id=ROOM, install_id=B, successor=False)
    dropped = configure(host)
    assert dropped["voters"] == [HOME, A]
    # Until a majority of the old voters holds the drop, B's old majority still counts: a standby
    # holding the three-voter configuration could otherwise still gather it.
    assert len(status(host)["voter_sets"]) == 2
    held = ack(host, A)
    current = status(host)
    assert (current["voter_sets"], current["mode"]) == ([[HOME, A]], "careful")
    append(host, "m1")
    append(host, "m2")
    ack(host, B)  # B is back and holds everything, but no longer votes
    assert status(host)["protected_seq"] == held < latest(host)


def test_dispatch_waits_for_copies_in_majority_mode(host):
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    admit(host, seq=1)
    announced = next(event for event in rooms.read_events(host, room_id=ROOM, limit=500)["events"]
                     if event["kind"] == "task.admitted")
    assert not custody.dispatch_ready(host, ROOM, TASK.task_id, 0)
    assert status(host)["waiting_for_copies"] == {"task_id": TASK.task_id, "seq": announced["seq"]}
    ack(host, B)
    assert custody.dispatch_ready(host, ROOM, TASK.task_id, 0)
    assert status(host)["waiting_for_copies"] is None
    assert driver.get_task(host, TASK)["status"] == "queued"


def test_a_host_that_does_not_serve_dispatches_nothing_in_any_mode(host, hooks):
    """R1 major 4: whatever the mode, a paused host starts no queued work."""
    enroll(host, A, now=1)
    settle(host, A)
    admit(host, seq=1)
    custody.set_automatic(host, room_id=ROOM, enabled=False)
    settle(host, A)
    assert status(host)["mode"] == "ask" and custody.dispatch_ready(host, ROOM, TASK.task_id, 0)
    hooks(serving_provider=lambda room_id: False)
    assert not custody.dispatch_ready(host, ROOM, TASK.task_id, 0)


def test_sends_wait_for_a_majority_only_in_majority_mode(host, hooks):
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    pub = publisher.HostedRoomReplicationPublisher(host)  # not started: protection reads the store only
    seq = append(host, "m1")["seq"]
    hooks(lease_remaining_provider=lambda room_id: 0.2)  # what is left of the host's lease bounds the wait
    started = time.monotonic()
    assert pub.protect(ROOM, seq) is False
    assert 0.15 < time.monotonic() - started < 2
    ack(host, A)
    assert pub.protect(ROOM, seq) is True
    # With only two voters (careful mode) nothing waits and nothing is reported: dispatch there doesn't
    # wait for copies, so a message offered again after a move could run its turn twice.
    custody.designate_successor(host, room_id=ROOM, install_id=B, successor=False)
    configure(host)
    ack(host, A)
    assert status(host)["mode"] == "careful"
    hooks(lease_remaining_provider=lambda room_id: 60.0)
    unheld = append(host, "m2")["seq"]
    started = time.monotonic()
    assert pub.protect(ROOM, unheld) is None and time.monotonic() - started < 0.5


def test_a_send_without_a_reachable_majority_is_unprotected_within_the_bound(host, monkeypatch):
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    monkeypatch.setattr(publisher, "SEND_PROTECTION_SECONDS", 0.3)  # no lease layer bound: the default one
    pub = publisher.HostedRoomReplicationPublisher(host)
    seq = append(host, "nobody-else-holds-this")["seq"]
    started = time.monotonic()
    assert pub.protect(ROOM, seq) is False
    assert 0.25 < time.monotonic() - started < 2


@pytest.mark.asyncio
async def test_groups_send_reports_whether_the_send_is_protected(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with SessionDB(tmp_path / "state.db") as db:
        authority = SimpleNamespace(profile_id=str(tmp_path), instance_id="owner", db=db, events={})
        connection = AuthorityConnection(authority, object(), {"user_id": "alice"})
        protected = []
        authority.hosted_room_service = SimpleNamespace(
            db_path=db.db_path, runtime=SimpleNamespace(status=lambda: {"running": True, "stopping": False}),
            authorize_room=lambda subject, room_id, create=False: None,
            send=lambda room_id, event_id, payload: {"room_id": room_id, "seq": 7, "event_id": event_id},
            replication=SimpleNamespace(protect=lambda room_id, seq: protected.append((room_id, seq)) or True))
        sent = await connection.dispatch({"id": 1, "method": "groups.send", "params": {
            "room_id": "owned", "event_id": "input", "payload": {"text": "hello"}}})
        assert (sent["result"]["protected"], protected) == (True, [("owned", 7)])
        # Outside majority mode protection reports nothing, and the reply carries no ``protected``.
        authority.hosted_room_service.replication = SimpleNamespace(protect=lambda room_id, seq: None)
        other = await connection.dispatch({"id": 2, "method": "groups.send", "params": {
            "room_id": "owned", "event_id": "input-2", "payload": {"text": "hello again"}}})
    assert other["result"]["accepted"] and "protected" not in other["result"]


def test_always_on_is_reported_unless_the_operator_says_otherwise(monkeypatch):
    import psutil
    from gateway import run
    battery = SimpleNamespace(percent=80, power_plugged=True)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(psutil, "sensors_battery", lambda: battery)
    assert custody.local_always_on(refresh=True) is False  # a laptop
    monkeypatch.setattr(psutil, "sensors_battery", lambda: None)
    assert custody.local_always_on(refresh=True) is True  # no battery
    assert custody.local_always_on() is True  # cached for a minute
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"group_chat": {"always_on": False}})
    assert custody.local_always_on(refresh=True) is False  # the operator's word wins
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"group_chat": {"always_on": True}})
    monkeypatch.setattr(psutil, "sensors_battery", lambda: battery)
    assert custody.local_always_on(refresh=True) is True

    def unreadable():
        raise RuntimeError("no battery information")

    monkeypatch.setattr(run, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(psutil, "sensors_battery", unreadable)
    assert custody.local_always_on(refresh=True) is False  # unknown is never assumed always on


def test_lease_hooks_ride_on_idle_heartbeats_to_voters(pair, monkeypatch, hooks):
    monkeypatch.setattr(custody, "local_always_on", lambda refresh=False: True)
    custody.set_local_consent(pair.target, room_id=ROOM, allowed=True)
    custody.enroll_custodian(pair.source, room_id=ROOM, install_id=TARGET, public_key=KEYS[TARGET],
                             endpoint="https://participant.example", name="Mini", role="custodian", active=True,
                             allowed=True, designated=True, always_on=True)
    assert configure(pair.source)["voters"] == [HOME, TARGET]
    calls = {"request": [], "grant": [], "ack": []}
    request = {"epoch": 1, "duration_s": 20, "sent_at": 41.5}

    def grant(room_id, epoch, authority_install_id, received):
        calls["grant"].append((room_id, epoch, authority_install_id, received))
        return {"granted_until_s": 61.5}

    hooks(lease_request_provider=lambda room_id: calls["request"].append(room_id) or request,
          lease_grant_hook=grant, lease_ack_hook=lambda *args: calls["ack"].append(args))
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)  # the copy catches up, the voter answers the lease request
    assert calls["grant"] == [(ROOM, 1, HOME, request)]
    assert calls["ack"] == [(ROOM, TARGET, {"granted_until_s": 61.5}, 41.5)]
    assert status(pair.source)["custodians"][0]["always_on"] is True
    sent = len(pair.http.requests)
    pub._publish_one(KEY)  # caught up and just heard from: nothing to send
    assert len(pair.http.requests) == sent
    pub._exchanged[(ROOM, TARGET)] -= publisher.HEARTBEAT_SECONDS  # five quiet seconds later
    pub._publish_one(KEY)
    _, body = pair.http.requests[-1]
    assert (body["page"]["events"], body["custody"]["lease_request"]) == ([], request)
    assert len(calls["grant"]) == len(calls["ack"]) == 2
    assert calls["ack"][-1] == (ROOM, TARGET, {"granted_until_s": 61.5}, 41.5)
    assert status(pair.source)["custodians"][0]["last_seen"] is not None
    # A push that asks for no lease still reaches the voter's lease layer: hearing the host is contact.
    hooks(lease_request_provider=lambda room_id: None)
    pub._exchanged[(ROOM, TARGET)] -= publisher.HEARTBEAT_SECONDS
    pub._publish_one(KEY)
    assert calls["grant"][-1] == (ROOM, 1, HOME, None)
    assert "lease_grant" not in pair.http.requests[-1][1]["custody"] and len(calls["ack"]) == 2


def test_a_custodian_that_does_not_vote_gets_no_lease_request(pair, monkeypatch, hooks):
    monkeypatch.setattr(custody, "local_always_on", lambda refresh=False: False)  # a laptop
    custody.enroll_custodian(pair.source, room_id=ROOM, install_id=TARGET, public_key=KEYS[TARGET],
                             endpoint="https://participant.example", name="Laptop", role="custodian", active=True,
                             allowed=True, designated=True, always_on=False)
    configure(pair.source)
    asked = []
    hooks(lease_request_provider=lambda room_id: asked.append(room_id) or {"epoch": 1, "sent_at": 1.0})
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    assert asked == [] and "lease_request" not in pair.http.requests[-1][1]["custody"]
    pub._exchanged[(ROOM, TARGET)] -= publisher.HEARTBEAT_SECONDS
    sent = len(pair.http.requests)
    pub._publish_one(KEY)  # no heartbeat yet: a custodian that does not vote hears from the host each minute
    assert len(pair.http.requests) == sent
    pub._exchanged[(ROOM, TARGET)] -= publisher.KEEPALIVE_SECONDS
    pub._publish_one(KEY)
    assert len(pair.http.requests) == sent + 1 and asked == []


def test_reconfiguring_after_a_move_keeps_the_voter_order_and_the_switch(host, monkeypatch):
    from gateway import hosted_room_replicas as replicas
    from gateway.hosted_room_safety import transition_proof_digest
    for index, install_id in enumerate((A, B, C)):
        enroll(host, install_id, now=index)
    settle(host, A, B, C)
    custody.set_automatic(host, room_id=ROOM, enabled=False)
    settle(host, A, B, C)
    assert status(host)["voters"] == [HOME, A, B, C]
    copy = host.parent / "b.db"
    page = rooms.read_events(host, room_id=ROOM, limit=500)
    replicas.ingest_page(copy, room_id=ROOM, room_name="Workshop", members=MEMBERS, page=page)
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: B)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: B)
    proof = {"room_id": ROOM, "from_epoch": 1, "to_epoch": 2, "successor_gateway_id": B}
    replicas.promote_replica(copy, room_id=ROOM, transition={
        "proof_kind": "certified", "proof_digest": transition_proof_digest(proof), "proof": proof})
    with rooms._transaction(copy, immediate=True) as conn:
        moved = custody.reconfigure_after_transition_locked(conn, ROOM, successor=B, previous_host=HOME)
    # The previous host stays a voter (it is always on) and a successor, right after the new host.
    assert (moved["voters"], moved["automatic"]) == ([B, HOME, A, C], False)
    entries = {entry["install_id"]: entry for entry in moved["custodians"]}
    assert (entries[HOME]["role"], entries[HOME]["successor"], entries[HOME]["voter"]) == ("custodian", True, True)
    # The new host's own next configuration keeps that order and the owner's switch.
    following = custody.maintain_configuration(copy, room_id=ROOM, local_gateway_id=B, public_key=KEYS[B],
                                               endpoint=None, name="B", owner_name="Dana", always_on=True)
    assert following is None or (following["voters"], following["automatic"]) == ([B, HOME, A, C], False)


def move_to(host, monkeypatch, successor, *, previous=HOME):
    """``successor`` continues the room on its copy and records its first configuration."""
    from gateway import hosted_room_replicas as replicas
    from gateway.hosted_room_safety import transition_proof_digest
    copy = host.parent / f"{successor[8:]}.db"
    page = rooms.read_events(host, room_id=ROOM, limit=500)
    replicas.ingest_page(copy, room_id=ROOM, room_name="Workshop", members=MEMBERS, page=page)
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: successor)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: successor)
    proof = {"room_id": ROOM, "from_epoch": 1, "to_epoch": 2, "successor_gateway_id": successor}
    replicas.promote_replica(copy, room_id=ROOM, transition={
        "proof_kind": "certified", "proof_digest": transition_proof_digest(proof), "proof": proof})
    with rooms._transaction(copy, immediate=True) as conn:
        moved = custody.reconfigure_after_transition_locked(conn, ROOM, successor=successor, previous_host=previous)
    return copy, moved


def test_a_move_keeps_majority_mode_and_lets_the_group_move_back(host, monkeypatch):
    """R1 major 3: a majority move must not drop the group to careful mode."""
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    assert (status(host)["voters"], status(host)["mode"]) == ([HOME, A, B], "majority")
    copy, moved = move_to(host, monkeypatch, A)
    assert moved["voters"] == [A, HOME, B]
    with rooms._transaction(copy, immediate=True) as conn:
        after = custody.protection_locked(conn, ROOM, A)
    assert (custody.mode_of(after["configuration"]), after["voter_sets"]) == ("majority", [[A, HOME, B]])
    entries = {entry["install_id"]: entry for entry in moved["custodians"]}
    assert entries[HOME]["successor"] and entries[HOME]["voter"]  # the owner can move it back


def test_the_new_host_copies_to_the_previous_host_and_hears_its_lease_grants(host, monkeypatch, hooks):
    """The previous host stays a voter after a move, but none of its Bots is a member route of the new
    host: the new host copies to it on the copy-only grant it gave when it stepped down (#105197), and
    its lease grants come back on those pushes."""
    import json
    from gateway import hosted_room_replicas as replicas
    from gateway import hosted_room_safety as safety
    from tests.gateway.fixtures.passive_copy import HTTP, catalog, grant, reserve
    monkeypatch.setattr(custody, "local_always_on", lambda refresh=False: True)
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    new, moved = move_to(host, monkeypatch, A)
    assert moved["voters"] == [A, HOME, B]
    append(new, "after-the-move", epoch=2, gateway=A)

    def accept(conn, event):
        payload = json.loads(event["payload_json"])
        safety.mark_verified_transition(
            conn, room_id=ROOM, from_epoch=payload["from_epoch"], to_epoch=payload["to_epoch"],
            successor_gateway_id=payload["successor_gateway_id"], proof_kind=payload["proof_kind"],
            proof_digest=payload["proof_digest"])

    # The previous host keeps a copy that follows the new host, and gave it a copy-only grant.
    previous = host.parent / "previous.db"
    token = grant(member_id=custody.CUSTODY_MEMBER_ID, target=HOME, home_install_id=A, authority_gateway_id=A,
                  authority_epoch=2, permissions=("status", "replicate"))
    reserve(previous, token)
    replicas.ingest_page(previous, room_id=ROOM, room_name="Workshop", members=MEMBERS,
                         page=rooms.read_events(new, room_id=ROOM, limit=500), _verify_transition=accept)
    custody.save_custody_route(new, room_id=ROOM, install_id=HOME, target_url="http://127.0.0.1:9876",
                               target_profile="default", grant=token, catalog=catalog(HOME).as_mapping())
    monkeypatch.setattr("tui_gateway.hosted_room_peer_http._open_roomlink_url", HTTP(previous))
    calls = {"grant": [], "ack": []}
    request = {"epoch": 2, "duration_s": 20, "sent_at": 7.5}
    hooks(lease_request_provider=lambda room_id: request,
          lease_grant_hook=lambda *args: calls["grant"].append(args) or {"granted_until_s": 27.5},
          lease_ack_hook=lambda *args: calls["ack"].append(args))
    pub = publisher.HostedRoomReplicationPublisher(new)
    pub._scan(0)
    key = (ROOM, "custody@" + HOME)
    assert key in pub._routes
    pub._publish_one(key)
    assert calls["grant"] == [(ROOM, 2, A, request)]
    assert calls["ack"] == [(ROOM, HOME, {"granted_until_s": 27.5}, 7.5)]
    assert replicas.replica_state(previous, room_id=ROOM)["last_seq"] == latest(new)
    entry = {entry["install_id"]: entry for entry in status(new)["custodians"]}[HOME]
    assert (entry["role"], entry["voter"], entry["watermark"]["seq"]) == ("custodian", True, latest(new))


def test_a_laptop_host_leaves_the_voters_as_one_change_after_a_move(host, monkeypatch):
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    while custody.maintain_configuration(host, room_id=ROOM, local_gateway_id=HOME, public_key=KEYS[A],
                                         endpoint=None, name="Laptop", owner_name="Dana", always_on=False):
        ack(host, A)
        ack(host, B)
    assert status(host)["voters"] == [HOME, A, B]  # a host always votes
    copy, moved = move_to(host, monkeypatch, A)
    entries = {entry["install_id"]: entry for entry in moved["custodians"]}
    assert moved["voters"] == [A, B] and (entries[HOME]["successor"], entries[HOME]["voter"]) == (True, False)
    with rooms._transaction(copy, immediate=True) as conn:
        pending = custody.protection_locked(conn, ROOM, A)["voter_sets"]
    assert {frozenset(voters) for voters in pending} == {frozenset({A, B}), frozenset({A, HOME, B})}


def test_a_paused_host_appends_nothing(host, hooks):
    """While the lease layer says the host is paused, nothing here appends to the room's log."""
    enroll(host, A, now=1)
    settle(host, A)
    before = latest(host)
    hooks(serving_provider=lambda room_id: False)
    enroll(host, B, now=2)
    assert configure(host) is None  # its custodians changed, but a configuration would split the history
    with pytest.raises(custody.HostPausedError):
        admit(host, seq=1)
    assert latest(host) == before and driver.list_tasks(host, room_id=ROOM) == []
    hooks(serving_provider=lambda room_id: None)  # no opinion in careful mode: still paused
    assert configure(host) is None and latest(host) == before
    hooks(serving_provider=lambda room_id: True)  # the lease layer lets it serve again
    assert configure(host)["voters"] == [HOME, A, B]
    admit(host, seq=1)
    assert [event["kind"] for event in rooms.read_events(host, room_id=ROOM, limit=500)["events"]][-1] == (
        "task.admitted")


def test_without_the_lease_layer_a_host_that_could_move_by_itself_fails_closed(host, hooks):
    """R3: no serving provider means no lease, so a host in careful or majority mode appends nothing,
    while a host whose group asks first runs as before."""
    hooks(serving_provider=None)
    enroll(host, A, now=1)
    assert configure(host)["voters"] == [HOME, A]  # from ask mode the first voter may still join
    assert status(host)["mode"] == "careful"
    before = latest(host)
    enroll(host, B, now=2)
    assert configure(host) is None
    with pytest.raises(custody.HostPausedError):
        admit(host, seq=1)
    assert not custody.dispatch_ready(host, ROOM, TASK.task_id, 0) and latest(host) == before
    hooks(serving_provider=lambda room_id: True)
    custody.set_automatic(host, room_id=ROOM, enabled=False)
    settle(host, A, B)
    hooks(serving_provider=None)
    assert status(host)["mode"] == "ask"
    admit(host, seq=1)  # a group that asks first never needed a lease


def test_where_the_lease_layer_is_not_installed_no_group_moves_by_itself(host, monkeypatch, hooks):
    """Without the lease layer's code nothing could move a group, so its host offers no automatic
    moves: the group asks first, and the host never fails closed in a mode it could never serve."""
    monkeypatch.setattr(custody, "lease_layer_installed", lambda: False)
    hooks(serving_provider=None)
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    current = status(host)
    assert (current["voters"], current["automatic"], current["mode"]) == ([HOME, A, B], False, "ask")
    admit(host, seq=1)
    assert custody.automatic_pending(host, ROOM, enabled=False) is False


def test_the_automatic_switch_is_pending_until_a_majority_stores_it(host):
    enroll(host, A, now=1)
    enroll(host, B, now=2)
    settle(host, A, B)
    assert custody.automatic_pending(host, ROOM, enabled=True) is False
    custody.set_automatic(host, room_id=ROOM, enabled=False)
    assert custody.automatic_pending(host, ROOM, enabled=False) is True  # not in a configuration yet
    assert configure(host)["automatic"] is False
    assert custody.automatic_pending(host, ROOM, enabled=False) is True  # not stored on a majority yet
    ack(host, A)
    assert custody.automatic_pending(host, ROOM, enabled=False) is False

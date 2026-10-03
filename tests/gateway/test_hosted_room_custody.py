"""Custody of a Group Chat's history: identity keys, custodians, watermarks and the tail at risk.

Real home and participant stores; only urllib is replaced (``tests/gateway/fixtures/passive_copy.py``).
"""

import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_room_custody as custody
from gateway import hosted_room_driver as driver
from gateway import hosted_room_identity as identity
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replication as publisher
from gateway import hosted_rooms as rooms
from tests.gateway.fixtures.passive_copy import (  # noqa: F401
    HOME, KEY, TARGET, add_route, admit, append, pair, start)

TARGET_KEY = identity.local_public_key(secret=b"participant-room-identity-secret-32b")
OTHER_KEY = identity.local_public_key(secret=b"another-installation-identity-secret")
DOMAIN = b"hermes.test.custody.v1"


def _home_key():
    return identity.local_public_key()


def _enroll(db, *, install_id=TARGET, key=TARGET_KEY, allowed=True, designated=True, active=True, **extra):
    return custody.enroll_custodian(db, room_id="room", install_id=install_id, public_key=key,
                                    endpoint="https://participant.example", name=extra.pop("name", "Mac mini"),
                                    operator_name=extra.pop("operator_name", "Dana"), role=extra.pop("role", "custodian"),
                                    active=active, allowed=allowed, designated=designated, **extra)


def _configure(db, **extra):
    return custody.maintain_configuration(db, room_id="room", local_gateway_id=HOME, public_key=_home_key(),
                                          endpoint=None, name=extra.get("name", "Home VPS"),
                                          owner_name=extra.get("owner_name", "Dana"))


def _events(db, kind):
    return [event for event in rooms.read_events(db, room_id="room", limit=500)["events"] if event["kind"] == kind]


def test_room_identity_keys_belong_to_the_installation_and_are_pinned(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    key, install_id = identity.local_public_key(), rooms.local_authority_gateway_id()
    # A process started in a named profile's home signs as the same installation.
    (root / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    assert (identity.local_public_key(), rooms.local_authority_gateway_id()) == (key, install_id)
    monkeypatch.setenv("HERMES_HOME", str(root))
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(root / "profiles" / "ops"):  # one multiplex gateway serving that profile
        assert (identity.local_public_key(), rooms.local_authority_gateway_id()) == (key, install_id)

    db = tmp_path / "store.db"
    rooms.create_room(db, room_id="room", name="Room", members=[{"kind": "bot", "id": "a"}],
                      authority_gateway_id=install_id)
    signature = identity.sign(DOMAIN, {"room_id": "room", "n": 1})
    assert not identity.verify("room", install_id, DOMAIN, {"room_id": "room", "n": 1}, signature, db_path=db)
    with rooms._transaction(db, immediate=True) as conn:
        assert identity.pin_locked(conn, room_id="room", install_id=install_id, public_key=key, source="test")
        assert not identity.pin_locked(conn, room_id="room", install_id=install_id, public_key=key, source="test")
        with pytest.raises(identity.RoomIdentityError, match="different room identity key"):
            identity.pin_locked(conn, room_id="room", install_id=install_id, public_key=OTHER_KEY, source="test")
    assert identity.verify("room", install_id, DOMAIN, {"room_id": "room", "n": 1}, signature, db_path=db)
    # Distinct domains, payloads and rooms never share a signature.
    assert not identity.verify("room", install_id, b"hermes.test.other.v1", {"room_id": "room", "n": 1}, signature,
                               db_path=db)
    assert not identity.verify("room", install_id, DOMAIN, {"room_id": "room", "n": 2}, signature, db_path=db)
    assert not identity.verify("other", install_id, DOMAIN, {"room_id": "room", "n": 1}, signature, db_path=db)


def test_a_single_installation_room_configures_and_announces_nothing(pair):
    assert _configure(pair.source) is None
    admit(pair.source)
    assert _events(pair.source, "custody.configured") == _events(pair.source, "task.admitted") == []
    status = custody.custody_status(pair.source, "room")
    assert (status["custodians"], status["at_risk_after_seq"], status["configuration_seq"]) == ([], 0, 0)


def test_the_configuration_lists_custodians_with_successors_only_when_allowed_and_designated(pair):
    _enroll(pair.source, allowed=True, designated=False)
    _enroll(pair.source, install_id="install:opted-out", key=OTHER_KEY, active=False)
    _enroll(pair.source, install_id="install:older", key=None)
    configured = _configure(pair.source)
    assert configured["owner_name"] == "Dana"
    assert [(c["install_id"], c["role"], c["successor"], c["name"]) for c in configured["custodians"]] == [
        (HOME, "authority", False, "Home VPS"), (TARGET, "custodian", False, "Mac mini")]
    assert _configure(pair.source) is None  # unchanged: nothing appended
    custody.designate_successor(pair.source, room_id="room", install_id=TARGET, successor=True)
    configured = _configure(pair.source)
    assert [c["successor"] for c in configured["custodians"]] == [False, True]
    assert [event["payload"]["custodians"][1]["successor"] for event in _events(pair.source, "custody.configured")] == [
        False, True]
    states = {c["install_id"]: (c["state"], c["successor"], c["allowed"], c["designated"])
              for c in custody.custody_status(pair.source, "room")["custodians"]}
    # Without a current copy neither can continue the group, whatever was allowed or designated.
    assert states == {TARGET: ("active", True, True, True), "install:opted-out": ("opted_out", False, True, True),
                      "install:older": ("unsupported", False, True, True)}
    with pytest.raises(custody.CustodyError):
        custody.parse_configuration({"custodians": [{**configured["custodians"][0], "successor": True}],
                                     "owner_name": None})


def test_display_names_are_clean_labels_and_never_identities(pair):
    _enroll(pair.source, name="Mac\x00 mini\n", operator_name="  ")
    configured = _configure(pair.source, owner_name="‮Dana")
    assert configured["custodians"][1]["name"] == "Mac mini"
    assert configured["custodians"][1]["operator_name"] is None
    assert configured["owner_name"] == "Dana"
    assert custody.display_label("x" * 500) == "x" * rooms.MAX_ACTOR_LABEL_CHARS


def test_custodians_acknowledge_with_watermarks_and_successors_bound_the_tail_at_risk(pair):
    # The participant's operator allowed it; each acknowledgment reports that consent to the host.
    custody.set_local_consent(pair.target, room_id="room", allowed=True)
    _enroll(pair.source, designated=False)
    _configure(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._publish_one(KEY)
    status = custody.custody_status(pair.source, "room")
    held = status["custodians"][0]["watermark"]
    with closing(sqlite3.connect(pair.target)) as conn:
        conn.row_factory = sqlite3.Row
        assert custody.custody_watermark_locked(conn, "room", store=False) == held
    assert held["seq"] == rooms.room_state(pair.source, room_id="room")["latest_seq"]
    # Every push carries the head the host signs for the page's end; the copy keeps it.
    vouched = custody.custody_status(pair.target, "room")["head"]
    assert (vouched["host"], vouched["epoch"], vouched["seq"], vouched["chain_hash"]) == (
        HOME, 1, held["seq"], held["event_hash"])
    assert status["head"]["seq"] == held["seq"]  # the host signs its own head at its latest event
    # Not designated: its copy holds everything, and still nothing is safe from losing this host.
    assert status["at_risk_after_seq"] == 0
    # A laptop never votes: the host alone does, so protection is only its own log (mode ask).
    assert (status["voters"], status["mode"], status["protected_seq"]) == ([HOME], "ask", held["seq"])
    custody.designate_successor(pair.source, room_id="room", install_id=TARGET, successor=True)
    _configure(pair.source)
    pub._publish_one(KEY)
    latest = rooms.room_state(pair.source, room_id="room")["latest_seq"]
    assert custody.custody_status(pair.source, "room")["at_risk_after_seq"] == latest
    append(pair.source, "only-here")
    assert custody.custody_status(pair.source, "room")["at_risk_after_seq"] == latest  # the new tail is at risk
    # The custodian holds the configuration and pins every custodian's key from it.
    copy = custody.custody_status(pair.target, "room")
    assert copy["role"] == "custodian" and copy["configuration_seq"] > 0
    with closing(sqlite3.connect(pair.target)) as conn:
        assert identity.pinned_key_locked(conn, room_id="room", install_id=HOME) == _home_key()
        assert identity.pinned_key_locked(conn, room_id="room", install_id=TARGET) == TARGET_KEY


def test_a_divergent_copy_is_never_counted_and_stops_its_route(pair):
    custody.set_local_consent(pair.target, room_id="room", allowed=True)
    _enroll(pair.source)
    _configure(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pair.http.reply_transform = lambda reply: {**reply, "watermark": {**reply["watermark"], "event_hash": "0" * 64}}
    pub._publish_one(KEY)
    entry = custody.custody_status(pair.source, "room")["custodians"][0]
    assert (entry["watermark"], entry["divergent"]) == (None, True)
    assert pub.status("room")["routes"][0]["status"] == "divergent_copy"
    assert custody.custody_status(pair.source, "room")["at_risk_after_seq"] == 0


def test_an_older_peer_without_watermarks_is_unsupported_and_never_counted(pair):
    _enroll(pair.source)
    _configure(pair.source)
    pair.http.reply_transform = lambda reply: {k: v for k, v in reply.items() if k not in {"watermark", "custody"}}
    publisher.HostedRoomReplicationPublisher(pair.source)._publish_one(KEY)
    entry = custody.custody_status(pair.source, "room")["custodians"][0]
    assert (entry["state"], entry["watermark"]) == ("unsupported", None)
    assert custody.custody_status(pair.source, "room")["at_risk_after_seq"] == 0


def test_the_chain_names_the_exact_prefix_across_checkpoints(pair):
    for index in range(300):
        append(pair.source, f"m{index}", f"message {index} é日")
    _enroll(pair.source)
    _configure(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    while pub._publish_one(KEY):
        pass
    pub._publish_one(KEY)
    with closing(sqlite3.connect(pair.source)) as home, closing(sqlite3.connect(pair.target)) as copy:
        home.row_factory = copy.row_factory = sqlite3.Row
        for seq in (1, 128, 257, int(copy.execute("SELECT last_seq FROM hosted_room_replicas").fetchone()[0])):
            assert custody.chain_hash_locked(home, "room", seq, store=False) == custody.chain_hash_locked(
                copy, "room", seq, store=False)
        assert custody.chain_hash_locked(home, "room", 5, store=False) != custody.chain_hash_locked(
            home, "room", 6, store=False)
        assert home.execute("SELECT COUNT(*) FROM hosted_room_custody_chain").fetchone()[0] >= 2


def test_admissions_are_announced_per_generation_and_dispatch_never_waits(pair):
    _enroll(pair.source)
    _configure(pair.source)
    task = admit(pair.source)
    announced = _events(pair.source, "task.admitted")
    assert [event["payload"] for event in announced] == [{
        "task": {"room_id": "room", "task_id": "task", "thread_id": "thread", "turn_id": "turn"},
        "execution_generation": 1, "target_member_id": "reviewer", "target_install_id": TARGET,
        "source_event_seq": 1}]
    assert announced[0]["actor"] == {"kind": "system", "id": "room-driver"}
    assert task["status"] == "queued"
    attempt = start(pair.source)  # no copy holds the announcement yet: the start is not held back
    driver.requeue_not_admitted_task(pair.source, attempt, clock=lambda: 101)
    assert [event["payload"]["execution_generation"] for event in _events(pair.source, "task.admitted")] == [1, 2]
    assert admit(pair.source)["idempotent"]  # a replayed admission announces nothing new
    assert len(_events(pair.source, "task.admitted")) == 2


def test_consent_reaches_the_host_and_is_confirmed_by_its_report(pair):
    _enroll(pair.source, allowed=False)
    _configure(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    changed = custody.set_local_consent(pair.target, room_id="room", allowed=True)
    assert (changed["allowed"], changed["confirmed"]) == (True, False)
    assert custody.local_consent(pair.target, "room") is True
    pub._publish_one(KEY)
    assert custody.custody_status(pair.source, "room")["custodians"][0]["allowed"] is True
    append(pair.source, "next")
    pub._publish_one(KEY)
    assert custody.set_local_consent(pair.target, room_id="room", allowed=True)["confirmed"] is True
    withdrawn = custody.set_local_consent(pair.target, room_id="room", allowed=False)
    assert (withdrawn["allowed"], withdrawn["confirmed"], custody.local_consent(pair.target, "room")) == (
        False, False, False)


def test_a_custodian_only_installation_keeps_the_history_without_a_bot(pair):
    from tests.gateway.fixtures.passive_copy import catalog, grant, reserve
    token = grant(member_id=custody.CUSTODY_MEMBER_ID, permissions=("status", "replicate", "successor"))
    reserve(pair.target, token)
    with sqlite3.connect(pair.source) as conn:
        conn.execute("DELETE FROM hosted_room_links")  # no member route reaches it: it has no Bot in the room
    custody.save_custody_route(pair.source, room_id="room", install_id=TARGET, target_url="http://127.0.0.1:9876",
                               target_profile="default", grant=token, catalog=catalog().as_mapping())
    _enroll(pair.source, role="custodian_only")
    _configure(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._scan(0)
    assert ("room", "custody@" + TARGET) in pub._routes
    pub._publish_one(("room", "custody@" + TARGET))
    latest = rooms.room_state(pair.source, room_id="room")["latest_seq"]
    assert replicas.replica_state(pair.target, room_id="room")["last_seq"] == latest
    entry = custody.custody_status(pair.source, "room")["custodians"][0]
    assert (entry["role"], entry["watermark"]["seq"]) == ("custodian_only", latest)
    # Removing it stops the copy and drops it from the configuration.
    assert custody.remove_custody_route(pair.source, room_id="room", install_id=TARGET)
    assert [c["install_id"] for c in _configure(pair.source)["custodians"]] == [HOME]
    with pytest.raises(custody.CustodyError, match="not a custodian-only"):
        custody.remove_custody_route(pair.source, room_id="room", install_id=TARGET)


def test_a_custodian_no_member_route_reaches_keeps_its_copy_on_a_copy_only_grant(pair):
    """Like the previous host after a move: none of its Bots is a member route of this host, so the host
    copies the history there on the copy-only grant it holds; a member route, while one reaches it,
    carries the copy instead."""
    from tests.gateway.fixtures.passive_copy import catalog, grant, reserve
    token = grant(member_id=custody.CUSTODY_MEMBER_ID, permissions=("status", "replicate"))
    reserve(pair.target, token)
    custody.save_custody_route(pair.source, room_id="room", install_id=TARGET, target_url="http://127.0.0.1:9876",
                               target_profile="default", grant=token, catalog=catalog().as_mapping())
    _enroll(pair.source)  # a member installation's custodian row
    _configure(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    custody_key = ("room", "custody@" + TARGET)
    pub._scan(0)
    assert KEY in pub._routes and custody_key not in pub._routes
    assert pub._load_route(custody_key) is None
    with sqlite3.connect(pair.source) as conn:
        conn.execute("DELETE FROM hosted_room_links")  # no member route reaches it any more
    pub._scan(0)
    assert custody_key in pub._routes
    pub._publish_one(custody_key)
    latest = rooms.room_state(pair.source, room_id="room")["latest_seq"]
    assert replicas.replica_state(pair.target, room_id="room")["last_seq"] == latest
    entry = custody.custody_status(pair.source, "room")["custodians"][0]
    assert (entry["install_id"], entry["role"], entry["watermark"]["seq"]) == (TARGET, "custodian", latest)


DAY = 24 * 60 * 60


def _copy_only_route(pair, monkeypatch):
    """A custodian-only installation on a 30-day copy-only grant, and a clock the test moves."""
    import time
    from tests.gateway.fixtures.passive_copy import catalog, grant, reserve
    clock = [time.time()]
    monkeypatch.setattr(time, "time", lambda: clock[0])
    token = grant(member_id=custody.CUSTODY_MEMBER_ID, permissions=("status", "replicate"), now=clock[0],
                  ttl_seconds=DAY, status_ttl_seconds=30 * DAY)
    reserve(pair.target, token, now=clock[0])
    with sqlite3.connect(pair.source) as conn:
        conn.execute("DELETE FROM hosted_room_links")
    custody.save_custody_route(pair.source, room_id="room", install_id=TARGET, target_url="http://127.0.0.1:9876",
                               target_profile="default", grant=token, catalog=catalog().as_mapping())
    _enroll(pair.source, role="custodian_only")
    _configure(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    pub._scan(0)
    return pub, clock, token


def _held_grant(pair):
    with closing(sqlite3.connect(pair.source)) as conn:
        return conn.execute(f"SELECT grant FROM {custody.ROUTES_TABLE}").fetchone()[0]


def _copied(pair):
    return replicas.replica_state(pair.target, room_id="room")["last_seq"] == rooms.room_state(
        pair.source, room_id="room")["latest_seq"]


def test_a_copy_only_grant_is_renewed_through_acknowledgements_past_its_horizon(pair, monkeypatch):
    """The custodian hands back a fresh copy-only grant once less than a week of it is left, so the host
    keeps copying there long past the first grant's 30 days."""
    from gateway.hosted_room_peer import unverified_room_grant_claims
    pub, clock, token = _copy_only_route(pair, monkeypatch)
    key, renewed_on = ("room", "custody@" + TARGET), []
    for day in range(1, 61):  # a message a day for two months
        clock[0] += DAY
        append(pair.source, f"day-{day}")
        held = _held_grant(pair)
        while pub._publish_one(key):
            pass
        assert _copied(pair), day
        if _held_grant(pair) != held:
            renewed_on.append(day)
            old, new = (unverified_room_grant_claims(grant) for grant in (held, _held_grant(pair)))
            assert 0 < old["status_expires_at"] - clock[0] < 7 * DAY  # only within its last week
            assert new["status_expires_at"] - new["issued_at"] == 30 * DAY
            assert {k: v for k, v in new.items() if k not in {"grant_id", "issued_at", "expires_at",
                                                              "status_expires_at"}} == {
                k: v for k, v in old.items() if k not in {"grant_id", "issued_at", "expires_at", "status_expires_at"}}
    assert renewed_on == [23, 46] and _held_grant(pair) != token


def test_a_quiet_group_renews_copy_only_grants_on_its_keepalives(pair, monkeypatch):
    pub, clock, token = _copy_only_route(pair, monkeypatch)
    key = ("room", "custody@" + TARGET)
    pub._publish_one(key)
    clock[0] += 24 * DAY  # nothing written since, and less than a week of the grant left
    pub._exchanged[("room", TARGET)] -= publisher.KEEPALIVE_SECONDS
    sent = len(pair.http.requests)
    pub._publish_one(key)
    assert len(pair.http.requests) == sent + 1 and pair.http.requests[-1][1]["page"]["events"] == []
    assert _held_grant(pair) != token


def test_a_refused_copy_only_route_resumes_on_a_renewed_grant_or_a_later_probe(pair, monkeypatch):
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
    from tests.gateway.fixtures.passive_copy import grant, reserve
    pub, clock, token = _copy_only_route(pair, monkeypatch)
    key = ("room", "custody@" + TARGET)
    pub._publish_one(key)
    assert _copied(pair)
    pair.http.error = (401, "invalid_room_grant")  # the custodian refuses the grant for now
    append(pair.source, "refused")
    assert not pub._publish_one(key) and not _copied(pair)
    pair.http.error = None
    sent = len(pair.http.requests)
    pub._publish_one(key)
    assert len(pair.http.requests) == sent  # blocked: nothing is sent until a probe says otherwise
    probes = []
    monkeypatch.setattr(PeerRunsHTTPClient, "probe", lambda self, *, grant: probes.append(grant) or {})
    clock[0] += publisher.REAUTHORIZATION_PROBE_SECONDS - 60
    pub._publish_one(key)
    assert probes == [] and not _copied(pair)
    clock[0] += 60  # an hour after the refusal the grant is probed, accepted, and copying resumes
    pub._publish_one(key)
    assert probes == [token] and _copied(pair)
    # Refused again, then a fresh grant arrives (in any acknowledgment or report): copying resumes at once.
    pair.http.error = (403, "invalid_room_grant")
    append(pair.source, "refused-again")
    pub._publish_one(key)
    pair.http.error = None

    def failing_probe(self, *, grant):
        raise PeerRunsHTTPError("refused", status_code=401)

    monkeypatch.setattr(PeerRunsHTTPClient, "probe", failing_probe)
    fresh = grant(member_id=custody.CUSTODY_MEMBER_ID, permissions=("status", "replicate"), now=clock[0],
                  grant_id="grant-fresh", ttl_seconds=DAY, status_ttl_seconds=30 * DAY)
    reserve(pair.target, fresh, now=clock[0])
    route = pub._load_route(key)
    pub._keep_renewed_grant(route, fresh)
    pub._publish_one(key)
    assert _held_grant(pair) == fresh and _copied(pair)


def test_a_custodian_only_grant_never_runs_work_and_needs_no_roster_member(pair):
    from gateway.hosted_room_peer import HostedRoomGrantError
    from gateway.hosted_room_replica_ingress import authorize_granted_room
    from tests.gateway.fixtures.passive_copy import MEMBERS, SECRET, grant, reserve
    page = rooms.read_events(pair.source, room_id="room")
    for permissions, allowed in ((("status", "replicate"), True), (("dispatch", "status", "replicate"), False)):
        token = grant(member_id=custody.CUSTODY_MEMBER_ID, permissions=permissions)
        reserve(pair.target, token)
        check = lambda: authorize_granted_room(  # noqa: E731
            token=token, secret=SECRET, target_install_id=TARGET, target_profile="default", room_id="room",
            members=MEMBERS, authority=page["authority"], permission="replicate")
        if allowed:
            assert callable(check())
        else:
            with pytest.raises(HostedRoomGrantError, match="never runs work"):
                check()


def test_the_new_host_reconfigures_custody_without_losing_a_custodian(pair, monkeypatch):
    custody.set_local_consent(pair.target, room_id="room", allowed=True)
    _enroll(pair.source)
    _enroll(pair.source, install_id="install:other", key=OTHER_KEY, name="Laptop", designated=False)
    _configure(pair.source)
    publisher.HostedRoomReplicationPublisher(pair.source)._publish_one(KEY)
    # The participant continues the group: its copy becomes the room, at a verified later epoch.
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: TARGET)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: TARGET)
    proof = {"room_id": "room", "from_epoch": 1, "to_epoch": 2, "successor_gateway_id": TARGET}
    from gateway.hosted_room_safety import transition_proof_digest
    replicas.promote_replica(pair.target, room_id="room", transition={
        "proof_kind": "attested", "proof_digest": transition_proof_digest(proof), "proof": proof})
    with rooms._transaction(pair.target, immediate=True) as conn:
        with pytest.raises(custody.CustodyError, match="previous host"):
            custody.reconfigure_after_transition_locked(conn, "room", successor=TARGET, previous_host="install:other")
        with pytest.raises(custody.CustodyError, match="keeps no copy"):
            custody.reconfigure_after_transition_locked(conn, "room", successor="install:nobody", previous_host=HOME)
        moved = custody.reconfigure_after_transition_locked(conn, "room", successor=TARGET, previous_host=HOME)
    # The previous host may continue the group again: the owner can move it back.
    assert [(c["install_id"], c["role"], c["successor"]) for c in moved["custodians"]] == [
        (HOME, "custodian", True), ("install:other", "custodian", False), (TARGET, "authority", False)]
    assert moved["owner_name"] == "Dana"
    # The new host's own recompute keeps every custodian its records never enrolled.
    following = custody.maintain_configuration(
        pair.target, room_id="room", local_gateway_id=TARGET, public_key=TARGET_KEY, endpoint=None,
        name="Mac mini", owner_name="Dana")
    assert following is None or [c["install_id"] for c in following["custodians"]] == [HOME, "install:other", TARGET]
    status = custody.custody_status(pair.target, "room")
    assert status["role"] == "authority"
    assert {c["install_id"]: c["state"] for c in status["custodians"]} == {HOME: "active", "install:other": "active"}


def test_an_unaddressed_installation_with_two_bots_keeps_one_exact_full_copy(pair):
    """Nobody addresses its Bots, the history spans several pages, and it still holds every event."""
    add_route(pair)  # a second Bot on the same participant installation
    for index in range(70):
        append(pair.source, f"m{index}")
    _enroll(pair.source)
    _configure(pair.source)
    pub = publisher.HostedRoomReplicationPublisher(pair.source)
    while pub._publish_one(KEY) or pub._publish_one(("room", "z-other")):
        pass
    status = custody.custody_status(pair.source, "room")
    assert [custodian["install_id"] for custodian in status["custodians"]] == [TARGET]  # one copy
    marks = []
    for db in (pair.target, pair.source):
        with closing(sqlite3.connect(db)) as conn:
            conn.row_factory = sqlite3.Row
            marks.append(custody.custody_watermark_locked(conn, "room", store=False))
    assert marks[0] == marks[1] == status["custodians"][0]["watermark"]
    assert marks[0]["seq"] == rooms.room_state(pair.source, room_id="room")["latest_seq"] > publisher.PAGE_LIMIT

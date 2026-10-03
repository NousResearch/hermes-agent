"""Verified-transition marks: an authority change is accepted only with its own mark.

Exclusive-authority recovery verifies its proof (the owner's attested decision to continue, a majority
certificate, the old host's signed handover, or the successor's signed evidence of the host's silence),
then marks the transition in the transaction that makes it. These tests drive the storage primitives and raw SQL writers; they do not certify any recovery
protocol.
"""

from contextlib import closing
import copy
import json
import sqlite3

import pytest

from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_safety as safety
from gateway import hosted_rooms as rooms

USER = {"kind": "user", "id": "tek"}
MEMBERS = [{"kind": "bot", "id": "planner"}]
AUTH_A = "install:" + "a" * 32
AUTH_B = "install:" + "b" * 32
AUTH_C = "install:" + "c" * 32
SYSTEM = json.dumps({"kind": "system", "id": "authority-control"}, separators=(",", ":"), sort_keys=True)


def _proof(room_id="room-1", from_epoch=1, to_epoch=2, successor=AUTH_B, **extra):
    return {"room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch,
            "successor_gateway_id": successor, "configuration_seq": 1, "promises": [], **extra}


def _handover(room_id="room-1", from_epoch=1, to_epoch=2, successor=AUTH_B, last_seq=3, **changes):
    """The old authority's handover: its statement and signature (the caller verifies the signature)."""
    statement = {"room_id": room_id, "from_epoch": from_epoch, "to_epoch": to_epoch, "successor": successor,
                 "last_seq": last_seq, "last_hash": "a" * 64, **changes}
    return {"statement": statement, "signature": "ed25519-v1." + "S" * 86}


def _evidence(**changes):
    """The successor's signed evidence: the handover fields, and how long the old host was silent."""
    return _handover(**{"silent_since": 1700000000.5, "silent_for_s": 180, **changes})


STATEMENTS = {"handover": _handover, "evidence": _evidence}


def _transition(proof=None, kind="certified"):
    proof = _proof() if proof is None else proof
    return {"proof_kind": kind, "proof_digest": safety.transition_proof_digest(proof), "proof": proof}


def _copy(tmp_path, *, room_id="room-1", n_events=3):
    """A passive copy of an epoch-1 room held by authority A, on this (B's) store."""
    source = tmp_path / f"{room_id}-authority.db"
    rooms.create_room(source, room_id=room_id, name="Field Room", members=MEMBERS, authority_gateway_id=AUTH_A)
    for index in range(n_events):
        rooms.append_event(
            source, room_id=room_id, event_id=f"{room_id}-e{index}", kind="message.user", actor=USER,
            payload={"text": f"msg {index}"}, authority_gateway_id=AUTH_A, authority_epoch=1)
    page = rooms.read_events(source, room_id=room_id, since_seq=0, limit=100)
    db = tmp_path / "store.db"
    replicas.ingest_page(db, room_id=room_id, room_name="Field Room", members=MEMBERS, page=page)
    return db, page


def _hosted(tmp_path, *, room_id="room-1", n_events=2):
    db = tmp_path / "store.db"
    rooms.create_room(db, room_id=room_id, name="Field Room", members=MEMBERS, authority_gateway_id=AUTH_A)
    for index in range(n_events):
        rooms.append_event(
            db, room_id=room_id, event_id=f"{room_id}-e{index}", kind="message.user", actor=USER,
            payload={"text": f"msg {index}"}, authority_gateway_id=AUTH_A, authority_epoch=1)
    return db


def _transition_payload(*, from_epoch=1, to_epoch=2, successor=AUTH_B, kind="certified", digest=None, proof=None):
    proof = _proof(from_epoch=from_epoch, to_epoch=to_epoch, successor=successor) if proof is None else proof
    return {"from_epoch": from_epoch, "to_epoch": to_epoch, "successor_gateway_id": successor,
            "proof_kind": kind, "proof_digest": digest or safety.transition_proof_digest(proof), "proof": proof}


def _insert_transition(conn, room_id, seq, *, epoch=2, payload=None, event_id=None, table="hosted_room_events"):
    payload = _transition_payload() if payload is None else payload
    conn.execute(
        f"""INSERT INTO {table} (room_id, seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at)
            VALUES (?, ?, ?, 'authority.transition', ?, ?, ?, 5)""",
        (room_id, seq, event_id or f"system:authority-transition:{epoch}", SYSTEM, epoch,
         json.dumps(payload, separators=(",", ":"), sort_keys=True)))


def _mark(conn, room_id="room-1", *, from_epoch=1, to_epoch=2, successor=AUTH_B, kind="certified", digest=None):
    safety.mark_verified_transition(
        conn, room_id=room_id, from_epoch=from_epoch, to_epoch=to_epoch, successor_gateway_id=successor,
        proof_kind=kind, proof_digest=digest or safety.transition_proof_digest(_proof(room_id=room_id)))


def _marks(db):
    with closing(sqlite3.connect(db)) as conn:
        return (conn.execute("SELECT room_id, from_epoch, to_epoch, successor_gateway_id, proof_kind "
                             "FROM hosted_room_verified_transitions ORDER BY room_id, to_epoch").fetchall(),
                conn.execute("SELECT room_id, to_epoch, seq, event_id FROM hosted_room_verified_transition_uses "
                             "ORDER BY room_id, to_epoch").fetchall())


def _quarantine(db):
    with closing(sqlite3.connect(db)) as conn:
        return dict(conn.execute("SELECT room_id, reason FROM hosted_room_quarantine").fetchall())


def test_marked_promotion_continues_the_room_and_keeps_it_writable(tmp_path, monkeypatch):
    db, page = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)

    promoted = replicas.promote_replica(db, room_id="room-1", transition=_transition())
    assert (promoted["authority_gateway_id"], promoted["authority_epoch"], promoted["claim_seq"]) == (AUTH_B, 2, 4)

    replay = rooms.read_events(db, room_id="room-1", since_seq=0, limit=100)
    # Original ids, actors and sequence continue across the transition.
    assert [(e["seq"], e["event_id"], e["actor"]) for e in replay["events"][:3]] == [
        (e["seq"], e["event_id"], e["actor"]) for e in page["events"]]
    transition = replay["events"][-1]
    assert transition["kind"] == "authority.transition"
    assert transition["authority_epoch"] == 2
    assert transition["payload"] == {
        "from_epoch": 1, "to_epoch": 2, "successor_gateway_id": AUTH_B, **_transition()}
    assert _quarantine(db) == {}
    assert _marks(db) == ([("room-1", 1, 2, AUTH_B, "certified")], [("room-1", 2, 4, "system:authority-transition:2")])

    appended = rooms.append_event(
        db, room_id="room-1", event_id="after", kind="message.user", actor=USER, payload={"text": "continuing"},
        authority_gateway_id=AUTH_B, authority_epoch=2)
    assert appended["seq"] == 5


def test_a_transition_notice_is_shown_and_never_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    for name, notice in (("first", "This group continues on its new host."), ("second", "Another wording.")):
        (tmp_path / name).mkdir()
        db, _ = _copy(tmp_path / name)
        # The same proof under different notices: the mark matches either way.
        replicas.promote_replica(db, room_id="room-1", transition=_transition(), text=notice)
        transition = rooms.read_events(db, room_id="room-1")["events"][-1]
        assert (transition["payload"]["text"], transition["payload"]["proof_digest"]) == (
            notice, safety.transition_proof_digest(_proof()))
        assert _quarantine(db) == {}
        assert _marks(db)[0] == [("room-1", 1, 2, AUTH_B, "certified")]
    # The triggers match a marked event on its proof alone, whatever its notice says.
    (tmp_path / "raw").mkdir()
    hosted = _hosted(tmp_path / "raw")
    with rooms._transaction(hosted, immediate=True) as conn:
        _mark(conn)
        _insert_transition(conn, "room-1", 3, payload={**_transition_payload(), "text": "edited notice"})
    assert _quarantine(hosted) == {}
    (tmp_path / "refused").mkdir()
    db, _ = _copy(tmp_path / "refused")
    with pytest.raises(replicas.ReplicaError, match="accompanies a verified transition"):
        replicas.promote_replica(db, room_id="room-1", text="no proof")
    for refused in ("", "   ", 7, "x" * (64 * 1024 + 1)):
        with pytest.raises(replicas.ReplicaError, match="text"):
            replicas.promote_replica(db, room_id="room-1", transition=_transition(), text=refused)
    assert _marks(db) == ([], [])


def test_transition_display_names_are_shown_and_never_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    db, _ = _copy(tmp_path)
    display = {"from_name": "Mac mini", "to_name": "Home VPS", "offline_since": 1700000000.5, "reason": "automatic",
               "at_risk": 0}
    replicas.promote_replica(db, room_id="room-1", transition=_transition(), text="Continues here.", display=display)
    payload = rooms.read_events(db, room_id="room-1")["events"][-1]["payload"]
    assert {key: payload[key] for key in (*display, "text")} == {**display, "text": "Continues here."}
    assert payload["proof_digest"] == safety.transition_proof_digest(_proof()) and _quarantine(db) == {}
    (tmp_path / "refused").mkdir()
    db, _ = _copy(tmp_path / "refused")
    with pytest.raises(replicas.ReplicaError, match="display accompanies"):
        replicas.promote_replica(db, room_id="room-1", display={"to_name": "Home VPS"})
    for refused in ({"to_name": "Home\nVPS"}, {"to_name": "x" * 201}, {"offline_since": -1},
                    {"offline_since": "yesterday"}, {"host": "Mac mini"}, ["Mac mini"], {"reason": "vote"},
                    {"at_risk": -1}, {"at_risk": True}):
        with pytest.raises(replicas.ReplicaError, match="display"):
            replicas.promote_replica(db, room_id="room-1", transition=_transition(), display=refused)
    assert _marks(db) == ([], [])


def test_unmarked_promotion_and_transition_stay_quarantined(tmp_path, monkeypatch):
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    replicas.promote_replica(db, room_id="room-1")
    assert _quarantine(db) == {"room-1": "unsafe_replica_promotion"}

    (tmp_path / "other").mkdir()
    other = _hosted(tmp_path / "other")
    with rooms._transaction(other, immediate=True) as conn:
        _insert_transition(conn, "room-1", 3)
    assert _quarantine(other) == {"room-1": "unverified_authority_transition"}
    assert _marks(other) == ([], [])
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.room_state(other, room_id="room-1")


@pytest.mark.parametrize("forged", ["epoch", "successor", "digest", "kind"])
def test_a_forged_mark_refuses_the_whole_transition(tmp_path, forged):
    db = _hosted(tmp_path)
    marked = {"from_epoch": 2, "to_epoch": 3} if forged == "epoch" else {}
    if forged == "successor":
        marked["successor"] = AUTH_C
    if forged == "digest":
        marked["digest"] = "0" * 64
    if forged == "kind":
        marked["kind"] = "attested"
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with rooms._transaction(db, immediate=True) as conn:
            _mark(conn, **marked)
            _insert_transition(conn, "room-1", 3)
    # Nothing of the transaction survives: no event, no mark, no quarantine.
    assert [e["kind"] for e in rooms.read_events(db, room_id="room-1")["events"]] == ["message.user"] * 2
    assert _marks(db) == ([], [])
    assert _quarantine(db) == {}


def test_a_replayed_mark_does_not_verify_a_second_transition(tmp_path, monkeypatch):
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    replicas.promote_replica(db, room_id="room-1", transition=_transition())
    with rooms._transaction(db, immediate=True) as conn:
        with pytest.raises(rooms.VerifiedTransitionError, match="already has a verified transition"):
            _mark(conn)
    # The same proof, replayed later in the log, is an unverified change.
    with rooms._transaction(db, immediate=True) as conn:
        _insert_transition(conn, "room-1", 5, event_id="replayed")
    assert _quarantine(db) == {"room-1": "unverified_authority_transition"}
    assert _marks(db)[1] == [("room-1", 2, 4, "system:authority-transition:2")]


def test_a_mark_never_carries_over_to_another_room(tmp_path, monkeypatch):
    db = _hosted(tmp_path)
    rooms.create_room(db, room_id="room-2", name="Other", members=MEMBERS, authority_gateway_id=AUTH_A)
    rooms.append_event(
        db, room_id="room-2", event_id="room-2-e0", kind="message.user", actor=USER, payload={"text": "x"},
        authority_gateway_id=AUTH_A, authority_epoch=1)
    # In one transaction: the mark names room-1, the transition is room-2's. Room-1's mark is
    # unused, so the commit fails and room-2 is not changed either.
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with rooms._transaction(db, immediate=True) as conn:
            _mark(conn, "room-1")
            _insert_transition(conn, "room-2", 2)
    assert _marks(db) == ([], []) and _quarantine(db) == {}
    # Room-1's transition, marked and used, leaves room-2's identical transition unverified.
    with rooms._transaction(db, immediate=True) as conn:
        _mark(conn, "room-1")
        _insert_transition(conn, "room-1", 3)
    with rooms._transaction(db, immediate=True) as conn:
        _insert_transition(conn, "room-2", 2)
    assert _quarantine(db) == {"room-2": "unverified_authority_transition"}


def test_a_mark_is_never_usable_outside_its_transaction(tmp_path):
    db = _hosted(tmp_path)
    # A mark alone cannot commit.
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        with rooms._transaction(db, immediate=True) as conn:
            _mark(conn)
    # A mark rolled back with its transaction leaves nothing a later transition could use.
    with pytest.raises(RuntimeError):
        with rooms._transaction(db, immediate=True) as conn:
            _mark(conn)
            raise RuntimeError("abandon")
    with rooms._transaction(db, immediate=True) as conn:
        _insert_transition(conn, "room-1", 3)
    assert _marks(db) == ([], [])
    assert _quarantine(db) == {"room-1": "unverified_authority_transition"}


def test_a_mark_needs_an_open_transaction_with_foreign_keys(tmp_path):
    db = _hosted(tmp_path)
    with closing(sqlite3.connect(db)) as conn:  # no explicit transaction, no foreign keys
        with pytest.raises(rooms.VerifiedTransitionError, match="inside the transaction"):
            _mark(conn)
        conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(rooms.VerifiedTransitionError, match="foreign key"):
            _mark(conn)
        conn.rollback()
    with rooms._transaction(db, immediate=True) as conn:
        with pytest.raises(rooms.VerifiedTransitionError, match="later epoch"):
            _mark(conn, from_epoch=2, to_epoch=2)
        with pytest.raises(rooms.VerifiedTransitionError, match="later epoch"):
            _mark(conn, from_epoch=3, to_epoch=2)
        with pytest.raises(rooms.VerifiedTransitionError, match="proof_kind"):
            _mark(conn, kind="operator")
        with pytest.raises(rooms.VerifiedTransitionError, match="proof_digest"):
            _mark(conn, digest="not-a-digest")
    assert _marks(db) == ([], [])


def test_the_transition_proof_must_bind_its_digest_and_room(tmp_path, monkeypatch):
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    forged = {**_transition(), "proof_digest": "0" * 64}
    with pytest.raises(replicas.ReplicaError, match="does not match its proof"):
        replicas.promote_replica(db, room_id="room-1", transition=forged)
    with pytest.raises(replicas.ReplicaError, match="another room or epoch"):
        replicas.promote_replica(db, room_id="room-1", transition=_transition(_proof(room_id="room-2")))
    with pytest.raises(replicas.ReplicaError, match="another room or epoch"):
        replicas.promote_replica(db, room_id="room-1", transition=_transition(_proof(successor=AUTH_C)))
    assert replicas.replica_state(db, room_id="room-1")["authority"] == {"gateway_id": AUTH_A, "epoch": 1}
    assert _marks(db) == ([], [])


def test_a_marked_demotion_fences_without_quarantine(tmp_path, monkeypatch):
    db = _hosted(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    proof = _proof(successor=AUTH_B)
    result = replicas.demote_room(db, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2,
                                  transition=_transition(proof), text="This host handed the group on.")
    assert (result["authority_gateway_id"], result["authority_epoch"]) == (AUTH_B, 2)
    lost = rooms.read_events(db, room_id="room-1")["events"][-1]
    assert lost["kind"] == "authority.lost"
    assert lost["payload"]["proof_digest"] == safety.transition_proof_digest(proof)
    assert lost["payload"]["text"] == "This host handed the group on."
    assert _quarantine(db) == {}
    # The old authority is fenced: it can no longer append at its stale epoch.
    with pytest.raises(rooms.AuthorityConflictError):
        rooms.append_event(
            db, room_id="room-1", event_id="stale", kind="message.user", actor=USER, payload={"text": "x"},
            authority_gateway_id=AUTH_A, authority_epoch=1)


@pytest.mark.parametrize("kind", sorted(safety.PROOF_KINDS))
def test_each_proof_kind_continues_the_room_with_its_mark(tmp_path, monkeypatch, kind):
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    proof = STATEMENTS[kind]() if kind in STATEMENTS else _proof()
    replicas.promote_replica(db, room_id="room-1", transition=_transition(proof, kind=kind))
    transition = rooms.read_events(db, room_id="room-1")["events"][-1]
    assert (transition["seq"], transition["payload"]["proof_kind"], transition["payload"]["proof"]) == (4, kind, proof)
    assert _marks(db)[0] == [("room-1", 1, 2, AUTH_B, kind)] and _quarantine(db) == {}


@pytest.mark.parametrize("kind", sorted(STATEMENTS))
@pytest.mark.parametrize("replayed, error", [
    ({"room_id": "room-2"}, "another room, epoch or successor"),  # another group's handover
    ({"from_epoch": 2, "to_epoch": 3}, "another room, epoch or successor"),  # a later handover of this group
    ({"successor": AUTH_C}, "another room, epoch or successor"),  # handed to another computer
    ({"last_seq": 2}, "another last event"),  # an older handover: this copy holds more than it names
    ({"last_seq": 4}, "another last event"),  # this copy has not caught up with what was handed over
])
def test_a_replayed_signed_statement_is_refused(tmp_path, monkeypatch, kind, replayed, error):
    """A handover or evidence moves exactly this history, in this group, from this epoch, to this computer."""
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    with pytest.raises(replicas.ReplicaError, match=error):
        replicas.promote_replica(db, room_id="room-1", transition=_transition(STATEMENTS[kind](**replayed), kind=kind),
                                 to_epoch=replayed.get("to_epoch", 2))
    assert replicas.replica_state(db, room_id="room-1")["authority"] == {"gateway_id": AUTH_A, "epoch": 1}
    assert _marks(db) == ([], [])


@pytest.mark.parametrize("kind", sorted(STATEMENTS))
def test_a_forged_or_malformed_signed_statement_is_refused(tmp_path, monkeypatch, kind):
    """The caller checks the signature and last_hash against the signer's key and the room's history.

    Here, a statement changed after its digest was taken, or not in its exact shape, is refused, and a
    statement used once can't verify a second change.
    """
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    make = STATEMENTS[kind]
    tampered = _transition(make(), kind=kind)
    tampered["proof"]["statement"]["successor"] = AUTH_C
    with pytest.raises(replicas.ReplicaError, match="does not match its proof"):
        replicas.promote_replica(db, room_id="room-1", transition=tampered)
    statement = make()["statement"]
    malformed = [
        ({**make(), "signer": AUTH_A}, "unknown fields"),
        ({"statement": statement}, "missing fields"),
        ({"statement": {**statement, "note": "x"}, "signature": make()["signature"]}, "unknown fields"),
        ({**make(), "signature": "hmac-v1." + "S" * 86}, "ed25519-v1"),
        (make(last_hash="A" * 64), "last_hash"),
        (make(last_seq=-1), "last_seq"),
        (make(from_epoch=True), "from_epoch"),
        (make(successor=""), "successor"),
        (_proof(), "missing fields")]  # another kind's proof is no signed statement
    if kind == "evidence":
        malformed += [(make(silent_for_s=-1), "silent_for_s"), (make(silent_since=True), "silent_since"),
                      (make(silent_since="yesterday"), "silent_since"),
                      ({"statement": _handover()["statement"], "signature": make()["signature"]}, "missing fields")]
    for proof, error in malformed:
        with pytest.raises(replicas.ReplicaError, match=error):
            replicas.promote_replica(db, room_id="room-1", transition=_transition(proof, kind=kind))
    assert _marks(db) == ([], [])
    signed = _transition(make(), kind=kind)
    replicas.promote_replica(db, room_id="room-1", transition=copy.deepcopy(signed))
    with rooms._transaction(db, immediate=True) as conn:
        _insert_transition(conn, "room-1", 5, event_id="replayed", payload={
            "from_epoch": 1, "to_epoch": 2, "successor_gateway_id": AUTH_B, **signed})
    assert _quarantine(db) == {"room-1": "unverified_authority_transition"}


@pytest.mark.parametrize("kind, reason", [("handover", "handover"), ("evidence", "automatic")])
def test_the_old_authority_marks_its_own_step_down(tmp_path, monkeypatch, kind, reason):
    """It hands over its exact history, or learns that its successor continued after that history."""
    db = _hosted(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    for stale in (1, 3):
        with pytest.raises(replicas.ReplicaError, match="another last event"):
            replicas.demote_room(db, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2,
                                 transition=_transition(STATEMENTS[kind](last_seq=stale), kind=kind))
    proof = STATEMENTS[kind](last_seq=2)
    replicas.demote_room(db, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2,
                         transition=_transition(proof, kind=kind), display={"reason": reason})
    lost = rooms.read_events(db, room_id="room-1")["events"][-1]
    assert (lost["seq"], lost["kind"], lost["payload"]["proof_kind"], lost["payload"]["reason"]) == (
        3, "authority.lost", kind, reason)
    assert _marks(db)[0] == [("room-1", 1, 2, AUTH_B, kind)] and _quarantine(db) == {}


def test_a_transition_may_skip_an_epoch_nobody_certified(tmp_path, monkeypatch):
    """An attempt fenced N + 1 and did not finish: authority goes from N to N + 2, and N + 1 never had one."""
    db, page = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    proof = _proof(to_epoch=3)
    promoted = replicas.promote_replica(db, room_id="room-1", transition=_transition(proof), to_epoch=3)
    assert (promoted["authority_epoch"], promoted["previous_epoch"]) == (3, 1)
    transition = rooms.read_events(db, room_id="room-1")["events"][-1]
    assert (transition["authority_epoch"], transition["payload"]["from_epoch"], transition["payload"]["to_epoch"]) == (
        3, 1, 3)
    assert _marks(db)[0] == [("room-1", 1, 3, AUTH_B, "certified")] and _quarantine(db) == {}
    rooms.append_event(db, room_id="room-1", event_id="after", kind="message.user", actor=USER,
                       payload={"text": "x"}, authority_gateway_id=AUTH_B, authority_epoch=3)
    # The old authority learns of it the same way, straight to the later epoch.
    (tmp_path / "old").mkdir()
    old = _hosted(tmp_path / "old")
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    replicas.demote_room(old, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=3,
                         transition=_transition(proof))
    assert _marks(old)[0] == [("room-1", 1, 3, AUTH_B, "certified")] and _quarantine(old) == {}


def test_a_transition_only_moves_to_a_later_epoch(tmp_path, monkeypatch):
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    with pytest.raises(replicas.ReplicaEpochRegressionError, match="later epoch"):
        replicas.promote_replica(db, room_id="room-1", transition=_transition(_proof(to_epoch=1)), to_epoch=1)
    with pytest.raises(replicas.ReplicaError, match="only a verified transition"):
        replicas.promote_replica(db, room_id="room-1", to_epoch=3)
    with rooms._transaction(db, immediate=True) as conn:
        for from_epoch, to_epoch in ((2, 2), (3, 2)):
            with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
                conn.execute("INSERT INTO hosted_room_verified_transitions VALUES ('room-1', ?, ?, ?, 'certified', ?, 1)",
                             (from_epoch, to_epoch, AUTH_B, "0" * 64))
    assert replicas.replica_state(db, room_id="room-1")["authority"] == {"gateway_id": AUTH_A, "epoch": 1}
    assert _marks(db) == ([], [])


def test_a_later_epoch_mark_does_not_verify_an_earlier_event(tmp_path):
    db = _hosted(tmp_path)
    for payload in (_transition_payload(), _transition_payload(to_epoch=3)):
        # A mark for N + 2 cannot be spent on an event at N + 1, whatever the event claims.
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            with rooms._transaction(db, immediate=True) as conn:
                _mark(conn, from_epoch=1, to_epoch=3)
                _insert_transition(conn, "room-1", 3, epoch=2, payload=payload)
    assert _marks(db) == ([], []) and _quarantine(db) == {}
    assert [e["kind"] for e in rooms.read_events(db, room_id="room-1")["events"]] == ["message.user"] * 2


def test_moved_history_keeps_its_verified_transitions(tmp_path):
    """A verified transition copied between this store's copy and room tables stays verified."""
    db = tmp_path / "store.db"
    rooms.create_room(db, room_id="seed", name="Seed", members=MEMBERS, authority_gateway_id=AUTH_A)
    message = json.dumps(USER, separators=(",", ":"), sort_keys=True)
    with rooms._transaction(db, immediate=True) as conn:
        for seq in (1, 2):
            conn.execute("INSERT INTO hosted_room_replica_events VALUES ('room-1', ?, ?, 'message.user', ?, 1, "
                         "'{\"text\":\"x\"}', 1)", (seq, f"e{seq}", message))
        _mark(conn)
        _insert_transition(conn, "room-1", 3, table="hosted_room_replica_events")
    with rooms._transaction(db, immediate=True) as conn:
        # An unmarked transition is refused outright in a copy.
        with pytest.raises(sqlite3.IntegrityError, match="not verified"):
            _insert_transition(conn, "room-1", 4, epoch=3, table="hosted_room_replica_events",
                               payload=_transition_payload(from_epoch=2, to_epoch=3, successor=AUTH_C))
    with rooms._transaction(db, immediate=True) as conn:
        rows = conn.execute("SELECT * FROM hosted_room_replica_events WHERE room_id='room-1' ORDER BY seq").fetchall()
        conn.execute("DELETE FROM hosted_room_replica_events WHERE room_id='room-1'")
        conn.execute("INSERT INTO hosted_rooms (room_id, name, members_json, authority_gateway_id, authority_epoch, "
                     "next_seq, event_bytes, revision, created_at, updated_at) "
                     "VALUES ('room-1', 'Field Room', '[]', ?, 2, 4, 0, 1, 1, 1)", (AUTH_B,))
        conn.executemany("INSERT INTO hosted_room_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         [tuple(row) for row in rows])
    assert _quarantine(db) == {}
    # The same verified epoch at another position is a different, unverified event.
    with rooms._transaction(db, immediate=True) as conn:
        _insert_transition(conn, "room-1", 4, event_id="copied-elsewhere")
    assert _quarantine(db) == {"room-1": "unverified_authority_transition"}


def test_first_open_keeps_verified_lineage_and_quarantines_the_rest(tmp_path, monkeypatch):
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    replicas.promote_replica(db, room_id="room-1", transition=_transition())
    rooms.create_room(db, room_id="room-2", name="Other", members=MEMBERS, authority_gateway_id=AUTH_A)
    with sqlite3.connect(db) as conn:
        # An older writer, before these triggers: it records an unmarked transition and demotion.
        conn.execute("DROP TRIGGER trg_hosted_events_quarantine_unsafe_lineage")
        _insert_transition(conn, "room-2", 2, payload=_transition_payload())
        conn.execute("INSERT INTO hosted_room_events VALUES ('room-1', 5, 'lost', 'authority.lost', ?, 3, "
                     "'{}', 6)", (SYSTEM,))
        conn.execute("DROP TABLE hosted_room_quarantine")
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.room_state(db, room_id="room-2")
    assert _quarantine(db) == {"room-1": "unsafe_authority_demotion", "room-2": "unverified_authority_transition"}
    with closing(sqlite3.connect(db)) as conn:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='trg_hosted_events_quarantine_unsafe_lineage'")
        assert "hosted_room_verified_transitions" in sql.fetchone()[0]


def test_an_older_trigger_definition_is_replaced_on_open(tmp_path, monkeypatch):
    db, _ = _copy(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TRIGGER trg_hosted_events_quarantine_unsafe_lineage")
        conn.execute("""CREATE TRIGGER trg_hosted_events_quarantine_unsafe_lineage
            AFTER INSERT ON hosted_room_events WHEN NEW.kind='authority.lost'
            BEGIN INSERT OR IGNORE INTO hosted_room_quarantine VALUES (NEW.room_id, 'old', NEW.created_at); END""")
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    replicas.promote_replica(db, room_id="room-1", transition=_transition())
    assert _quarantine(db) == {}
    with closing(sqlite3.connect(db)) as conn:
        assert safety.safety_schema_is_current(conn)


def test_a_losing_successors_mark_moves_aside_with_its_event(tmp_path, monkeypatch):
    """Two successors claimed epoch 2 and the owner kept the other one: this one's transition and its
    mark go to a quarantined branch, and the kept transition is marked in their place."""
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    replicas.promote_replica(db, room_id="room-1", transition=_transition())
    kept = _proof(successor=AUTH_C)
    with rooms._transaction(db, immediate=True) as conn:
        # The caller moves its own transition out of the log into its branch first.
        conn.execute("DELETE FROM hosted_room_events WHERE room_id='room-1' AND seq=4")
        archived = safety.move_transition_mark_to_branch(conn, room_id="room-1", to_epoch=2, branch_id="branch-1")
        _mark(conn, successor=AUTH_C, digest=safety.transition_proof_digest(kept))
        _insert_transition(conn, "room-1", 4, payload=_transition_payload(successor=AUTH_C, proof=kept))
    assert (archived["successor_gateway_id"], archived["seq"], archived["event_id"]) == (
        AUTH_B, 4, "system:authority-transition:2")
    assert _marks(db) == ([("room-1", 1, 2, AUTH_C, "certified")], [("room-1", 2, 4, "system:authority-transition:2")])
    assert _quarantine(db) == {}
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT branch_id, successor_gateway_id FROM hosted_room_branch_transitions").fetchall() == [
            ("branch-1", AUTH_B)]
        for statement in ("DELETE FROM hosted_room_branch_transitions",
                          "UPDATE hosted_room_branch_transitions SET branch_id='other'"):
            with pytest.raises(sqlite3.IntegrityError, match="is kept"):
                conn.execute(statement)


def test_a_mark_moves_aside_only_once_its_event_left_the_log(tmp_path, monkeypatch):
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    replicas.promote_replica(db, room_id="room-1", transition=_transition())
    with rooms._transaction(db, immediate=True) as conn:
        with pytest.raises(rooms.VerifiedTransitionError, match="still in this room's log"):
            safety.move_transition_mark_to_branch(conn, room_id="room-1", to_epoch=2, branch_id="branch-1")
        with pytest.raises(rooms.VerifiedTransitionError, match="no verified transition"):
            safety.move_transition_mark_to_branch(conn, room_id="room-1", to_epoch=3, branch_id="branch-1")
    assert _marks(db)[0] == [("room-1", 1, 2, AUTH_B, "certified")]
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_branch_transitions").fetchone()[0] == 0


def test_legacy_import_does_not_carry_marks(tmp_path, monkeypatch):
    source, target = tmp_path / "state.db", tmp_path / "shared-state.db"
    db, _ = _copy(tmp_path)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    replicas.promote_replica(db, room_id="room-1", transition=_transition())
    db.rename(source)
    for suffix in ("-wal", "-shm"):
        sidecar = db.with_name(db.name + suffix)
        if sidecar.exists():
            sidecar.rename(source.with_name(source.name + suffix))
    # A mark vouches for the store that wrote it; the imported copy of its history is unverified.
    assert rooms.list_rooms(target)[0]["safety_reason"] == "unverified_authority_transition"
    assert _marks(target) == ([], [])

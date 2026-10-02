"""Copy retirement on real SQLite: no cleanup authority is ever inferred, and retired copies stay retired."""

import base64
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import replace

import pytest

from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_work_records as records
from gateway import hosted_room_work_storage as storage
from gateway import hosted_rooms as rooms
from gateway.hosted_room_safety import _prune_disbanded_replicas_locked
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.fixtures.passive_copy import (
    HOME, HOME_SECRET, TARGET, admit, append, disband, enroll, member, notice, prepare, retire)

OTHER = "install:other"
MEMBERS = [{"member_id": "writer", "profile": "default", "handle": "writer",
            "target": {"kind": "local", "profile": "default"}},
           member("reviewer"), member("other", target=OTHER)]


@pytest.fixture
def pair(tmp_path):
    home, target = tmp_path / "home.db", tmp_path / "participant.db"
    rooms.create_room(home, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    for index in range(3):
        append(home, f"message-{index}")
    return home, target


def enrolled(pair):
    entry = prepare(pair[0])
    enroll(pair[1], entry)
    return entry


def copied_prefix(pair, count=1, room_id="room"):
    replicas.ingest_page(pair[1], room_id=room_id, room_name="Workshop", members=MEMBERS,
                         page=rooms.read_events(pair[0], room_id=room_id, limit=count))


def retired(pair, *, count=1):
    entry = enrolled(pair)
    if count:
        copied_prefix(pair, count)
    disband(pair[0])
    return entry, retire(pair[1], notice(pair[0], entry))


def home_state(home):
    return retirement.home_status(home)[0]["state"]


def test_setup_is_idempotent_and_the_value_is_revealed_only_after_disband(pair):
    entry = enrolled(pair)
    assert prepare(pair[0]) == entry
    assert prepare(pair[0], enrollment_id=entry["enrollment_id"]) == entry
    with pytest.raises(retirement.RetirementConflictError, match="disband has not completed"):
        notice(pair[0], entry)
    assert retirement.pending_notice_ids(pair[0], local_gateway_id=HOME) == []
    disband(pair[0])
    with pytest.raises(retirement.RetirementConflictError, match="not active"):
        prepare(pair[0], target=OTHER)
    outgoing = notice(pair[0], entry)
    assert outgoing.enrollment_id in retirement.pending_notice_ids(pair[0], local_gateway_id=HOME)
    for public in (repr(outgoing), json.dumps(retirement.home_status(pair[0])), json.dumps(entry)):
        assert outgoing.value not in public


@pytest.mark.parametrize("count", [0, 1, 3])
def test_retirement_records_the_actual_coverage_and_is_idempotent(pair, count):
    entry, result = retired(pair, count=count)
    assert (result["stored_seq"], result["source_latest_seq"]) == (count, 3 if count else 0)
    if count:
        state = replicas.copy_state(pair[1], room_id="room")
        assert (state["safety_status"], state["disbanded_at"]) == ("retired", None)
        assert state["copy_retired_at"] == result["retired_at"]
    outgoing = notice(pair[0], entry)
    assert retire(pair[1], outgoing) == result
    with rooms._transaction(pair[1]) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_replica_events WHERE kind='room.disbanded'").fetchone()[0] == 0
    retirement.acknowledge_notice(pair[0], notice=outgoing, response=result)
    assert home_state(pair[0]) == "acknowledged"
    assert retirement.pending_notice_ids(pair[0], local_gateway_id=HOME) == []


def test_a_retired_copy_refuses_late_pages_reenrollment_and_reuse_of_its_id(pair):
    entry, _ = retired(pair)
    with pytest.raises(replicas.ReplicaHistoryExpiredError):
        replicas.ingest_page(pair[1], room_id="room", room_name="Workshop", members=MEMBERS,
                             page=rooms.read_events(pair[0], room_id="room", include_disbanded=True))
    with pytest.raises(retirement.RetirementConflictError, match="reenrolled"):
        enroll(pair[1], entry)
    with pytest.raises(rooms.RoomConflictError):
        rooms.create_room(pair[1], room_id="room", name="Resurrected", members=MEMBERS, authority_gateway_id=TARGET)
    with sqlite3.connect(pair[1]) as conn, pytest.raises(sqlite3.IntegrityError, match="retired"):
        conn.execute("INSERT INTO hosted_room_replica_events VALUES ('room',2,'late','message.user','{}',1,'{}',0)")


@pytest.mark.parametrize("changed", [
    {"room_id": "other"}, {"authority_gateway_id": OTHER}, {"authority_epoch": 2}, {"authority_epoch": True},
    {"target_install_id": OTHER}, {"enrollment_id": "other"}, {"extra": True},
])
def test_the_retirement_scope_cannot_be_rebound(pair, changed):
    entry = enrolled(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    with pytest.raises(retirement.RetirementError):
        retirement.retire_copy(pair[1], payload={**outgoing.payload(), **changed}, value=outgoing.value,
                               local_gateway_id=TARGET)


def test_a_commitment_copied_into_another_enrollment_authorizes_nothing(pair):
    entry = enrolled(pair)
    enroll(pair[1], {**entry, "enrollment_id": "another-enrollment", "room_id": "another-room"})
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(pair[1], payload={**outgoing.payload(), "enrollment_id": "another-enrollment",
                                                  "room_id": "another-room"}, value=outgoing.value,
                               local_gateway_id=TARGET)


def test_an_ordinary_value_cannot_retire_a_copy(pair):
    entry = enrolled(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(pair[1], payload=outgoing.payload(), value="not-a-retirement-capability",
                               local_gateway_id=TARGET)


def test_an_owner_replacement_refuses_the_old_value_and_keeps_its_own(pair):
    original = enrolled(pair)
    replacement = prepare(pair[0], enrollment_id="replacement", replace_enrollment_id=original["enrollment_id"])
    enroll(pair[1], replacement, expected_enrollment_id=original["enrollment_id"])
    disband(pair[0])
    assert len(retirement.pending_notice_ids(pair[0], local_gateway_id=HOME)) == 2
    with pytest.raises(retirement.RetirementAuthorizationError, match="replaced"):
        retire(pair[1], notice(pair[0], original))
    assert retire(pair[1], notice(pair[0], replacement))["retired"]


def test_revocation_and_a_stale_replacement_never_reactivate_cleanup(pair):
    original = enrolled(pair)
    replacement = prepare(pair[0], enrollment_id="replacement", replace_enrollment_id=original["enrollment_id"])
    retirement.revoke_target_enrollment(pair[1], room_id="room", enrollment_id=original["enrollment_id"])
    with pytest.raises(retirement.RetirementConflictError, match="expected state"):
        enroll(pair[1], replacement, expected_enrollment_id=original["enrollment_id"])
    enroll(pair[1], replacement, expected_enrollment_id=original["enrollment_id"], expected_state="revoked")
    disband(pair[0])
    with pytest.raises(retirement.RetirementAuthorizationError):
        retire(pair[1], notice(pair[0], original))


def test_a_lost_reply_reuses_the_revealed_value_even_after_the_home_key_is_gone(pair):
    entry = enrolled(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    result = retire(pair[1], outgoing)

    def lost():
        raise FileNotFoundError("test key unavailable")

    resumed = notice(pair[0], entry, lost)
    assert resumed == outgoing and retire(pair[1], resumed) == result
    retirement.acknowledge_notice(pair[0], notice=resumed, response=result)


def test_a_rotated_key_before_close_is_explicit_and_never_a_new_commitment(pair):
    entry = enrolled(pair)
    with pytest.raises(retirement.RetirementKeyUnavailable):
        prepare(pair[0], secret=b"different-key-for-test-only-32-bytes")
    disband(pair[0])
    with pytest.raises(retirement.RetirementKeyUnavailable):
        notice(pair[0], entry, lambda: b"different-key-for-test-only-32-bytes")
    assert home_state(pair[0]) == "closed"


def test_a_wrong_receipt_never_completes_the_notice(pair):
    entry = enrolled(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    result = retire(pair[1], outgoing)
    with pytest.raises(retirement.RetirementConflictError):
        retirement.acknowledge_notice(pair[0], notice=replace(outgoing, enrollment_id="another"), response=result)
    with pytest.raises(retirement.RetirementConflictError):
        retirement.acknowledge_notice(pair[0], notice=outgoing, response={**result, "commitment": "a" * 64})
    assert home_state(pair[0]) == "ready"


def test_open_obligations_are_never_evicted_for_capacity(pair, monkeypatch):
    entry = enrolled(pair)
    monkeypatch.setattr(retirement, "MAX_PENDING_ENROLLMENTS", 1)
    with pytest.raises(retirement.RetirementCapacityError):
        prepare(pair[0], target=OTHER)
    assert retirement.home_status(pair[0])[0]["enrollment_id"] == entry["enrollment_id"]


def test_pruning_the_disbanded_room_keeps_the_closed_obligation_deliverable(pair):
    entry = enrolled(pair)
    disband(pair[0])
    rooms.prune_disbanded_rooms(pair[0], now=10**12)
    assert notice(pair[0], entry).enrollment_id == entry["enrollment_id"]


def test_the_close_rolls_back_with_its_disband(pair):
    entry = enrolled(pair)
    with sqlite3.connect(pair[0]) as conn:
        conn.execute(f"""CREATE TRIGGER fail_retirement_close BEFORE UPDATE ON {retirement.HOME_TABLE}
            WHEN NEW.state='closed' BEGIN SELECT RAISE(ABORT,'test write failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test write failure"):
        disband(pair[0])
    assert rooms.room_state(pair[0], room_id="room").get("disbanded_at") is None
    assert rooms.room_state(pair[0], room_id="room")["latest_seq"] == 3
    assert retirement.pending_notice_ids(pair[0], local_gateway_id=HOME) == []
    with pytest.raises(retirement.RetirementConflictError):
        notice(pair[0], entry)


def test_an_enrollment_never_overwrites_a_local_room(pair):
    entry = prepare(pair[0])
    rooms.create_room(pair[1], room_id="room", name="Local", members=MEMBERS, authority_gateway_id=TARGET)
    with pytest.raises(retirement.RetirementConflictError, match="locally authoritative"):
        enroll(pair[1], entry)


def test_reclaiming_a_retired_partial_copy_keeps_its_denial_and_active_copies(pair):
    entry = enrolled(pair)
    copied_prefix(pair)
    rooms.create_room(pair[0], room_id="still-active", name="Active", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(pair[0], room_id="still-active", event_id="keep", kind="message.user",
                       actor={"kind": "user", "id": "owner"}, payload={"text": "keep this"},
                       authority_gateway_id=HOME, authority_epoch=1)
    replicas.ingest_page(pair[1], room_id="still-active", room_name="Active", members=MEMBERS,
                         page=rooms.read_events(pair[0], room_id="still-active"))
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    receipt = retire(pair[1], outgoing)
    with rooms._transaction(pair[1], immediate=True) as conn:
        assert _prune_disbanded_replicas_locked(conn, now=None, max_replica_event_bytes=0) == 1
    assert replicas.copy_state(pair[1], room_id="still-active")["last_seq"] == 1
    with pytest.raises(replicas.ReplicaHistoryExpiredError):
        replicas.copy_state(pair[1], room_id="room")
    assert retire(pair[1], outgoing) == receipt
    with pytest.raises(retirement.RetirementConflictError):
        enroll(pair[1], entry)


def test_a_quarantined_copy_stays_unreclaimable_after_retirement(pair):
    retired(pair)
    with rooms._transaction(pair[1], immediate=True) as conn:
        conn.execute("INSERT INTO hosted_room_quarantine VALUES ('room','test-evidence',0)")
        assert _prune_disbanded_replicas_locked(conn, now=10**12, max_replica_event_bytes=0) == 0
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_replica_events").fetchone()[0] == 1


@pytest.mark.parametrize("pressure", ["bytes", "count", "page"])
@pytest.mark.parametrize("quarantined", [False, True])
def test_exact_retirement_remains_available_for_held_history(pair, monkeypatch, pressure, quarantined):
    entry = enrolled(pair)
    copied_prefix(pair, count=3)
    if quarantined:
        with sqlite3.connect(pair[1]) as conn:
            conn.execute("UPDATE hosted_room_replicas SET quarantine_reason='duplicate_event_id',quarantined_at=1")
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    with sqlite3.connect(pair[1]) as conn:
        before = conn.execute("SELECT * FROM hosted_room_replica_events ORDER BY seq").fetchall()
    if pressure == "bytes":
        monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", 1)
    elif pressure == "count":
        monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", 1)
        monkeypatch.setattr(rooms, "CONTROL_EVENT_COUNT_RESERVE", 0)
    else:
        monkeypatch.setattr(replicas, "MAX_LOG_PAGE_BYTES", 1)
    with pytest.raises(replicas.ReplicaCapacityError, match="preserved"):
        replicas.copy_state(pair[1], room_id="room")
    with rooms._transaction(pair[1], immediate=True) as conn:
        assert _prune_disbanded_replicas_locked(conn, now=10**12, max_replica_event_bytes=0) == 0
        assert [tuple(row) for row in conn.execute("SELECT * FROM hosted_room_replica_events ORDER BY seq")] == before
        assert conn.execute("SELECT quarantine_reason FROM hosted_room_replicas").fetchone()[0] == (
            "duplicate_event_id" if quarantined else None)
    if quarantined:
        with pytest.raises(retirement.RetirementConflictError):
            retire(pair[1], outgoing)
        with sqlite3.connect(pair[1]) as conn:
            assert conn.execute("SELECT * FROM hosted_room_replica_events ORDER BY seq").fetchall() == before
            assert conn.execute(f"SELECT COUNT(*) FROM {retirement.RETIREMENT_TABLE}").fetchone()[0] == 0
        return
    receipt = retire(pair[1], outgoing)
    assert (receipt["stored_seq"], receipt["source_latest_seq"]) == (3, 3)
    assert retire(pair[1], outgoing) == receipt
    with rooms._transaction(pair[1], immediate=True) as conn:
        assert _prune_disbanded_replicas_locked(conn, now=None, max_replica_event_bytes=0) == 1
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_replica_events").fetchone()[0] == 0
        assert conn.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "replica"
    with pytest.raises(replicas.ReplicaHistoryExpiredError):
        replicas.copy_state(pair[1], room_id="room")
    assert retire(pair[1], outgoing) == receipt
    with pytest.raises(retirement.RetirementConflictError):
        enroll(pair[1], entry)


@pytest.mark.parametrize("column", ["name", "authority_epoch"])
def test_retirement_rejects_unbounded_header_before_fetch(pair, monkeypatch, column):
    entry = enrolled(pair)
    copied_prefix(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    malformed = "a\0" + "x" * 4096
    with sqlite3.connect(pair[1]) as conn:
        conn.execute(f"UPDATE hosted_room_replicas SET {column}=?", (malformed,))
    transaction = retirement._transaction

    @contextmanager
    def bounded_transaction(path):
        with transaction(path) as conn:
            def bounded_row(cursor, values):
                assert not any(isinstance(value, str) and len(value) > 2048 for value in values), "unbounded fetch"
                return sqlite3.Row(cursor, values)
            conn.row_factory = bounded_row
            yield conn

    monkeypatch.setattr(retirement, "_transaction", bounded_transaction)
    with pytest.raises(retirement.RetirementConflictError, match="metadata"):
        retire(pair[1], outgoing)
    with sqlite3.connect(pair[1]) as conn:
        assert conn.execute(f"SELECT {column} FROM hosted_room_replicas").fetchone()[0] == malformed
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_replica_events").fetchone()[0] == 1
        assert conn.execute(f"SELECT COUNT(*) FROM {retirement.RETIREMENT_TABLE}").fetchone()[0] == 0


def test_one_receipt_supersedes_only_that_copys_earlier_obligations(pair):
    first = enrolled(pair)
    second = prepare(pair[0], enrollment_id="replacement", replace_enrollment_id=first["enrollment_id"])
    other_target = prepare(pair[0], target=OTHER)
    enroll(pair[1], second, expected_enrollment_id=first["enrollment_id"])
    disband(pair[0])
    outgoing = notice(pair[0], second)
    retirement.acknowledge_notice(pair[0], notice=outgoing, response=retire(pair[1], outgoing))
    states = {row["enrollment_id"]: row for row in retirement.home_status(pair[0])}
    assert (states[first["enrollment_id"]]["state"], states[first["enrollment_id"]]["superseded_by"]) == (
        "superseded", second["enrollment_id"])
    assert states[other_target["enrollment_id"]]["state"] == "closed"


def test_a_failed_participant_write_leaves_no_denial_or_retirement(pair):
    entry = enrolled(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    with sqlite3.connect(pair[1]) as conn:
        conn.execute(f"""CREATE TRIGGER fail_retirement_insert BEFORE INSERT ON {retirement.RETIREMENT_TABLE}
            BEGIN SELECT RAISE(ABORT,'test full disk'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test full disk"):
        retire(pair[1], outgoing)
    with rooms._transaction(pair[1]) as conn:
        assert not retirement.copy_retired_locked(conn, "room")
        assert conn.execute("SELECT 1 FROM hosted_room_id_reservations WHERE room_id='room'").fetchone() is None


def test_an_invalid_capability_never_takes_the_writer(pair, monkeypatch):
    entry = enrolled(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    monkeypatch.setattr(retirement, "_transaction", lambda _db: pytest.fail("unauthorized write transaction"))
    signature = base64.urlsafe_b64decode(outgoing.value.split(".", 1)[1] + "==")
    changed = bytes([signature[0] ^ 1]) + signature[1:]
    wrong = "ed25519-v2." + base64.urlsafe_b64encode(changed).decode("ascii").rstrip("=")
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(pair[1], payload=outgoing.payload(), value=wrong, local_gateway_id=TARGET)


def test_replaying_a_completed_retirement_is_read_only(pair):
    entry, receipt = retired(pair)
    outgoing = notice(pair[0], entry)
    with pytest.MonkeyPatch.context() as scoped:
        scoped.setattr(retirement, "_transaction", lambda _db: pytest.fail("completed replay must be read-only"))
        assert retire(pair[1], outgoing) == receipt


def test_a_revocation_between_the_read_check_and_the_writer_wins(pair, monkeypatch):
    entry = enrolled(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    original = retirement._transaction

    @contextmanager
    def revoke_then_enter(db):
        monkeypatch.setattr(retirement, "_transaction", original)
        retirement.revoke_target_enrollment(db, room_id="room", enrollment_id=entry["enrollment_id"])
        with original(db) as conn:
            yield conn

    monkeypatch.setattr(retirement, "_transaction", revoke_then_enter)
    with pytest.raises(retirement.RetirementAuthorizationError, match="revoked"):
        retire(pair[1], outgoing)


def test_a_late_confirmation_cannot_reopen_a_closed_obligation(pair):
    entry = enrolled(pair)
    proof = retirement.current_target_enrollment(pair[1], room_id="room", authority_gateway_id=HOME, authority_epoch=1)
    disband(pair[0])
    assert not retirement.confirm_home_enrollment(pair[0], enrollment_id=entry["enrollment_id"], proof=proof)
    assert home_state(pair[0]) == "closed"


def test_confirmation_is_bound_to_the_exact_owner_enrollment(pair):
    entry = enrolled(pair)
    proof = retirement.current_target_enrollment(pair[1], room_id="room", authority_gateway_id=HOME, authority_epoch=1)
    assert not retirement.confirm_home_enrollment(pair[0], enrollment_id=entry["enrollment_id"],
                                                  proof={**proof, "commitment": "a" * 64})
    assert home_state(pair[0]) == "prepared"
    assert retirement.confirm_home_enrollment(pair[0], enrollment_id=entry["enrollment_id"], proof=proof)
    assert home_state(pair[0]) == "enrolled"


def test_an_enrollment_fences_the_scope_of_the_first_page(pair):
    enrolled(pair)
    page = rooms.read_events(pair[0], room_id="room")
    page["authority"] = {"gateway_id": OTHER, "epoch": 1}
    with pytest.raises(replicas.ReplicaError, match="enrollment"):
        replicas.ingest_page(pair[1], room_id="room", room_name="Workshop", members=MEMBERS, page=page)


def test_a_corrupt_older_copy_is_neither_retired_nor_made_reclaimable(pair):
    entry = enrolled(pair)
    copied_prefix(pair)
    with sqlite3.connect(pair[1]) as conn:
        conn.execute("UPDATE hosted_room_replicas SET last_seq=99,latest_seq=99 WHERE room_id='room'")
    disband(pair[0])
    with pytest.raises(retirement.RetirementConflictError):
        retire(pair[1], notice(pair[0], entry))
    with rooms._transaction(pair[1]) as conn:
        assert not retirement.copy_retired_locked(conn, "room")
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_replica_events").fetchone()[0] == 1


def test_an_older_writer_cannot_change_a_retired_copy(pair):
    retired(pair)
    for sql in ("UPDATE hosted_room_replicas SET authority_epoch=2 WHERE room_id='room'",
                "UPDATE hosted_room_replica_events SET payload_json='{}' WHERE room_id='room'"):
        with sqlite3.connect(pair[1]) as conn, pytest.raises(sqlite3.IntegrityError, match="retired"):
            conn.execute(sql)


def _retired_beside_an_active_copy(pair, *, state):
    entry = enrolled(pair)
    if state != "before_first_page":
        copied_prefix(pair)
    disband(pair[0])
    retire(pair[1], notice(pair[0], entry))
    if state == "reclaimed":
        with rooms._transaction(pair[1], immediate=True) as conn:
            assert _prune_disbanded_replicas_locked(conn, now=None, max_replica_event_bytes=0) == 1
    rooms.create_room(pair[0], room_id="active", name="Active", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(pair[0], room_id="active", event_id="active-event", kind="message.user",
                       actor={"kind": "user", "id": "owner"}, payload={"text": "Active work"},
                       authority_gateway_id=HOME, authority_epoch=1)
    replicas.ingest_page(pair[1], room_id="active", room_name="Active", members=MEMBERS,
                         page=rooms.read_events(pair[0], room_id="active"))
    return pair[1]


def _insert_copy(conn, table, room_id):
    row = dict(conn.execute(f"SELECT * FROM {table} WHERE room_id='active'").fetchone())
    row["room_id"] = room_id
    conn.execute(f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values()))


@pytest.mark.parametrize("state", ["before_first_page", "reclaimed"])
@pytest.mark.parametrize("table", ["hosted_room_replica_events", "hosted_room_replicas"])
def test_nothing_can_be_written_or_moved_into_a_retired_identity(pair, table, state):
    db = _retired_beside_an_active_copy(pair, state=state)
    with closing(sqlite3.connect(db)) as conn:
        conn.row_factory = sqlite3.Row
        assert conn.execute(f"SELECT 1 FROM {table} WHERE room_id='room'").fetchone() is None
        assert conn.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "replica"
        before = tuple(conn.execute(f"SELECT * FROM {table} WHERE room_id='active'").fetchone())
        with pytest.raises(sqlite3.IntegrityError, match="retired"):
            _insert_copy(conn, table, "room")
            conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="retired"):
            conn.execute(f"UPDATE {table} SET room_id='room' WHERE room_id='active'")
            conn.commit()
        assert conn.execute(f"SELECT 1 FROM {table} WHERE room_id='room'").fetchone() is None
        assert tuple(conn.execute(f"SELECT * FROM {table} WHERE room_id='active'").fetchone()) == before


def test_retirement_guards_keep_bookkeeping_and_unrelated_writes_working(pair):
    db = _retired_beside_an_active_copy(pair, state="populated")
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.row_factory = sqlite3.Row
        before = conn.execute("SELECT updated_at,event_bytes FROM hosted_room_replicas WHERE room_id='room'").fetchone()
        conn.execute("UPDATE hosted_room_replicas SET updated_at=updated_at+1,event_bytes=event_bytes+1 WHERE room_id='room'")
        assert tuple(conn.execute("SELECT updated_at,event_bytes FROM hosted_room_replicas WHERE room_id='room'").fetchone()) == (
            before[0] + 1, before[1] + 1)
        conn.execute("UPDATE hosted_room_replicas SET event_bytes=? WHERE room_id='room'", (before[1],))
        conn.execute("UPDATE hosted_room_replicas SET name='Active update' WHERE room_id='active'")
        conn.execute("UPDATE hosted_room_replica_events SET created_at=created_at+1 WHERE room_id='active'")
        for table in ("hosted_room_replicas", "hosted_room_replica_events"):
            _insert_copy(conn, table, "unrelated-active")
        for sql in ("UPDATE hosted_room_replicas SET room_id='moved' WHERE room_id='room'",
                    "UPDATE hosted_room_replica_events SET room_id='moved' WHERE room_id='room'",
                    "UPDATE hosted_room_replicas SET name='Rewritten' WHERE room_id='room'",
                    "UPDATE hosted_room_replica_events SET payload_json='{}' WHERE room_id='room'"):
            with pytest.raises(sqlite3.IntegrityError, match="retired"):
                conn.execute(sql)
        for table in ("hosted_room_replicas", "hosted_room_replica_events"):
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE room_id='room'").fetchone()[0] == 1
            assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE room_id='unrelated-active'").fetchone()[0] == 1


# -- task evidence and retirement --------------------------------------------------------------


def _evidence_on_the_copy(pair):
    admit(pair[0], member_id="reviewer")
    record = records.capture(pair[0], room_id="room", local_gateway_id=HOME)
    copied_prefix(pair, 3)
    with rooms._transaction(pair[1], immediate=True) as conn:
        records.initialize(conn)
        storage.save_locked(conn, records.TARGET_TABLE, record)
    return record


@pytest.mark.parametrize("retirement_first", [False, True])
def test_retirement_removes_the_copys_records_and_fences_older_raw_writers(pair, retirement_first):
    if retirement_first:
        with retirement._transaction(pair[1]):
            pass
    entry = enrolled(pair)
    record = _evidence_on_the_copy(pair)
    disband(pair[0])
    assert retire(pair[1], notice(pair[0], entry))["retired"]
    with sqlite3.connect(pair[1]) as old:
        old.execute("PRAGMA foreign_keys=OFF")
        assert old.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 0
        for verb in ("INSERT", "INSERT OR REPLACE"):
            with pytest.raises(sqlite3.IntegrityError):
                old.execute(f"{verb} INTO {records.TARGET_TABLE} (room_id,revision,digest,record_json,"
                            "producer_gateway_id,producer_epoch,disposition) VALUES (?,?,?,?,?,?,'current')",
                            ("room", record["revision"], record["digest"], records.encode(record), HOME, 1))
        assert old.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id='room'").fetchone()[0] == "replica"


def test_retired_evidence_cannot_be_resurrected_through_another_copy(pair):
    entry = enrolled(pair)
    _evidence_on_the_copy(pair)
    with closing(open_sqlite(pair[1])) as conn:
        old = dict(conn.execute(f"SELECT * FROM {records.TARGET_TABLE}").fetchone())
    disband(pair[0])
    retire(pair[1], notice(pair[0], entry))
    rooms.create_room(pair[0], room_id="active", name="Active", members=MEMBERS, authority_gateway_id=HOME)
    active = records.capture(pair[0], room_id="active", local_gateway_id=HOME)
    replicas.ingest_page(pair[1], room_id="active", room_name="Active", members=MEMBERS,
                         page=rooms.read_events(pair[0], room_id="active"))
    with rooms._transaction(pair[1], immediate=True) as conn:
        records.initialize(conn)
        storage.save_locked(conn, records.TARGET_TABLE, active)
    with closing(open_sqlite(pair[1])) as conn:
        assert conn.execute(f"SELECT * FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone() is None
        before = dict(conn.execute(f"SELECT * FROM {records.TARGET_TABLE}").fetchone())
        for disposition in ("current", "invalid", "historical"):
            proposed = {**old, "disposition": disposition}
            with pytest.raises(sqlite3.IntegrityError, match="retired"):
                conn.execute(f"INSERT INTO {records.TARGET_TABLE} ({','.join(proposed)}) "
                             f"VALUES ({','.join('?' for _ in proposed)})", tuple(proposed.values()))
            with pytest.raises(sqlite3.IntegrityError, match="retired"):
                conn.execute(f"UPDATE {records.TARGET_TABLE} SET room_id='room',disposition=? WHERE room_id='active'",
                             (disposition,))
            assert dict(conn.execute(f"SELECT * FROM {records.TARGET_TABLE}").fetchone()) == before
    assert replicas.copy_state(pair[1], room_id="room")["safety_status"] == "retired"


def _orphan_evidence(path, record):
    """Room-keyed evidence for a room this store has no parent for, as an older store may hold."""
    data = "{ original invalid json bytes"
    with sqlite3.connect(path) as raw:
        raw.execute("PRAGMA foreign_keys=OFF")
        for (name,) in raw.execute("SELECT name FROM sqlite_master WHERE type='trigger' "
                                   "AND name LIKE 'trg_work_invalid_%'").fetchall():
            raw.execute(f'DROP TRIGGER "{name}"')
        raw.execute(f"DROP TABLE IF EXISTS {storage.INVALID_TABLE}")
        for table in (records.SOURCE_TABLE, records.TARGET_TABLE, records.PENDING_TABLE):
            raw.execute(f"DROP TABLE IF EXISTS {table}")
            extra = (",target_install_id TEXT NOT NULL,route_generation TEXT NOT NULL,status TEXT NOT NULL"
                     if table == records.PENDING_TABLE else "")
            raw.execute(f"CREATE TABLE {table} (room_id TEXT PRIMARY KEY,revision INTEGER NOT NULL,"
                        f"digest TEXT NOT NULL,record_json TEXT NOT NULL{extra})")
            values = ("orphan", record["revision"], record["digest"], data)
            if table == records.PENDING_TABLE:
                values += ("original-target", "original-route", "unavailable")
            raw.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in values)})", values)
    with rooms._transaction(path, immediate=True) as conn:
        records.initialize(conn)


@pytest.mark.parametrize("cleanup", ["home_disband", "copy_retirement"])
def test_orphan_evidence_goes_only_with_its_owners_explicit_cleanup(pair, tmp_path, cleanup):
    record = records.capture(pair[0], room_id="room", local_gateway_id=HOME)
    store = tmp_path / "cleanup.db"
    _orphan_evidence(store, record)
    home = store if cleanup == "home_disband" else tmp_path / "owner.db"
    rooms.create_room(home, room_id="orphan", name="Owner", members=MEMBERS, authority_gateway_id=HOME)
    if cleanup == "copy_retirement":
        entry = retirement.prepare_home_enrollment(home, room_id="orphan", target_install_id=TARGET,
                                                   endpoint="https://participant.example", local_gateway_id=HOME,
                                                   secret=HOME_SECRET)
        retirement.enroll_target(store, enrollment=entry, target_install_id=TARGET)
    rooms.disband_room(home, room_id="orphan", expected_gateway_id=HOME, expected_epoch=1)
    if cleanup == "copy_retirement":
        outgoing = retirement.materialize_notice(home, enrollment_id=entry["enrollment_id"], local_gateway_id=HOME,
                                                 secret_loader=lambda: HOME_SECRET)
        assert retirement.retire_copy(store, payload=outgoing.payload(), value=outgoing.value,
                                      local_gateway_id=TARGET)["retired"]
    with rooms._transaction(store) as conn:
        kinds = {r[0] for r in conn.execute(f"SELECT source_table FROM {storage.INVALID_TABLE} WHERE room_id='orphan'")}
        assert kinds == ({records.TARGET_TABLE} if cleanup == "home_disband"
                         else {records.SOURCE_TABLE, records.PENDING_TABLE})
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"DELETE FROM {storage.INVALID_TABLE} WHERE room_id='orphan'")


@pytest.mark.parametrize("retirement_first", [False, True])
def test_either_store_owner_may_initialize_first(pair, retirement_first):
    record = records.capture(pair[0], room_id="room", local_gateway_id=HOME)
    with rooms._transaction(pair[0], immediate=True) as conn:
        if retirement_first:
            retirement._initialize(conn)
        records.initialize(conn)
        retirement._initialize(conn)
        records.initialize(conn)
        row = conn.execute(f"SELECT * FROM {records.SOURCE_TABLE}").fetchone()
        assert (row["revision"], row["digest"], row["disposition"]) == (record["revision"], record["digest"], "current")
        triggers = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert {"trg_work_records_retired_insert", "trg_work_records_retired_update", "trg_work_records_retired_cleanup",
            f"trg_{records.TARGET_TABLE}_delete_v2", "trg_work_invalid_retired"} <= triggers


def test_participant_public_verifier_cannot_authorize_its_own_retirement(pair):
    entry = enrolled(pair)
    disband(pair[0])
    outgoing = notice(pair[0], entry)
    # Treating the enrolled public bytes as a seed produces a different key, not authority.
    forged = retirement._sign_notice(bytes.fromhex(entry['commitment']), entry)
    with pytest.raises(retirement.RetirementAuthorizationError):
        retirement.retire_copy(pair[1], payload=outgoing.payload(), value=forged, local_gateway_id=TARGET)
    result = retirement.retire_copy(pair[1], payload=outgoing.payload(), value=outgoing.value, local_gateway_id=TARGET)
    assert result['retired'] is True

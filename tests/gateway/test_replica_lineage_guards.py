"""V2 evidence budgets, initializer order, and exact canonical control bytes."""

import copy
import json
import sqlite3

import pytest

from gateway import hosted_room_passive_lineage as lineage
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_rooms as rooms
from tests.gateway.test_hosted_room_replica_retirement import (
    HOME, TARGET, MEMBERS, pair, enroll, close, notice,  # noqa: F401
)
from tests.gateway.test_hosted_room_replica_lineage import (
    SUCCESSOR, transfer, setup_successor, page_v2, ingest,
)


@pytest.mark.parametrize("key", ["enrollment_id", "authority_gateway_id", "target_install_id"])
def test_v2_enrollment_identifiers_are_not_silently_repaired(pair, key):
    source, target = pair
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    entry = setup_successor(source, target, spans)
    altered = {**entry, key: " " + entry[key] + " "}
    with pytest.raises(rooms.HostedRoomError):
        retirement.enroll_target(target, enrollment=altered, target_install_id=TARGET, authority_history=spans)


def test_v1_completed_retirement_keeps_its_unversioned_response(pair):
    source, target = pair
    entry = enroll(pair)
    close(source)
    closing = notice(source, entry)
    result = retirement.retire_copy(target, payload=closing.payload(), value=closing.value, local_gateway_id=TARGET)
    assert "version" not in result and "lineage_sha256" not in result and "lineage_status" not in result
    assert retirement.retire_copy(target, payload=closing.payload(), value=closing.value, local_gateway_id=TARGET) == result


@pytest.mark.parametrize("order", ["rooms_first", "replicas_first"])
def test_descriptor_and_raw_old_writer_guards_survive_initializers(pair, order):
    source, target = pair
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    entry = setup_successor(source, target, spans)
    ingest(target, page_v2(source, spans))
    initializers = [rooms._transaction, replicas._replica_transaction]
    if order == "replicas_first":
        initializers.reverse()
    for initializer in initializers:
        with initializer(target):
            pass
    with retirement._transaction(target) as conn:
        row = dict(conn.execute(f"SELECT * FROM {retirement.ENROLLMENT_TABLE}").fetchone())
        fields = tuple(row)
        for index in range(1, lineage.MAX_STORED_DESCRIPTORS):
            candidate = {**row, "enrollment_id": f"quota-{index}", "is_current": 0}
            conn.execute(f"INSERT INTO {retirement.ENROLLMENT_TABLE} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                         tuple(candidate[k] for k in fields))
        with pytest.raises(sqlite3.IntegrityError, match="capacity"):
            candidate = {**row, "enrollment_id": "overflow", "is_current": 0}
            conn.execute(f"INSERT INTO {retirement.ENROLLMENT_TABLE} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                         tuple(candidate[k] for k in fields))
        with pytest.raises(sqlite3.IntegrityError, match="capacity"):
            conn.execute(f"UPDATE {retirement.ENROLLMENT_TABLE} SET authority_history_json=? WHERE enrollment_id=?",
                         ("x" * (lineage.MAX_DESCRIPTOR_BYTES + 1), row["enrollment_id"]))
    with pytest.raises(retirement.RetirementCapacityError):
        retirement.enroll_target(target, enrollment={**entry, "enrollment_id": "capacity-blocked"},
            target_install_id=TARGET, authority_history=spans, expected_enrollment_id=entry["enrollment_id"])
    assert retirement.current_target_enrollment(target, room_id="room", authority_gateway_id=SUCCESSOR,
                                               authority_epoch=2)["enrollment_id"] == entry["enrollment_id"]
    assert replicas.replica_state(target, room_id="room")["lineage_status"] == "verified"
    # An old raw writer with a superficially valid event epoch cannot rewrite a claim.
    with sqlite3.connect(target) as conn:
        bad = copy.deepcopy(rooms.read_events(source, room_id="room")["events"][-1]["payload"])
        bad["previous_gateway_id"] = "install:forged"
        conn.execute("UPDATE hosted_room_replica_events SET payload_json=? WHERE kind='authority.claimed'", (lineage.canonical(bad),))
    state = replicas.replica_state(target, room_id="room")
    assert state["safety_status"] == "quarantined"
    assert "lineage_status" not in state
    assert "authority_history" not in state
    assert "lineage_status" not in replicas.replica_state(target, room_id="room")
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT payload_json FROM hosted_room_replica_events WHERE kind='authority.claimed'").fetchone()[0] == lineage.canonical(bad)
    with pytest.raises(rooms.HostedRoomError):
        ingest(target, page_v2(source, spans))


def test_claim_payload_metadata_is_retained_as_evidence_not_authority(pair):
    source, target = pair
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    # The upper manual-recovery owner adds metadata after the same canonical CAS;
    # materialize that legitimate extra payload without importing activation code.
    with rooms._transaction(source) as conn:
        row = conn.execute("SELECT payload_json FROM hosted_room_events WHERE kind='authority.claimed'").fetchone()
        payload = json.loads(row[0])
        payload["manual_recovery"] = {"confirmed_by": "operator", "unseen_tail": "unknown", "note": "Café 日本語"}
        encoded = lineage.canonical(payload)
        conn.execute("UPDATE hosted_room_events SET payload_json=? WHERE kind='authority.claimed'", (encoded,))
        conn.execute("UPDATE hosted_rooms SET event_bytes=event_bytes+?", (len(encoded.encode()) - len(row[0].encode()),))
    setup_successor(source, target, spans)
    ingest(target, page_v2(source, spans))
    with rooms._transaction(target) as conn:
        assert conn.execute("SELECT payload_json FROM hosted_room_replica_events WHERE kind='authority.claimed'").fetchone()[0] == encoded
    with pytest.raises(rooms.RoomNotFoundError):
        rooms.room_state(target, room_id="room")

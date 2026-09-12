"""A v1 prefix header is not authorization for a successor-enrolled copy."""

import pytest
import sqlite3

from gateway import hosted_room_driver as driver
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_work_records as records
from gateway import hosted_rooms as rooms
from tests.gateway.test_hosted_room_replica_ingress import (
    HOME, TARGET, SECRET, grant, ingest, pair,  # noqa: F401
)
from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, transfer


def test_old_v1_work_cannot_update_a_successor_enrolled_initial_prefix(pair):
    source, target = pair
    task = driver.TaskIdentity("room", "task", "thread", "turn")
    driver.admit_task(source, task, payload={"prompt": "private", "target_profile": "reviewer",
        "target_member_id": "reviewer", "source_event_seq": 1}, clock=lambda: 100)
    old, claims = grant(target, permissions=("status", "replicate", "work_records"))
    original = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
        endpoint="https://participant.example", local_gateway_id=HOME, secret=SECRET, enrollment_id="original")
    retirement.enroll_target(target, enrollment=original, target_install_id=TARGET)
    ingest(pair, old)
    first = records.capture(source, room_id="room", local_gateway_id=HOME)
    kwargs = dict(secret=SECRET, target_install_id=TARGET, target_profile="reviewer")
    records.ingest(target, record=first, token=old, **kwargs)
    lease = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
        process_generation="process", ttl_seconds=30, clock=lambda: 100)
    driver.start_task(source, task, lease, expected_cancel_generation=0, clock=lambda: 100)
    late = records.capture(source, room_id="room", local_gateway_id=HOME)
    assert late["revision"] > first["revision"] and late["digest"] != first["digest"]
    spans = [{"gateway_id": HOME, "epoch": 1, "from_seq": 0}, transfer(source)]
    successor = retirement.prepare_home_enrollment(source, room_id="room", target_install_id=TARGET,
        endpoint="https://participant.example", local_gateway_id=SUCCESSOR, secret=SECRET,
        enrollment_id="successor", replace_enrollment_id="original")
    retirement.enroll_target(target, enrollment=successor, target_install_id=TARGET,
        authority_history=spans, expected_enrollment_id="original")
    with sqlite3.connect(target) as raw:
        with pytest.raises(sqlite3.IntegrityError, match="historical"):
            raw.execute(f"UPDATE {records.TARGET_TABLE} SET record_json=? WHERE room_id='room'", (records.encode(late),))
    state = replicas.replica_state(target, room_id="room")
    assert state["authority"] == first["authority"]
    assert state["source_authority"] == {"gateway_id": SUCCESSOR, "epoch": 2}
    assert state["lineage_status"] == "pending"
    assert rooms.peer_room_grant_is_current(target, claims=claims)
    with pytest.raises(records.WorkRecordError, match="unsupported.*lineage"):
        records.ingest(target, record=late, token=old, **kwargs)
    with rooms._transaction(target) as conn:
        retained = conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone()[0]
    assert retained == records.encode(first)
    fresh, _ = grant(target, permissions=("status", "replicate", "work_records"), grant_id="successor",
        home_install_id=SUCCESSOR, authority_gateway_id=SUCCESSOR, authority_epoch=2)
    assert ingest(pair, fresh, page=rooms.read_events(source, room_id="room", replica_version=2))["lineage_status"] == "verified"
    work = replicas.replica_state(target, room_id="room")["work_records"]
    assert work["availability"] == "not_retained"
    assert work["scopes"][0]["digest"] == first["digest"]
    assert work["scopes"][0]["disposition"] == "historical"

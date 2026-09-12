"""Consistent, bounded metadata without prompt/result/private-state copying."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_link_records as links
from gateway import hosted_rooms as rooms
from gateway import hosted_room_work_records as records

HOME = "install:home"
MEMBERS = [{"member_id": "ops", "profile": "ops", "target": {"kind": "local", "profile": "ops"}}]
IDENTITY = driver.TaskIdentity("room", "task", "thread", "turn")


@pytest.fixture
def source(tmp_path):
    db = tmp_path / "source.db"
    rooms.create_room(db, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(db, room_id="room", event_id="hello", kind="message.user",
                       actor={"kind": "user", "id": "owner"}, payload={"text": "PRIVATE_MESSAGE"},
                       authority_gateway_id=HOME, authority_epoch=1)
    driver.admit_task(db, IDENTITY, payload={"prompt": "PRIVATE_PROMPT /private/workspace API_KEY",
                     "target_profile": "ops", "source_event_seq": 1}, clock=lambda: 100)
    return db


def capture(db):
    return records.capture(db, room_id="room", local_gateway_id=HOME)


def test_phase_revisions_do_not_depend_on_history_and_survive_reopen(source):
    first = capture(source)
    assert capture(source) == first
    held = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
                                process_generation="process", ttl_seconds=30, clock=lambda: 100)
    attempt = driver.start_task(source, IDENTITY, held, expected_cancel_generation=0, clock=lambda: 100)
    second = capture(source)
    assert second["history"] == first["history"]
    assert second["revision"] == first["revision"] + 1
    assert second["tasks"][0]["phase"] == "running"
    driver.settle_task(source, attempt, settlement_id="settled", status="settled",
                       result={"text": "PRIVATE_RESULT", "path": "/private/result"}, clock=lambda: 100)
    third = capture(source)
    assert third["revision"] == second["revision"] + 1
    assert third["tasks"][0]["settlement_id"] == "settled"
    encoded = json.dumps(third)
    for private in ("PRIVATE_MESSAGE", "PRIVATE_PROMPT", "PRIVATE_RESULT", "/private", "API_KEY", "result_json", "prompt"):
        assert private not in encoded
    assert third["limitations"] == records.LIMITATIONS


@pytest.mark.parametrize("failure", ["bound", "unsupported"])
def test_incomplete_capture_is_explicit_not_a_truncated_complete_list(source, monkeypatch, failure):
    if failure == "bound":
        monkeypatch.setattr(records, "MAX_TASKS", 0)
    else:
        with rooms._transaction(source, immediate=True) as conn:
            conn.execute("UPDATE hosted_room_driver_tasks SET payload_json=?", ('{"field_private_state":true}',))
    result = capture(source)
    assert result["availability"] == "unavailable"
    assert result["reason"] == ("bounds_exceeded" if failure == "bound" else "unsupported_task")
    assert result["tasks"] == result["receipts"] == []
    assert capture(source)["revision"] == result["revision"]


def test_capture_preserves_close_fact_before_stop_event(source):
    first = capture(source)
    links.begin_room_link_retirement(source, room_id="room", authority_gateway_id=HOME, authority_epoch=1)
    second = capture(source)
    assert second["history"] == first["history"]
    assert second["stop"] == {"closing": True, "revocation_complete": False, "seq": 0, "cancel_id": None}
    rooms.request_room_stop(source, room_id="room", cancel_id="stop", expected_gateway_id=HOME, expected_epoch=1)
    assert capture(source)["stop"]["cancel_id"] == "stop"


def test_capture_is_one_sqlite_view_while_another_writer_changes_phase(source, monkeypatch):
    entered, release, writing = threading.Event(), threading.Event(), threading.Event()
    original = records._capture_tasks
    def paused(*args):
        result = original(*args)
        entered.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(records, "_capture_tasks", paused)
    with ThreadPoolExecutor(max_workers=2) as pool:
        snapshot = pool.submit(capture, source)
        assert entered.wait(5)
        def mutate():
            writing.set()
            links.begin_room_link_retirement(source, room_id="room", authority_gateway_id=HOME, authority_epoch=1)
        writer = pool.submit(mutate)
        assert writing.wait(5)
        release.set()
        result = snapshot.result(timeout=5)
        writer.result(timeout=5)
    assert result["stop"]["closing"] is False
    assert capture(source)["stop"]["closing"] is True


def test_capture_waits_for_the_acknowledged_history_prefix(source):
    with pytest.raises(records.WorkRecordPrefixError):
        records.capture(source, room_id="room", local_gateway_id=HOME, through_seq=0)


def test_missing_task_store_is_not_claimed_to_be_empty(tmp_path):
    db = tmp_path / "empty.db"
    rooms.create_room(db, room_id="room", name="Empty", members=MEMBERS, authority_gateway_id=HOME)
    assert capture(db)["reason"] == "task_store_missing"


@pytest.mark.parametrize("outcome", ["pending", "unavailable", "rejected"])
def test_authority_scopes_preserve_pending_bytes_and_capture_independently(source, outcome):
    from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, FINAL, transfer
    with rooms._transaction(source, immediate=True) as conn:
        old = records.prepare_delivery_locked(conn, room_id="room", target_install_id="install:target",
            route_generation="old", local_gateway_id=HOME, through_seq=1)
        records.delivery_status_locked(conn, room_id="room", target_install_id="install:target",
            route_generation="old", record=old, status=outcome)
        frozen = dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone())
    transfer(source)
    with rooms._transaction(source, immediate=True) as conn:
        current = records.prepare_delivery_locked(conn, room_id="room", target_install_id="install:target",
            route_generation="new", local_gateway_id=SUCCESSOR, through_seq=2)
        assert current["version"] == 2
        assert current["authority"] == {"gateway_id": SUCCESSOR, "epoch": 2}
        assert current["revision"] == 1
        assert current["tasks"] == old["tasks"]
        assert current["incompleteness"] == ["prior_authority_work_unknown"]
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {records.PENDING_TABLE} ORDER BY producer_epoch")]
        assert len(rows) == 2
        assert rows[0] == {**frozen, "disposition": "superseded_authority"}
        assert not records.delivery_status_locked(conn, room_id="room", target_install_id="install:target",
            route_generation="old", record=old, status="acked")
    transfer(source, SUCCESSOR, 2, FINAL)
    third = records.capture(source, room_id="room", local_gateway_id=FINAL)
    assert third["authority"]["epoch"] == 3 and third["revision"] == 1
    with rooms._transaction(source) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {records.SOURCE_TABLE}").fetchone()[0] == 3
        assert conn.execute(f"SELECT record_json FROM {records.SOURCE_TABLE} WHERE producer_epoch=1").fetchone()[0] == records.encode(old)


def test_successor_with_empty_driver_store_does_not_manufacture_complete_evidence(source):
    from tests.gateway.test_hosted_room_replica_lineage import transfer, SUCCESSOR
    transfer(source)
    with rooms._transaction(source, immediate=True) as conn:
        conn.execute("DELETE FROM hosted_room_driver_tasks")
    result = records.capture(source, room_id="room", local_gateway_id=SUCCESSOR)
    assert result["tasks"] == []
    assert result["incompleteness"] == ["prior_authority_work_unknown"]


@pytest.mark.parametrize("retirement_first", [False, True])
@pytest.mark.parametrize("invalid", [False, True])
def test_v1_migration_preserves_exact_rows_and_marks_invalid(source, retirement_first, invalid):
    from gateway import hosted_room_replica_retirement as retirement
    record = capture(source)
    data = "invalid original bytes" if invalid else json.dumps(record, indent=2)
    with rooms._transaction(source, immediate=True) as conn:
        for table in (records.SOURCE_TABLE, records.TARGET_TABLE, records.PENDING_TABLE):
            conn.execute(f"DROP TABLE {table}")
        for table in (records.SOURCE_TABLE, records.TARGET_TABLE):
            conn.execute(f"CREATE TABLE {table} (room_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL)")
        conn.execute(f"CREATE TABLE {records.PENDING_TABLE} (room_id TEXT NOT NULL, target_install_id TEXT NOT NULL, route_generation TEXT NOT NULL, revision INTEGER NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL, status TEXT NOT NULL, PRIMARY KEY(room_id,target_install_id))")
        conn.execute(f"INSERT INTO {records.SOURCE_TABLE} VALUES (?,?,?,?)", ("room", record["revision"], record["digest"], data))
        conn.execute(f"INSERT INTO {records.PENDING_TABLE} VALUES (?,?,?,?,?,?,?)", ("room", "install:target", "old", record["revision"], record["digest"], data, "unavailable"))
        if retirement_first:
            retirement._initialize(conn)
        records.initialize(conn)
        retirement._initialize(conn)
        records.initialize(conn)
        for table in (records.SOURCE_TABLE, records.PENDING_TABLE):
            row = conn.execute(f"SELECT * FROM {table}").fetchone()
            assert row["record_json"] == data
            assert (row["revision"], row["digest"]) == (record["revision"], record["digest"])
            assert (row["producer_gateway_id"], row["producer_epoch"], row["disposition"]) == (
                ("", 0, "invalid") if invalid else (HOME, 1, "current"))
        if not invalid:
            records.prepare_delivery_locked(conn, room_id="room", target_install_id="install:target",
                route_generation="retry", local_gateway_id=HOME, through_seq=1)
            assert conn.execute(f"SELECT record_json FROM {records.PENDING_TABLE}").fetchone()[0] == data


@pytest.mark.parametrize("bound", ["rows", "bytes"])
def test_all_scopes_count_against_capacity_without_evicting_old_evidence(source, monkeypatch, bound):
    from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, transfer
    old = capture(source)
    transfer(source)
    if bound == "rows":
        monkeypatch.setattr(records, "MAX_STORE_ROWS", 1)
    else:
        monkeypatch.setattr(records, "MAX_STORE_BYTES", len(records.encode(old).encode()))
    with pytest.raises(records.WorkRecordCapacityError):
        records.capture(source, room_id="room", local_gateway_id=SUCCESSOR)
    with rooms._transaction(source) as conn:
        assert conn.execute(f"SELECT record_json FROM {records.SOURCE_TABLE}").fetchone()[0] == records.encode(old)


def test_same_producer_can_replace_latest_at_the_exact_store_row_limit(source, monkeypatch):
    monkeypatch.setattr(records, "MAX_STORE_ROWS", 1)
    old = capture(source)
    held = driver.acquire_lease(source, room_id="room", gateway_id=HOME, authority_epoch=1,
        process_generation="process", ttl_seconds=30, clock=lambda: 100)
    driver.start_task(source, IDENTITY, held, expected_cancel_generation=0, clock=lambda: 100)
    current = capture(source)
    assert current["revision"] == old["revision"] + 1
    with rooms._transaction(source) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {records.SOURCE_TABLE}").fetchone()[0] == 1


def test_store_capacity_is_hard_bounded(source, monkeypatch):
    monkeypatch.setattr(records, "MAX_STORE_BYTES", 10)
    with pytest.raises(records.WorkRecordCapacityError):
        capture(source)

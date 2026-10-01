"""The home's task evidence: one consistent, bounded, private view, frozen per participant until acknowledged."""

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_room_work_records as records
from gateway import hosted_room_work_storage as storage
from gateway import hosted_rooms as rooms
from gateway.hosted_room_replication import HostedRoomReplicationPublisher
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.fixtures.passive_copy import HOME, KEY, MEMBERS, TASK, admit, copying, pair, start  # noqa: F401

TABLES = (records.SOURCE_TABLE, records.TARGET_TABLE, records.PENDING_TABLE)


@pytest.fixture
def source(tmp_path):
    db = tmp_path / "home.db"
    rooms.create_room(db, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    rooms.append_event(db, room_id="room", event_id="hello", kind="message.user", actor={"kind": "user", "id": "owner"},
                       payload={"text": "PRIVATE_MESSAGE"}, authority_gateway_id=HOME, authority_epoch=1)
    admit(db)  # The driver captures evidence on admission already.
    return db


def capture(db):
    return records.capture(db, room_id="room", local_gateway_id=HOME)


def rows(path, table):
    with closing(open_sqlite(path)) as conn:
        return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY room_id,producer_epoch")]


def retained(conn):
    return {t: conn.execute(f"SELECT room_id,revision,digest,record_json FROM {t}").fetchall() for t in TABLES}


def install_legacy(db, record, *, metadata_bytes=0, invalid=False, orphan=False):
    """Recreate the first, room-keyed evidence tables with original bytes, as an older store has them."""
    data = "{opaque invalid original" if invalid else json.dumps(record, indent=2)
    with sqlite3.connect(db) as raw:
        raw.execute("PRAGMA foreign_keys=OFF")
        for (name,) in raw.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'trg_work_invalid_%'").fetchall():
            raw.execute(f'DROP TRIGGER "{name}"')
        raw.execute(f"DROP TABLE IF EXISTS {storage.INVALID_TABLE}")
        for table in TABLES:
            raw.execute(f"DROP TABLE IF EXISTS {table}")
            extra = (",target_install_id TEXT NOT NULL,route_generation TEXT NOT NULL,status TEXT NOT NULL"
                     if table == records.PENDING_TABLE else "")
            raw.execute(f"CREATE TABLE {table} (room_id TEXT PRIMARY KEY,revision INTEGER NOT NULL,"
                        f"digest TEXT NOT NULL,record_json TEXT NOT NULL{extra})")
        room = "orphan" if orphan else "room"
        raw.execute(f"INSERT INTO {records.SOURCE_TABLE} VALUES (?,?,?,?)", (room, record["revision"], record["digest"], data))
        raw.execute(f"INSERT INTO {records.PENDING_TABLE} VALUES (?,?,?,?,?,?,?)",
                    (room, record["revision"], record["digest"], data, "target",
                     "x" * metadata_bytes or "legacy-route", "unavailable"))
    return data


def test_phase_revisions_are_independent_of_history_survive_reopen_and_stay_private(source):
    first = capture(source)
    assert capture(source) == first
    attempt = start(source)
    second = capture(source)
    assert second["history"] == first["history"]
    assert (second["revision"], second["tasks"][0]["phase"]) == (first["revision"] + 1, "running")
    driver.settle_task(source, attempt, settlement_id="settled", status="settled",
                       result={"text": "PRIVATE_RESULT", "path": "/private/result"}, clock=lambda: 100)
    third = capture(source)
    assert (third["revision"], third["tasks"][0]["settlement_id"]) == (second["revision"] + 1, "settled")
    encoded = json.dumps(third)
    for private in ("PRIVATE_MESSAGE", "PRIVATE_PROMPT", "PRIVATE_RESULT", "/private", "result_json", "prompt"):
        assert private not in encoded
    assert third["limitations"] == records.LIMITATIONS


@pytest.mark.parametrize("failure", ["bound", "unsupported"])
def test_an_incomplete_capture_is_explicit_never_a_truncated_complete_list(source, monkeypatch, failure):
    if failure == "bound":
        # Admission already captured one task; a second one now exceeds the bound.
        monkeypatch.setattr(records, "MAX_TASKS", 1)
        admit(source, driver.TaskIdentity("room", "task-2", "thread", "turn-2"))
    else:
        with rooms._transaction(source, immediate=True) as conn:
            conn.execute("UPDATE hosted_room_driver_tasks SET payload_json=?", ('{"field_private_state":true}',))
    result = capture(source)
    assert (result["availability"], result["reason"]) == (
        "unavailable", "bounds_exceeded" if failure == "bound" else "unsupported_task")
    assert result["tasks"] == result["receipts"] == []
    assert capture(source)["revision"] == result["revision"]


def test_the_stop_fact_is_captured_and_canonical_close_facts_are_never_claimed(source):
    first = capture(source)
    assert first["stop"] == {"closing": False, "revocation_complete": False, "seq": 0, "cancel_id": None}
    rooms.request_room_stop(source, room_id="room", cancel_id="stop", expected_gateway_id=HOME, expected_epoch=1)
    stopped = capture(source)
    assert stopped["stop"]["cancel_id"] == "stop" and stopped["stop"]["closing"] is False


def test_a_capture_is_one_sqlite_view_while_another_writer_changes_a_phase(source, monkeypatch):
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
            start(source)

        writer = pool.submit(mutate)
        assert writing.wait(5)
        release.set()
        result = snapshot.result(timeout=5)
        writer.result(timeout=5)
    assert result["tasks"][0]["phase"] == "queued"
    monkeypatch.setattr(records, "_capture_tasks", original)
    assert capture(source)["tasks"][0]["phase"] == "running"


def test_a_capture_waits_for_the_acknowledged_history_prefix(source):
    with pytest.raises(records.WorkRecordPrefixError):
        records.capture(source, room_id="room", local_gateway_id=HOME, through_seq=0)


def test_a_missing_task_store_is_never_claimed_to_be_empty(tmp_path):
    db = tmp_path / "empty.db"
    rooms.create_room(db, room_id="room", name="Empty", members=MEMBERS, authority_gateway_id=HOME)
    assert capture(db)["reason"] == "task_store_missing"


def test_a_quarantined_source_room_captures_nothing(source):
    with sqlite3.connect(source) as conn:
        conn.execute("INSERT INTO hosted_room_quarantine VALUES ('room','test-unverified-lineage',0)")
    with pytest.raises(records.WorkRecordError, match="source is unavailable"):
        capture(source)


def test_store_capacity_is_hard_bounded(source, monkeypatch):
    monkeypatch.setattr(records, "MAX_STORE_BYTES", 10)
    start(source)  # A real change: the unchanged admission capture needs no new row.
    with pytest.raises(records.WorkRecordCapacityError):
        capture(source)


def test_the_same_producer_can_replace_its_record_at_the_exact_row_limit(source, monkeypatch):
    monkeypatch.setattr(records, "MAX_STORE_ROWS", 1)
    old = capture(source)
    start(source)
    assert capture(source)["revision"] == old["revision"] + 1
    with rooms._transaction(source) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {records.SOURCE_TABLE}").fetchone()[0] == 1


@pytest.mark.parametrize("invalid", [False, True])
def test_migration_keeps_exact_original_bytes_and_marks_the_unreadable_invalid(source, invalid):
    record = capture(source)
    data = install_legacy(source, record, invalid=invalid)
    with rooms._transaction(source, immediate=True) as conn:
        records.initialize(conn)
        records.initialize(conn)
        for table in (records.SOURCE_TABLE, records.PENDING_TABLE):
            row = conn.execute(f"SELECT * FROM {table}").fetchone()
            assert (row["record_json"], row["revision"], row["digest"]) == (data, record["revision"], record["digest"])
            assert (row["producer_gateway_id"], row["producer_epoch"], row["disposition"]) == (
                ("", 0, "invalid") if invalid else (HOME, 1, "current"))
        if not invalid:
            records.prepare_delivery_locked(conn, room_id="room", target_install_id="target",
                                            route_generation="retry", local_gateway_id=HOME, through_seq=1)
            assert conn.execute(f"SELECT record_json FROM {records.PENDING_TABLE}").fetchone()[0] == data


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_the_first_initializer_is_atomic_and_joins_an_outer_transaction(source, outer, failure):
    install_legacy(source, capture(source))
    with closing(open_sqlite(source)) as conn:
        original = retained(conn)
        schema = conn.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
        if outer:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE hosted_rooms SET name='caller write'")
        copied = []

        def deny(action, first, second, database, origin):
            if action == sqlite3.SQLITE_INSERT and first == records.SOURCE_TABLE:
                copied.append(first)
            if failure and action == sqlite3.SQLITE_INSERT and first == records.PENDING_TABLE:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(deny)
        if failure:
            with pytest.raises(sqlite3.DatabaseError):
                records.initialize(conn)
        else:
            records.initialize(conn)
        conn.set_authorizer(None)
        assert copied and conn.in_transaction is outer
        if outer:
            assert conn.execute("SELECT name FROM hosted_rooms").fetchone()[0] == "caller write"
            if failure:
                assert retained(conn) == original
                assert conn.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall() == schema
            conn.rollback()
        assert retained(conn) == original
        if failure or outer:
            assert conn.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall() == schema
    with closing(open_sqlite(source)) as conn:
        assert retained(conn) == original
        conn.execute("BEGIN IMMEDIATE")  # No first-opener lease survived.
        conn.rollback()


def test_the_initializer_reserves_the_writer_before_reading_the_schema(source, monkeypatch):
    initialize_locked = storage._initialize_locked
    with closing(sqlite3.connect(source, timeout=0)) as competing_writer:
        def check_writer_reserved(conn):
            conn.execute("SELECT name FROM sqlite_master").fetchall()
            # A deferred read could lose its write upgrade to another publisher at once.
            try:
                with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                    competing_writer.execute("BEGIN IMMEDIATE")
            finally:
                competing_writer.rollback()
            initialize_locked(conn)

        monkeypatch.setattr(storage, "_initialize_locked", check_writer_reserved)
        with closing(sqlite3.connect(source)) as conn:
            conn.row_factory = sqlite3.Row
            storage.initialize(conn)
            assert not conn.in_transaction
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                storage.initialize(conn)
                assert conn.in_transaction  # Never commits the caller's transaction.


def test_public_status_never_drops_legacy_evidence(source):
    original = install_legacy(source, capture(source))
    publisher = HostedRoomReplicationPublisher(source)
    publisher.status("room")
    with sqlite3.connect(source) as raw:
        first = {t: raw.execute(f"SELECT record_json FROM {t}").fetchall() for t in (records.SOURCE_TABLE, records.PENDING_TABLE)}
    publisher.status("room")
    with sqlite3.connect(source) as raw:
        second = {t: raw.execute(f"SELECT record_json FROM {t}").fetchall() for t in (records.SOURCE_TABLE, records.PENDING_TABLE)}
    assert first == {records.SOURCE_TABLE: [(original,)], records.PENDING_TABLE: [(original,)]} == second


def test_parent_backed_invalid_metadata_exhausts_the_shared_budget(source):
    original = install_legacy(source, capture(source), metadata_bytes=records.MAX_STORE_BYTES, invalid=True)
    with rooms._transaction(source, immediate=True) as conn:
        records.initialize(conn)
        row = conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone()
        assert (row["disposition"], row["record_json"]) == ("invalid", original)
        assert len(row["route_generation"].encode()) == records.MAX_STORE_BYTES
    start(source)
    with pytest.raises(records.WorkRecordCapacityError):
        capture(source)


@pytest.mark.parametrize("bound", ["rows", "bytes"])
def test_orphan_evidence_is_kept_immutable_and_charged_without_inventing_a_parent(source, monkeypatch, bound):
    record = capture(source)
    data = install_legacy(source, {**record, "revision": record["revision"] + 7}, orphan=True)
    monkeypatch.setattr(records, "MAX_STORE_ROWS" if bound == "rows" else "MAX_STORE_BYTES", 2 if bound == "rows" else 1)
    with rooms._transaction(source, immediate=True) as conn:
        records.initialize(conn)
        invalid = [dict(r) for r in conn.execute(f"SELECT * FROM {storage.INVALID_TABLE} ORDER BY source_table")]
        assert {r["source_table"] for r in invalid} == {records.SOURCE_TABLE, records.PENDING_TABLE}
        for row in invalid:
            assert (row["room_id"], row["disposition"], row["record_json"]) == ("orphan", "invalid", data)
            assert "producer_gateway_id" not in row
        pending = next(r for r in invalid if r["source_table"] == records.PENDING_TABLE)
        assert (pending["target_install_id"], pending["status"]) == ("target", "unavailable")
        assert not conn.execute("PRAGMA foreign_key_check").fetchall()
        assert not conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id='orphan'").fetchone()
    start(source)
    with pytest.raises(records.WorkRecordCapacityError):
        capture(source)
    with sqlite3.connect(source) as raw:
        for sql in (f"UPDATE {storage.INVALID_TABLE} SET record_json='lost'", f"DELETE FROM {storage.INVALID_TABLE}"):
            with pytest.raises(sqlite3.IntegrityError):
                raw.execute(sql)


@pytest.mark.parametrize("damage", ["digest", "revision", "record_json"])
def test_a_capture_never_reuses_or_overwrites_a_metadata_invalid_source_record(source, damage):
    capture(source)
    with sqlite3.connect(source) as conn:
        conn.execute(f"UPDATE {records.SOURCE_TABLE} SET {damage}=?", (99 if damage == "revision" else "wrong",))
        before = conn.execute(f"SELECT revision,digest,record_json FROM {records.SOURCE_TABLE}").fetchone()
    with pytest.raises(records.WorkRecordError):
        capture(source)
    with sqlite3.connect(source) as conn:
        assert conn.execute(f"SELECT revision,digest,record_json FROM {records.SOURCE_TABLE}").fetchone() == before
        assert conn.execute(f"SELECT disposition FROM {records.SOURCE_TABLE}").fetchone()[0] == "invalid"


def test_a_delivery_outcome_is_recorded_only_for_the_exact_frozen_bytes(source):
    with rooms._transaction(source, immediate=True) as conn:
        record = records.prepare_delivery_locked(conn, room_id="room", target_install_id="target",
                                                 route_generation="route", local_gateway_id=HOME, through_seq=1)
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET record_json='corrupt after preparation'")
        assert not records.delivery_status_locked(conn, room_id="room", target_install_id="target",
                                                  route_generation="route", record=record, status="acked")
    with sqlite3.connect(source) as conn:
        assert conn.execute(f"SELECT status,disposition,record_json FROM {records.PENDING_TABLE}").fetchone() == (
            "pending", "invalid", "corrupt after preparation")


def test_a_delivery_summary_never_labels_mismatched_metadata_current(source):
    with rooms._transaction(source, immediate=True) as conn:
        records.prepare_delivery_locked(conn, room_id="room", target_install_id="target",
                                        route_generation="generation", local_gateway_id=HOME, through_seq=1)
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET digest='wrong'")
        summary = records.delivery_summaries_locked(conn)[0]
    assert (summary["disposition"], summary["incompleteness"], summary["source_loss_safe"]) == (
        "invalid", ["invalid_work_evidence"], False)


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("column", ["digest", "record_json", "revision"])
@pytest.mark.parametrize("replace", [False, True])
def test_every_snapshot_metadata_change_obeys_the_budget(copying, monkeypatch, table, column, replace):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    path = copying.target if table == records.TARGET_TABLE else copying.source
    monkeypatch.setattr(records, "MAX_STORE_BYTES", 1)
    with rooms._transaction(path, immediate=True) as conn:
        records.initialize(conn)  # Keep the overage; refresh the durable limit.
        row = dict(conn.execute(f"SELECT * FROM {table}").fetchone())
        changed = {**row, column: str(row[column]) + "9"}
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            if replace:
                conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(changed)}) "
                             f"VALUES ({','.join('?' for _ in changed)})", tuple(changed.values()))
            else:
                conn.execute(f"UPDATE {table} SET {column}=?", (changed[column],))
        assert dict(conn.execute(f"SELECT * FROM {table}").fetchone()) == row


@pytest.mark.parametrize("column", ["target_install_id", "route_generation", "status"])
def test_pending_metadata_growth_is_bounded_but_shrinking_is_allowed(source, monkeypatch, column):
    with rooms._transaction(source, immediate=True) as conn:
        records.prepare_delivery_locked(conn, room_id="room", target_install_id="target", route_generation="route",
                                        local_gateway_id=HOME, through_seq=1)
    monkeypatch.setattr(records, "MAX_STORE_BYTES", 1)
    with rooms._transaction(source, immediate=True) as conn:
        records.initialize(conn)
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            conn.execute(f"UPDATE {records.PENDING_TABLE} SET {column}=?", ("x" * 2000,))
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET status='acked'")
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET disposition='invalid'")
        assert conn.execute(f"SELECT status FROM {records.PENDING_TABLE}").fetchone()[0] == "acked"


@pytest.mark.parametrize("table", [records.SOURCE_TABLE, records.PENDING_TABLE])
def test_budget_credit_covers_only_the_exact_producer_and_participant(source, monkeypatch, table):
    with rooms._transaction(source, immediate=True) as conn:
        records.prepare_delivery_locked(conn, room_id="room", target_install_id="target", route_generation="route",
                                        local_gateway_id=HOME, through_seq=1)
        proposed = dict(conn.execute(f"SELECT * FROM {table}").fetchone())
        total, count = storage.usage_sql((*TABLES, storage.INVALID_TABLE))
        size, used_rows = conn.execute(f"SELECT {total},{count}").fetchone()
    monkeypatch.setattr(records, "MAX_STORE_BYTES", size)
    monkeypatch.setattr(records, "MAX_STORE_ROWS", used_rows)
    with rooms._transaction(source, immediate=True) as conn:
        records.initialize(conn)
        records._budget(conn, table, proposed)  # An exact replacement gets its own credit.
        conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(proposed)}) VALUES ({','.join('?' for _ in proposed)})",
                     tuple(proposed.values()))
        variants = [{"producer_gateway_id": "another"}, {"producer_epoch": 2}]
        if table == records.PENDING_TABLE:
            variants.append({"target_install_id": "another"})
        for change in variants:
            other = {**proposed, **change}
            with pytest.raises(records.WorkRecordCapacityError):
                records._budget(conn, table, other)
            with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
                conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(other)}) "
                             f"VALUES ({','.join('?' for _ in other)})", tuple(other.values()))
        assert dict(conn.execute(f"SELECT * FROM {table}").fetchone()) == proposed


@pytest.mark.parametrize("disposition", ["historical", "superseded_authority", "invalid"])
@pytest.mark.parametrize("column,value", [
    ("status", "acked"), ("route_generation", "changed"), ("target_install_id", "changed"),
    ("record_json", "{}"), ("digest", "changed"), ("producer_epoch", 8), ("disposition", "current"),
])
def test_a_frozen_old_pending_record_and_its_outcome_stay_immutable(copying, disposition, column, value):
    copying.pub._publish_one(KEY)
    with closing(open_sqlite(copying.source)) as conn, conn:
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET disposition=?", (disposition,))
        before = dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone())
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE {records.PENDING_TABLE} SET {column}=?", (value,))
        assert dict(conn.execute(f"SELECT * FROM {records.PENDING_TABLE}").fetchone()) == before


def test_the_routing_hint_preserves_legacy_rows_on_first_open(copying):
    copying.pub._publish_one(KEY)
    with rooms._transaction(copying.source) as conn:
        record = json.loads(conn.execute(f"SELECT record_json FROM {records.PENDING_TABLE}").fetchone()[0])
    install_legacy(copying.source, record)
    with closing(open_sqlite(copying.source)) as conn:
        before = retained(conn)
    copying.pub._select_route(copying.pub._load_route(KEY))
    with closing(open_sqlite(copying.source)) as conn:
        assert retained(conn) == before
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()


@pytest.mark.parametrize("outer", [False, True])
def test_the_pending_hint_classifies_only_the_asked_participant_and_respects_the_caller(copying, outer):
    copying.pub._publish_one(KEY)
    with rooms._transaction(copying.source, immediate=True) as conn:
        records.prepare_delivery_locked(conn, room_id="room", target_install_id="another-target",
                                        route_generation="other", local_gateway_id=HOME, through_seq=1)
        conn.execute(f"UPDATE {records.PENDING_TABLE} SET digest='wrong'")
    before = rows(copying.source, records.PENDING_TABLE)
    with closing(open_sqlite(copying.source)) as conn:
        if outer:
            conn.execute("BEGIN IMMEDIATE")
        assert not records.pending_delivery_is_anchored_locked(conn, room_id="room",
                                                               target_install_id="another-target", through_seq=1)
        assert conn.in_transaction is outer
        classified = [dict(row) for row in conn.execute(f"SELECT * FROM {records.PENDING_TABLE}")]
        assert next(r for r in classified if r["target_install_id"] == "another-target")["disposition"] == "invalid"
        assert next(r for r in classified if r["target_install_id"] != "another-target")["disposition"] == "current"
        if outer:
            conn.rollback()
    expected = before if outer else [{**row, "disposition": "invalid"} if row["target_install_id"] == "another-target"
                                     else row for row in before]
    assert rows(copying.source, records.PENDING_TABLE) == expected

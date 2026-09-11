"""Retained recovery evidence shares full-byte and original-row limits."""

import json
import sqlite3

import pytest

from gateway import hosted_rooms as rooms, hosted_room_work_records as records
from gateway import hosted_room_manual_recovery as recovery
from gateway.hosted_room_manual_promotion_schema import TABLE, initialize
from tests.gateway.test_hosted_room_replica_ingress import TARGET, pair
from tests.gateway.test_manual_group_promotion_staging import saved, saved_v2, stage


def proposed():
    return dict(room_id="another", revision=1, digest="digest", record_json="{}",
                producer_gateway_id=TARGET, producer_epoch=1, disposition="current")


def insert(conn, values, *, replace=False):
    conn.execute(f"INSERT {'OR REPLACE ' if replace else ''}INTO {records.SOURCE_TABLE} "
                 f"({','.join(values)}) VALUES({','.join('?' for _ in values)})", tuple(values.values()))


def expected_recovery_bytes(target):
    with rooms._transaction(target) as conn:
        view = recovery.prepare_recovery_locked(conn, room_id="room", target_gateway_id=TARGET, evidence=True)
    data = records.encode(view["evidence"])
    values = ("room", "owner-decision", view["snapshot_id"], view["source_authority"]["gateway_id"],
              view["source_authority"]["epoch"], TARGET, view["saved_through_seq"], data, 123.0)
    return 22 + sum(len(str(v).encode()) for v in values)


@pytest.mark.parametrize("limit", ["bytes", "rows"])
@pytest.mark.parametrize("order", ["work-first", "recovery-first"])
def test_exact_limit_transfer_keeps_charge_and_blocks_both_current_and_old_writers(saved, monkeypatch, limit, order):
    target = saved[1]
    size = expected_recovery_bytes(target)
    monkeypatch.setattr(records, "MAX_STORE_BYTES" if limit == "bytes" else "MAX_STORE_ROWS", size if limit == "bytes" else 1)
    stage(saved, now=123)
    rooms.create_room(target, room_id="another", name="Other group", authority_gateway_id=TARGET,
                      members=[{"profile": "default", "handle": "writer"}])
    with rooms._transaction(target, immediate=True) as conn:
        for init in ((records.initialize, initialize) if order == "work-first" else (initialize, records.initialize)):
            init(conn)
        data = conn.execute(f"SELECT work_record_json FROM {TABLE}").fetchone()[0]
        assert json.loads(data)["rows"][records.TARGET_TABLE]
        assert conn.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 0
        conn.execute(f"UPDATE {TABLE} SET work_record_json=?", (data,))
        with pytest.raises(records.WorkRecordCapacityError, match="storage is full"):
            records._budget(conn, records.SOURCE_TABLE, proposed())
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            insert(conn, proposed())
        assert conn.execute(f"SELECT COUNT(*) FROM {records.SOURCE_TABLE}").fetchone()[0] == 0


def test_old_writer_updates_and_replacements_cannot_hide_retained_bytes(saved, monkeypatch):
    target = saved[1]
    extra = proposed()
    # Lower disposition reservation plus every retained variable column.
    extra_size = 20 + sum(len(str(v).encode()) for k, v in extra.items() if k != "disposition")
    monkeypatch.setattr(records, "MAX_STORE_BYTES", expected_recovery_bytes(target) + extra_size)
    stage(saved, now=123)
    rooms.create_room(target, room_id="another", name="Other group", authority_gateway_id=TARGET,
                      members=[{"profile": "default", "handle": "writer"}])
    with rooms._transaction(target, immediate=True) as conn:
        insert(conn, extra)
        insert(conn, {**extra, "revision": 2}, replace=True)
        for field in ("record_json", "digest"):
            with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
                conn.execute(f"UPDATE {records.SOURCE_TABLE} SET {field}={field}||' '")
            with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
                insert(conn, {**extra, field: extra[field] + " "}, replace=True)
        assert tuple(conn.execute(f"SELECT revision,record_json FROM {records.SOURCE_TABLE}").fetchone()) == (2, "{}")


@pytest.mark.parametrize("limit", [2, 3])
def test_original_row_cardinality_survives_envelope(saved_v2, monkeypatch, limit):
    from gateway.hosted_room_work_storage import INVALID_TABLE
    target = saved_v2[1]
    with sqlite3.connect(target) as conn:
        conn.execute("DROP TRIGGER trg_work_invalid_insert")
        conn.execute(f"INSERT INTO {INVALID_TABLE}(source_table,room_id,revision,digest,record_json,disposition) VALUES(?, 'room',3,'opaque','PRIVATE','invalid')", (records.TARGET_TABLE,))
    fresh = recovery.prepare_recovery(target, room_id="room", target_gateway_id=TARGET)
    monkeypatch.setattr(records, "MAX_STORE_ROWS", limit)
    if limit == 2:
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            stage(saved_v2, snapshot_id=fresh["snapshot_id"])
        with sqlite3.connect(target) as conn:
            assert conn.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 2
            assert conn.execute(f"SELECT COUNT(*) FROM {INVALID_TABLE}").fetchone()[0] == 1
    else:
        stage(saved_v2, snapshot_id=fresh["snapshot_id"])
        rooms.create_room(target, room_id="another", name="Other", authority_gateway_id=TARGET,
                          members=[{"profile": "default", "handle": "writer"}])
        with rooms._transaction(target) as conn:
            with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
                insert(conn, proposed())


@pytest.mark.parametrize("damage", ["none", "type", "producer", "room", "bytes", "foreign_archive"])
def test_temporary_credit_is_exact_storage_and_target_scope(saved_v2, damage):
    from gateway.hosted_room_work_record_budget import duplicate_credit_sql
    from gateway.hosted_room_work_storage import INVALID_TABLE
    target = saved_v2[1]
    with rooms._transaction(target, immediate=True) as conn:
        initialize(conn)
        conn.execute("DROP TRIGGER trg_work_invalid_insert")
        conn.execute(f"INSERT INTO {INVALID_TABLE}(source_table,room_id,revision,digest,record_json,disposition) VALUES(?, 'room',3,'opaque','PRIVATE','invalid')", (records.SOURCE_TABLE,))
        view = recovery.prepare_recovery_locked(conn, room_id="room", target_gateway_id=TARGET, evidence=True)
        data = view["evidence"]
        if damage == "foreign_archive":
            data["rows"][INVALID_TABLE] = [dict(conn.execute(f"SELECT * FROM {INVALID_TABLE}").fetchone())]
        elif damage != "none":
            original = data["rows"][records.TARGET_TABLE][0]
            field, value = {"type": ("producer_epoch", str(original["producer_epoch"])),
                            "producer": ("producer_gateway_id", "unrelated"), "room": ("room_id", "other"),
                            "bytes": ("record_json", original["record_json"] + " ")}[damage]
            original[field] = value
        _bytes, count = duplicate_credit_sql("decision.")
        actual = conn.execute(f"SELECT {count} FROM (SELECT ? AS work_record_json,'transferring' AS status,'room' AS room_id) decision", (records.encode(data),)).fetchone()[0]
        assert actual == (2 if damage in {"none", "foreign_archive"} else 1)


def test_transferring_status_cannot_turn_duplicate_credit_into_quota_bypass(saved_v2, monkeypatch):
    target = saved_v2[1]
    monkeypatch.setattr(records, "MAX_STORE_ROWS", 2)
    with rooms._transaction(target, immediate=True) as conn:
        initialize(conn)
        view = recovery.prepare_recovery_locked(conn, room_id="room", target_gateway_id=TARGET, evidence=True)
        fields = "room_id,recovery_id,snapshot_id,source_gateway_id,source_epoch,target_gateway_id,history_seq,work_record_json,created_at,status"
        conn.execute("SAVEPOINT simulated_transfer")
        conn.execute(f"INSERT INTO {TABLE}({fields}) VALUES(?,?,?,?,?,?,?,?,?,?)", (
            "room", "owner-decision", view["snapshot_id"], view["source_authority"]["gateway_id"], 2,
            TARGET, view["saved_through_seq"], records.encode(view["evidence"]), 123.0, "transferring"))
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            conn.execute(f"UPDATE {TABLE} SET status='pending_reconciliation'")
        conn.execute("ROLLBACK TO simulated_transfer")
        conn.execute("RELEASE simulated_transfer")


def usage(conn):
    from gateway.hosted_room_work_record_budget import usage_sql
    total, count = usage_sql(conn)
    return tuple(conn.execute(f"SELECT {total},{count}").fetchone())


def retain_transfer(conn, view, *, status="transferring"):
    fields = "room_id,recovery_id,snapshot_id,source_gateway_id,source_epoch,target_gateway_id,history_seq,work_record_json,created_at,status"
    conn.execute(f"INSERT INTO {TABLE}({fields}) VALUES(?,?,?,?,?,?,?,?,?,?)", (
        "room", "cleanup-decision", view["snapshot_id"], view["source_authority"]["gateway_id"],
        view["source_authority"]["epoch"], TARGET, view["saved_through_seq"],
        records.encode(view["evidence"]), 123.0, status))


@pytest.mark.parametrize("limit", ["bytes", "rows"])
@pytest.mark.parametrize("operation", ["ack", "source-update", "source-replace", "pending-update",
                                       "pending-replace", "target-update", "target-replace"])
def test_overlimit_cleanup_preserves_non_growth(saved, monkeypatch, limit, operation):
    target = saved[1]
    if not operation.startswith("target"):
        stage(saved)
    rooms.create_room(target, room_id="another", name="Other", authority_gateway_id=TARGET,
                      members=[{"profile": "default", "handle": "writer"}])
    with rooms._transaction(target, immediate=True) as conn:
        record = records.prepare_delivery_locked(conn, room_id="another", target_install_id="install:peer",
            route_generation="route", local_gateway_id=TARGET, through_seq=0)
        assert record is not None
        conn.execute("SAVEPOINT cleanup")
        if operation.startswith("target"):
            initialize(conn)
            view = recovery.prepare_recovery_locked(conn, room_id="room", target_gateway_id=TARGET, evidence=True)
            # Retain the envelope with no temporary credit. The deferred namespace
            # reference stays entirely inside this rolled-back SQLite savepoint.
            retain_transfer(conn, view, status="pending_reconciliation")
        table = {"ack": records.PENDING_TABLE, "source": records.SOURCE_TABLE,
                 "pending": records.PENDING_TABLE, "target": records.TARGET_TABLE}[operation.split("-")[0]]
        conn.execute(f"UPDATE {table} SET record_json=record_json||'  '")
        row = dict(conn.execute(f"SELECT * FROM {table}").fetchone())
        before = usage(conn)
        bound = before[0] - 100 if limit == "bytes" else before[1] - 1
        monkeypatch.setattr(records, "MAX_STORE_BYTES" if limit == "bytes" else "MAX_STORE_ROWS", bound)
        records.initialize(conn)
        if operation == "ack":
            assert records.delivery_status_locked(conn, room_id="another", target_install_id="install:peer",
                route_generation="route", record=record, status="acked")
            field = "status"
            expected = "acked"
        else:
            field = "record_json"
            expected = row[field][:-2]
            if operation.endswith("replace"):
                replacement = {**row, field: expected}
                conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(replacement)}) "
                    f"VALUES ({','.join('?' for _ in replacement)})", tuple(replacement.values()))
            else:
                conn.execute(f"UPDATE {table} SET {field}=?", (expected,))
        assert usage(conn) == (before[0] - 2, before[1])
        assert usage(conn)[0 if limit == "bytes" else 1] > bound
        assert conn.execute(f"SELECT {field} FROM {table}").fetchone()[0] == expected
        # Neither unchanged payload nor a full row store exempts metadata growth.
        after = usage(conn)
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            conn.execute(f"UPDATE {table} SET digest=digest||' '")
        changed = dict(conn.execute(f"SELECT * FROM {table}").fetchone())
        changed["digest"] += " "
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(changed)}) "
                f"VALUES ({','.join('?' for _ in changed)})", tuple(changed.values()))
        assert usage(conn) == after
        conn.execute("ROLLBACK TO cleanup")
        conn.execute("RELEASE cleanup")


@pytest.mark.parametrize("limit", ["bytes", "rows"])
@pytest.mark.parametrize("change", ["status", "shrink-update", "shrink-replace", "disposition"])
def test_overlimit_credit_loss_is_effective_growth(saved, monkeypatch, limit, change):
    with rooms._transaction(saved[1], immediate=True) as conn:
        initialize(conn)
        conn.execute(f"UPDATE {records.TARGET_TABLE} SET record_json=record_json||'  '")
        view = recovery.prepare_recovery_locked(conn, room_id="room", target_gateway_id=TARGET, evidence=True)
        conn.execute("SAVEPOINT credit_loss")
        retain_transfer(conn, view)
        before = usage(conn)
        monkeypatch.setattr(records, "MAX_STORE_BYTES" if limit == "bytes" else "MAX_STORE_ROWS",
                            before[0] - 1 if limit == "bytes" else before[1] - 1)
        records.initialize(conn)
        row = dict(conn.execute(f"SELECT * FROM {records.TARGET_TABLE}").fetchone())
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            if change == "status":
                conn.execute(f"UPDATE {TABLE} SET status='pending_reconciliation'")
            elif change == "disposition":
                conn.execute(f"UPDATE {records.TARGET_TABLE} SET disposition='historical'")
            elif change == "shrink-update":
                conn.execute(f"UPDATE {records.TARGET_TABLE} SET record_json=?", (row["record_json"][:-2],))
            else:
                row["record_json"] = row["record_json"][:-2]
                conn.execute(f"INSERT OR REPLACE INTO {records.TARGET_TABLE} ({','.join(row)}) "
                    f"VALUES ({','.join('?' for _ in row)})", tuple(row.values()))
        assert usage(conn) == before
        assert conn.execute(f"SELECT status FROM {TABLE}").fetchone()[0] == "transferring"
        assert conn.execute(f"SELECT disposition FROM {records.TARGET_TABLE}").fetchone()[0] == "current"
        conn.execute("ROLLBACK TO credit_loss")
        conn.execute("RELEASE credit_loss")


def test_envelope_overhead_is_not_free(saved, monkeypatch):
    target = saved[1]
    monkeypatch.setattr(records, "MAX_STORE_BYTES", expected_recovery_bytes(target) - 1)
    with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
        stage(saved, now=123)
    with sqlite3.connect(target) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hosted_rooms").fetchone()[0] == 0

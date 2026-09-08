"""Retained recovery evidence shares full-byte and original-row limits."""

import json
import sqlite3

import pytest

from gateway import hosted_rooms as rooms, hosted_room_work_records as records
from gateway import hosted_room_manual_recovery as recovery
from gateway.hosted_room_manual_promotion_schema import TABLE, initialize
from tests.gateway.test_hosted_room_replica_ingress import TARGET, pair
from tests.gateway.test_manual_group_promotion_staging import saved, stage


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


def test_envelope_overhead_is_not_free(saved, monkeypatch):
    target = saved[1]
    monkeypatch.setattr(records, "MAX_STORE_BYTES", expected_recovery_bytes(target) - 1)
    with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
        stage(saved, now=123)
    with sqlite3.connect(target) as conn:
        assert conn.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM hosted_rooms").fetchone()[0] == 0

"""Retained recovery evidence shares the existing bounded metadata store."""

import sqlite3

import pytest

from gateway import hosted_rooms as rooms, hosted_room_work_records as records
from gateway.hosted_room_manual_promotion_schema import TABLE
from tests.gateway.test_hosted_room_replica_ingress import TARGET, pair
from tests.gateway.test_manual_group_promotion_staging import saved, stage


def copied_record(target):
    with sqlite3.connect(target) as conn:
        return conn.execute(f"SELECT record_json FROM {records.TARGET_TABLE} WHERE room_id='room'").fetchone()[0]


@pytest.mark.parametrize("limit", ["bytes", "rows"])
def test_exact_limit_transfer_keeps_charge_and_blocks_both_current_and_old_writers(saved, monkeypatch, limit):
    target = saved[1]
    data = copied_record(target)
    if limit == "bytes":
        monkeypatch.setattr(records, "MAX_STORE_BYTES", len(data.encode("utf-8")))
    else:
        monkeypatch.setattr(records, "MAX_STORE_ROWS", 1)
    stage(saved)
    rooms.create_room(target, room_id="another", name="Other group", authority_gateway_id=TARGET,
                      members=[{"profile": "default", "handle": "writer"}])
    with rooms._transaction(target, immediate=True) as conn:
        records.initialize(conn)
        assert conn.execute(f"SELECT work_record_json FROM {TABLE} WHERE room_id='room'").fetchone()[0] == data
        assert conn.execute(f"SELECT COUNT(*) FROM {records.TARGET_TABLE}").fetchone()[0] == 0
        # Replacing the same retained record must receive its existing charge.
        records._budget(conn, TABLE, "room", data)
        conn.execute(f"UPDATE {TABLE} SET work_record_json=? WHERE room_id='room'", (data,))
        with pytest.raises(records.WorkRecordCapacityError, match="storage is full"):
            records._budget(conn, records.SOURCE_TABLE, "another", "{}")
        # An older process bypassing the Python accountant still hits the SQL guard.
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            conn.execute(f"INSERT INTO {records.SOURCE_TABLE} VALUES('another',1,'digest','{{}}')")
        assert conn.execute(f"SELECT COUNT(*) FROM {records.SOURCE_TABLE}").fetchone()[0] == 0


def test_old_writer_updates_and_replacements_cannot_hide_retained_bytes(saved, monkeypatch):
    target = saved[1]
    data = copied_record(target)
    monkeypatch.setattr(records, "MAX_STORE_BYTES", len(data.encode("utf-8")) + len("{}"))
    stage(saved)
    rooms.create_room(target, room_id="another", name="Other group", authority_gateway_id=TARGET,
                      members=[{"profile": "default", "handle": "writer"}])
    with rooms._transaction(target, immediate=True) as conn:
        conn.execute(f"INSERT INTO {records.SOURCE_TABLE} VALUES('another',1,'digest','{{}}')")
        conn.execute(f"INSERT OR REPLACE INTO {records.SOURCE_TABLE} VALUES('another',2,'digest','{{}}')")
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            conn.execute(f"UPDATE {records.SOURCE_TABLE} SET record_json='{{ }}' WHERE room_id='another'")
        with pytest.raises(sqlite3.IntegrityError, match="storage is full"):
            conn.execute(f"INSERT OR REPLACE INTO {records.SOURCE_TABLE} VALUES('another',3,'digest','{{ }}')")
        assert tuple(conn.execute(f"SELECT revision,record_json FROM {records.SOURCE_TABLE}").fetchone()) == (2, "{}")
        assert conn.execute(f"SELECT work_record_json FROM {TABLE}").fetchone()[0] == data

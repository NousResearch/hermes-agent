"""Public first-opener, metadata capacity and invalid-send regressions."""
import json
import sqlite3

import pytest

from gateway import hosted_rooms as rooms
from gateway import hosted_room_work_records as work
from gateway import hosted_room_work_storage as storage
from tests.gateway.test_hosted_room_work_records import source, capture, HOME  # noqa: F401
from tests.tui_gateway.test_hosted_room_replication import pair, KEY  # noqa: F401
from tests.tui_gateway.test_work_record_delivery_fairness import copying  # noqa: F401
from tui_gateway.hosted_room_replication import HostedRoomReplicationPublisher


def install_legacy_source(db, record, *, metadata_bytes=0, invalid=False):
    """Reproduce populated c1-shaped tables, retaining parent cleanup triggers."""
    data = '{opaque invalid original' if invalid else json.dumps(record, indent=2)
    with sqlite3.connect(db) as raw:
        for table in (work.SOURCE_TABLE, work.TARGET_TABLE, work.PENDING_TABLE):
            raw.execute(f'DROP TABLE {table}')
        for table in (work.SOURCE_TABLE, work.TARGET_TABLE):
            raw.execute(f'''CREATE TABLE {table} (room_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,digest TEXT NOT NULL,record_json TEXT NOT NULL)''')
        raw.execute(f'''CREATE TABLE {work.PENDING_TABLE} (
            room_id TEXT NOT NULL,target_install_id TEXT NOT NULL,route_generation TEXT NOT NULL,
            revision INTEGER NOT NULL,digest TEXT NOT NULL,record_json TEXT NOT NULL,status TEXT NOT NULL,
            PRIMARY KEY(room_id,target_install_id))''')
        raw.execute(f'INSERT INTO {work.SOURCE_TABLE} VALUES (?,?,?,?)',
                    ('room', record['revision'], record['digest'], data))
        raw.execute(f'INSERT INTO {work.PENDING_TABLE} VALUES (?,?,?,?,?,?,?)',
                    ('room', 'target', 'x' * metadata_bytes or 'legacy-route',
                     record['revision'], record['digest'], data, 'unavailable'))
    return data


def test_public_status_must_not_drop_legacy_evidence(source):
    record = capture(source)
    original = install_legacy_source(source, record)
    publisher = HostedRoomReplicationPublisher(source)
    publisher.status('room')
    with sqlite3.connect(source) as raw:
        after_first = {t: raw.execute(f'SELECT record_json FROM {t}').fetchall()
                       for t in (work.SOURCE_TABLE, work.PENDING_TABLE)}
    publisher.status('room')
    with sqlite3.connect(source) as raw:
        after_second = {t: raw.execute(f'SELECT record_json FROM {t}').fetchall()
                        for t in (work.SOURCE_TABLE, work.PENDING_TABLE)}
    print('STATUS MIGRATION ROWS', {t: len(v) for t, v in after_first.items()},
          {t: len(v) for t, v in after_second.items()})
    assert after_first == {work.SOURCE_TABLE: [(original,)], work.PENDING_TABLE: [(original,)]}
    assert after_second == after_first


def test_parent_backed_invalid_metadata_exhausts_shared_budget(source):
    record = capture(source)
    original = install_legacy_source(source, record, metadata_bytes=work.MAX_STORE_BYTES, invalid=True)
    with rooms._transaction(source, immediate=True) as conn:
        work.initialize(conn)
        row = conn.execute(f'SELECT * FROM {work.PENDING_TABLE}').fetchone()
        assert row['disposition'] == 'invalid'
        assert row['record_json'] == original
        assert len(row['route_generation'].encode()) == work.MAX_STORE_BYTES
        total, count = storage.usage_sql((work.SOURCE_TABLE, work.TARGET_TABLE, work.PENDING_TABLE, storage.INVALID_TABLE))
        charged = tuple(conn.execute(f'SELECT {total}, {count}').fetchone())
        print('PARENT-BACKED INVALID CHARGED', charged, 'RETAINED ROUTE BYTES', work.MAX_STORE_BYTES)
    with pytest.raises(work.WorkRecordCapacityError):
        capture(source)


def test_invalid_pending_metadata_is_not_sent_by_real_publisher(copying):
    copying.pub._publish_one(KEY)  # Freeze current record and copy its anchor.
    assert copying.records == []
    with sqlite3.connect(copying.source) as raw:
        raw.execute(f"UPDATE {work.PENDING_TABLE} SET digest='wrong' WHERE room_id='room'")
    assert copying.pub.status('room')['work_records'][0]['disposition'] == 'invalid'
    for _ in range(3):
        copying.pub._publish_one(KEY)
    status = copying.pub.status('room')
    print('INVALID PENDING SENDS', len(copying.records),
          'ROUTE', status['routes'][0]['work_record_status'],
          'PENDING', status['work_records'][0]['status'])
    assert copying.records == [], 'metadata-invalid pending evidence must fail closed before HTTP'

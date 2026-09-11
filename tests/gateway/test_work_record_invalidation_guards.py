"""One-way quarantine invalidation is not permission to rewrite retained evidence."""
import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_work_records as work
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import open_sqlite
from tests.tui_gateway.test_hosted_room_replication import pair, KEY  # noqa: F401
from tests.tui_gateway.test_work_record_delivery_fairness import copying  # noqa: F401


def quarantined(copying, *, historical=False, opaque=None):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    with rooms._transaction(copying.target, immediate=True) as conn:
        # Model an additive future metadata column; an exception must protect it
        # too, including NULL and BLOB, not just today's payload field names.
        conn.execute(f'ALTER TABLE {work.TARGET_TABLE} ADD COLUMN opaque_metadata')
        conn.execute(f'UPDATE {work.TARGET_TABLE} SET opaque_metadata=?', (opaque,))
        if historical:
            conn.execute(f"UPDATE {work.TARGET_TABLE} SET disposition='historical'")
        conn.execute("UPDATE hosted_room_replica_events SET payload_json='not-json'")
    assert replicas.replica_state(copying.target, room_id='room')['safety_status'] == 'quarantined'
    with closing(open_sqlite(copying.target)) as conn:
        return dict(conn.execute(f'SELECT * FROM {work.TARGET_TABLE}').fetchone())


@pytest.mark.parametrize('opaque', [None, b'\x00\xffmetadata'])
def test_raw_quarantine_invalidation_is_one_way_and_byte_preserving(copying, opaque):
    before = quarantined(copying, opaque=opaque)
    with closing(open_sqlite(copying.target)) as conn, conn:
        conn.execute(f"UPDATE {work.TARGET_TABLE} SET disposition='invalid'")
        assert dict(conn.execute(f'SELECT * FROM {work.TARGET_TABLE}').fetchone()) == {**before, 'disposition': 'invalid'}
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE {work.TARGET_TABLE} SET disposition='current'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE {work.TARGET_TABLE} SET disposition='historical'")
    assert replicas.replica_state(copying.target, room_id='room')['safety_status'] == 'quarantined'


@pytest.mark.parametrize('historical', [False, True])
@pytest.mark.parametrize('column,value', [
    ('room_id', 'other'), ('producer_gateway_id', 'other'), ('producer_epoch', 2),
    ('revision', 99), ('digest', 'rewritten'), ('record_json', '{}'),
    ('opaque_metadata', b'new\x00bytes'), ('rowid', 99),
])
def test_invalidation_cannot_smuggle_any_other_column_change(copying, historical, column, value):
    before = quarantined(copying, historical=historical)
    with closing(open_sqlite(copying.target)) as conn:
        for disposition in ('invalid', before['disposition']):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f'UPDATE {work.TARGET_TABLE} SET disposition=?,{column}=?', (disposition, value))
            assert dict(conn.execute(f'SELECT * FROM {work.TARGET_TABLE}').fetchone()) == before
        for operation in ('INSERT', 'INSERT OR REPLACE'):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"{operation} INTO {work.TARGET_TABLE} ({','.join(before)}) VALUES ({','.join('?' for _ in before)})", tuple(before.values()))
        if historical:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"UPDATE {work.TARGET_TABLE} SET disposition='invalid'")


@pytest.mark.parametrize('disposition', ['historical', 'superseded_authority', 'invalid'])
@pytest.mark.parametrize('column,value', [
    ('status', 'acked'), ('route_generation', 'changed'), ('target_install_id', 'changed'),
    ('record_json', '{}'), ('digest', 'changed'), ('producer_epoch', 8), ('disposition', 'current'),
])
def test_historical_pending_payload_scope_and_outcome_remain_immutable(copying, disposition, column, value):
    copying.pub._publish_one(KEY)
    with closing(open_sqlite(copying.source)) as conn, conn:
        conn.execute(f'UPDATE {work.PENDING_TABLE} SET disposition=?', (disposition,))
        before = dict(conn.execute(f'SELECT * FROM {work.PENDING_TABLE}').fetchone())
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f'UPDATE {work.PENDING_TABLE} SET {column}=?', (value,))
        assert dict(conn.execute(f'SELECT * FROM {work.PENDING_TABLE}').fetchone()) == before


def test_retirement_removes_evidence_and_invalidation_cannot_resurrect_it(copying):
    from gateway import hosted_room_link_records as links
    from gateway import hosted_room_replica_retirement as retirement
    from gateway import hosted_room_work_storage as storage
    from tests.tui_gateway.test_hosted_room_replication import SECRET, TARGET
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    with closing(open_sqlite(copying.target)) as conn:
        old = dict(conn.execute(f'SELECT * FROM {work.TARGET_TABLE}').fetchone())
    entry = retirement.prepare_home_enrollment(copying.source, room_id='room', target_install_id=TARGET,
        endpoint='http://127.0.0.1:9876', local_gateway_id=copying.pub.local_id, secret=SECRET)
    retirement.enroll_target(copying.target, enrollment=entry, target_install_id=TARGET)
    members = rooms.room_state(copying.source, room_id='room')['members']
    scope = dict(room_id='room', authority_gateway_id=copying.pub.local_id, authority_epoch=1)
    links.begin_room_link_retirement(copying.source, **scope)
    links.complete_room_link_retirement(copying.source, **scope)
    rooms.disband_room(copying.source, room_id='room', expected_gateway_id=copying.pub.local_id, expected_epoch=1)
    notice = retirement.materialize_notice(copying.source, enrollment_id=entry['enrollment_id'],
        local_gateway_id=copying.pub.local_id, secret_loader=lambda: SECRET)
    assert retirement.retire_copy(copying.target, payload=notice.payload(), value=notice.value, local_gateway_id=TARGET)['retired']
    rooms.create_room(copying.source, room_id='active', name='Active', members=members, authority_gateway_id=copying.pub.local_id)
    active = work.capture(copying.source, room_id='active', local_gateway_id=copying.pub.local_id)
    replicas.ingest_page(copying.target, room_id='active', room_name='Active', members=members,
                         page=rooms.read_events(copying.source, room_id='active'))
    with rooms._transaction(copying.target, immediate=True) as conn:
        work.initialize(conn)
        storage.save_locked(conn, work.TARGET_TABLE, active)
    with closing(open_sqlite(copying.target)) as conn:
        assert conn.execute(f"SELECT * FROM {work.TARGET_TABLE} WHERE room_id='room'").fetchone() is None
        before = dict(conn.execute(f'SELECT * FROM {work.TARGET_TABLE}').fetchone())
        for disposition in ('current', 'invalid', 'historical'):
            proposed = {**old, 'disposition': disposition}
            with pytest.raises(sqlite3.IntegrityError, match='retired'):
                conn.execute(f"INSERT INTO {work.TARGET_TABLE} ({','.join(proposed)}) VALUES ({','.join('?' for _ in proposed)})", tuple(proposed.values()))
            with pytest.raises(sqlite3.IntegrityError, match='retired'):
                conn.execute(f"UPDATE {work.TARGET_TABLE} SET room_id='room',disposition=? WHERE room_id='active'", (disposition,))
            assert dict(conn.execute(f'SELECT * FROM {work.TARGET_TABLE}').fetchone()) == before
    assert replicas.replica_state(copying.target, room_id='room')['safety_status'] == 'retired'

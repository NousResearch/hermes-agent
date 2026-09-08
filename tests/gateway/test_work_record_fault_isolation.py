"""Record-owned invalidity must not couple independent rooms or rewrite history."""
import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_work_records as work
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.test_hosted_room_work_records import HOME, MEMBERS
from tests.tui_gateway.test_hosted_room_replication import pair, KEY  # noqa: F401
from tests.tui_gateway.test_work_record_delivery_fairness import copying  # noqa: F401
from tui_gateway import hosted_room_replication as publishing


def rows(path, table):
    with closing(open_sqlite(path)) as conn:
        return [dict(row) for row in conn.execute(f'SELECT * FROM {table} ORDER BY room_id,producer_epoch')]


def damage_copy(copying):
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    assert copying.records
    with sqlite3.connect(copying.target) as conn:
        conn.execute(f"UPDATE {work.TARGET_TABLE} SET digest='wrong' WHERE room_id='room'")
        conn.execute("UPDATE hosted_room_replica_events SET payload_json='not-json' WHERE room_id='room'")
    return rows(copying.target, work.TARGET_TABLE)


def test_invalid_quarantined_replica_cannot_block_unrelated_source_capture(copying):
    before_copy = damage_copy(copying)
    members = rooms.room_state(copying.source, room_id='room')['members']
    rooms.create_room(copying.target, room_id='healthy', name='Unrelated source', members=members,
                      authority_gateway_id=HOME)
    # The real history auditor must classify this requested copy, not a future
    # unrelated capture. Preserve its damaged bytes and actual quarantine.
    state = replicas.replica_state(copying.target, room_id='room')
    assert state['safety_status'] == 'quarantined'
    before = work.capture(copying.target, room_id='healthy', local_gateway_id=HOME)
    after = work.capture(copying.target, room_id='healthy', local_gateway_id=HOME)
    status = publishing.HostedRoomReplicationPublisher(copying.target).status('healthy')
    assert status['error'] is None
    assert after == before
    assert rows(copying.target, work.TARGET_TABLE) == [{**before_copy[0], 'disposition': 'invalid'}]
    assert replicas.replica_state(copying.target, room_id='room')['safety_status'] == 'quarantined'


def test_initialization_does_not_validate_or_reclassify_unrelated_rows(copying):
    damaged = damage_copy(copying)
    # No replica_state call: ordinary work schema initialization has no audit ownership.
    with rooms._transaction(copying.target, immediate=True) as conn:
        work.initialize(conn)
    assert rows(copying.target, work.TARGET_TABLE) == damaged
    state = replicas.replica_state(copying.target, room_id='room')
    assert state['safety_status'] == 'quarantined'
    assert rows(copying.target, work.TARGET_TABLE) == [{**damaged[0], 'disposition': 'invalid'}]


def test_unrelated_invalid_source_pending_and_history_do_not_block_delivery(copying):
    for room_id in ('invalid-current', 'invalid-history'):
        rooms.create_room(copying.source, room_id=room_id, name=room_id, members=MEMBERS,
                          authority_gateway_id=HOME)
        with rooms._transaction(copying.source, immediate=True) as conn:
            work.prepare_delivery_locked(conn, room_id=room_id, target_install_id='elsewhere',
                route_generation='frozen-route', local_gateway_id=HOME, through_seq=0)
    # Damage only after all setup so no initializer can pre-classify the fixtures.
    with rooms._transaction(copying.source, immediate=True) as conn:
        for room_id in ('invalid-current', 'invalid-history'):
            for table in (work.SOURCE_TABLE, work.PENDING_TABLE):
                conn.execute(f"UPDATE {table} SET digest='damaged' WHERE room_id=?", (room_id,))
                if room_id == 'invalid-history':
                    conn.execute(f"UPDATE {table} SET disposition='historical' WHERE room_id=?", (room_id,))
    before = {table: rows(copying.source, table) for table in (work.SOURCE_TABLE, work.PENDING_TABLE)}
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    assert copying.records
    assert copying.pub.status('room')['work_records'][0]['status'] == 'acked'
    for table in before:
        assert [row for row in rows(copying.source, table) if row['room_id'] != 'room'] == before[table]
    # A broad read projects each bad row independently, without rewriting history
    # or making the healthy route disappear.
    all_status = copying.pub.status()
    assert all_status['error'] is None
    for table in before:
        assert [row for row in rows(copying.source, table) if row['room_id'] != 'room'] == before[table]
    assert next(row for row in all_status['work_records'] if row['room_id'] == 'room')['status'] == 'acked'
    bad = [row for row in all_status['work_records'] if row['room_id'].startswith('invalid-')]
    assert all(row['disposition'] == 'invalid' for row in bad)
    historical = next(row for row in rows(copying.source, work.PENDING_TABLE) if row['room_id'] == 'invalid-history')
    assert historical == next(row for row in before[work.PENDING_TABLE] if row['room_id'] == 'invalid-history')


def test_healthy_delivery_continues_beside_requested_quarantined_copy(copying, tmp_path):
    from gateway import hosted_room_work_storage as storage
    bad_source = tmp_path / 'other-owner.db'
    rooms.create_room(bad_source, room_id='bad-copy', name='Other owner', members=MEMBERS,
                      authority_gateway_id=HOME)
    rooms.append_event(bad_source, room_id='bad-copy', event_id='event', kind='message.user',
        actor={'kind': 'user', 'id': 'owner'}, payload={'text': 'copied'}, authority_gateway_id=HOME, authority_epoch=1)
    record = work.capture(bad_source, room_id='bad-copy', local_gateway_id=HOME)
    replicas.ingest_page(copying.source, room_id='bad-copy', room_name='Other owner', members=MEMBERS,
                         page=rooms.read_events(bad_source, room_id='bad-copy'))
    with rooms._transaction(copying.source, immediate=True) as conn:
        work.initialize(conn)
        storage.save_locked(conn, work.TARGET_TABLE, record)
        conn.execute(f"UPDATE {work.TARGET_TABLE} SET digest='wrong' WHERE room_id='bad-copy'")
        conn.execute("UPDATE hosted_room_replica_events SET payload_json='not-json' WHERE room_id='bad-copy'")
    before = rows(copying.source, work.TARGET_TABLE)
    assert replicas.replica_state(copying.source, room_id='bad-copy')['safety_status'] == 'quarantined'
    # The healthy canonical producer and the quarantined target share this DB.
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    assert copying.records
    assert copying.pub.status('room')['work_records'][0]['status'] == 'acked'
    assert rows(copying.source, work.TARGET_TABLE) == [{**before[0], 'disposition': 'invalid'}]
    assert replicas.replica_state(copying.source, room_id='bad-copy')['safety_status'] == 'quarantined'


@pytest.mark.parametrize('outer', [False, True])
def test_pending_hint_classifies_only_requested_target_and_respects_transaction(copying, outer):
    copying.pub._publish_one(KEY)
    with rooms._transaction(copying.source, immediate=True) as conn:
        work.prepare_delivery_locked(conn, room_id='room', target_install_id='another-target',
            route_generation='other', local_gateway_id=copying.pub.local_id, through_seq=1)
        conn.execute(f"UPDATE {work.PENDING_TABLE} SET digest='wrong'")
    before = rows(copying.source, work.PENDING_TABLE)
    with closing(open_sqlite(copying.source)) as conn:
        if outer:
            conn.execute('BEGIN IMMEDIATE')
        assert not work.pending_delivery_is_anchored_locked(conn, room_id='room',
            target_install_id='another-target', through_seq=1)
        assert conn.in_transaction is outer
        classified = [dict(row) for row in conn.execute(f'SELECT * FROM {work.PENDING_TABLE}')]
        assert next(row for row in classified if row['target_install_id'] == 'another-target')['disposition'] == 'invalid'
        assert next(row for row in classified if row['target_install_id'] != 'another-target')['disposition'] == 'current'
        if outer:
            conn.rollback()
    expected = before if outer else [{**row, 'disposition': 'invalid'} if row['target_install_id'] == 'another-target' else row for row in before]
    assert rows(copying.source, work.PENDING_TABLE) == expected


def test_unrelated_quarantined_old_scope_is_not_globally_reconciled(copying):
    from gateway import hosted_room_replica_retirement as retirement
    from tests.gateway.test_hosted_room_replica_lineage import SUCCESSOR, transfer
    from tests.tui_gateway.test_hosted_room_replication import SECRET, TARGET
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    old = rows(copying.target, work.TARGET_TABLE)[0]
    with sqlite3.connect(copying.target) as conn:
        conn.execute(f'DELETE FROM {work.TARGET_TABLE}')
    transfer(copying.source, copying.pub.local_id)
    entry = retirement.prepare_home_enrollment(copying.source, room_id='room', target_install_id=TARGET,
        endpoint='http://127.0.0.1:9876', local_gateway_id=SUCCESSOR, secret=SECRET)
    retirement.enroll_target(copying.target, enrollment=entry, target_install_id=TARGET,
        **retirement.home_enrollment_history(copying.source, enrollment_id=entry['enrollment_id']))
    # A late older writer inserts a valid old scope after the transition trigger.
    with sqlite3.connect(copying.target) as conn:
        conn.execute(f"INSERT INTO {work.TARGET_TABLE} ({','.join(old)}) VALUES ({','.join('?' for _ in old)})", tuple(old.values()))
        conn.execute("UPDATE hosted_room_replica_events SET payload_json='not-json'")
    assert replicas.replica_state(copying.target, room_id='room')['safety_status'] == 'quarantined'
    with rooms._transaction(copying.target, immediate=True) as conn:
        work.initialize(conn)
        summary = work.summary_locked(conn, 'room')
        assert summary['scopes'][0]['disposition'] == 'historical'
    # Read-only scope projection cannot turn into a forbidden historical UPDATE
    # merely because an unrelated operation initializes the work store.
    assert rows(copying.target, work.TARGET_TABLE) == [old]
    rooms.create_room(copying.target, room_id='healthy', name='Healthy', members=MEMBERS, authority_gateway_id=HOME)
    assert work.capture(copying.target, room_id='healthy', local_gateway_id=HOME)['room_id'] == 'healthy'
    assert rows(copying.target, work.TARGET_TABLE) == [old]


@pytest.mark.parametrize('outcome,disposition', [('pending', 'superseded_authority'), ('acked', 'historical')])
def test_legacy_migration_resolves_old_scope_at_insertion(copying, outcome, disposition):
    from tests.gateway.test_work_record_correction_boundaries import install_legacy_source
    from tests.gateway.test_hosted_room_replica_lineage import transfer
    record = work.capture(copying.source, room_id='room', local_gateway_id=copying.pub.local_id)
    transfer(copying.source, copying.pub.local_id)
    original = install_legacy_source(copying.source, record)
    with sqlite3.connect(copying.source) as conn:
        conn.execute(f'UPDATE {work.PENDING_TABLE} SET status=?', (outcome,))
    with rooms._transaction(copying.source, immediate=True) as conn:
        work.initialize(conn)
    source = rows(copying.source, work.SOURCE_TABLE)[0]
    pending = rows(copying.source, work.PENDING_TABLE)[0]
    assert source['disposition'] == 'historical'
    assert pending['disposition'] == disposition and pending['status'] == outcome
    assert source['record_json'] == pending['record_json'] == original


def test_requested_audit_projects_invalid_history_without_rewriting_it(copying):
    damage_copy(copying)
    with sqlite3.connect(copying.target) as conn:
        conn.execute(f"UPDATE {work.TARGET_TABLE} SET disposition='historical'")
    before = rows(copying.target, work.TARGET_TABLE)
    assert replicas.replica_state(copying.target, room_id='room')['safety_status'] == 'quarantined'
    with closing(open_sqlite(copying.target)) as conn:
        item = work.summary_locked(conn, 'room')['scopes'][0]
        assert item['availability'] == 'invalid' and item['disposition'] == 'historical'
        assert item['incompleteness'] == ['invalid_work_evidence']
        assert 'tasks' not in item
    assert rows(copying.target, work.TARGET_TABLE) == before


def test_stale_examined_row_cannot_invalidate_its_valid_replacement(copying):
    from gateway import hosted_room_work_storage as storage
    copying.pub._publish_one(KEY)
    with closing(open_sqlite(copying.source)) as conn, conn:
        original = conn.execute(f'SELECT * FROM {work.PENDING_TABLE}').fetchone()
        conn.execute(f"UPDATE {work.PENDING_TABLE} SET digest='wrong'")
        stale = conn.execute(f'SELECT * FROM {work.PENDING_TABLE}').fetchone()
        conn.execute(f'UPDATE {work.PENDING_TABLE} SET digest=?', (original['digest'],))
    with closing(open_sqlite(copying.source)) as conn:
        with pytest.raises(work.InvalidStoredWorkRecord):
            storage.validate_stored_locked(conn, work.PENDING_TABLE, stale)
    assert rows(copying.source, work.PENDING_TABLE) == [dict(original)]

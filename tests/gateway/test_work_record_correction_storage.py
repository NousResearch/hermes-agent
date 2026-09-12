"""First-opener atomicity, complete budget and stored-binding siblings."""
import json
import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_rooms as rooms
from gateway import hosted_room_work_records as work
from gateway.hosted_rooms_common import open_sqlite
from tests.gateway.test_hosted_room_work_records import source, capture, HOME  # noqa: F401
from tests.gateway.test_work_record_correction_boundaries import install_legacy_source
from tests.gateway.test_work_record_migration_closure import legacy_database
from tests.tui_gateway.test_hosted_room_replication import pair, KEY  # noqa: F401
from tests.tui_gateway.test_work_record_delivery_fairness import copying  # noqa: F401


def retained(conn):
    return {t: conn.execute(f'SELECT room_id,revision,digest,record_json FROM {t}').fetchall()
            for t in (work.SOURCE_TABLE, work.TARGET_TABLE, work.PENDING_TABLE)}


@pytest.mark.parametrize('outer', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_first_initializer_is_atomic_and_joins_outer_transaction(source, outer, failure):
    install_legacy_source(source, capture(source))
    with closing(open_sqlite(source)) as conn:
        original = retained(conn)
        schema = conn.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
        if outer:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute("UPDATE hosted_rooms SET name='caller write'")
        copied = []
        def deny(action, first, second, database, origin):
            if action == sqlite3.SQLITE_INSERT and first == work.SOURCE_TABLE:
                copied.append(first)
            if failure and action == sqlite3.SQLITE_INSERT and first == work.PENDING_TABLE:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        conn.set_authorizer(deny)
        if failure:
            with pytest.raises(sqlite3.DatabaseError):
                work.initialize(conn)
        else:
            work.initialize(conn)
        conn.set_authorizer(None)
        assert copied
        assert conn.in_transaction is outer
        if outer:
            assert conn.execute('SELECT name FROM hosted_rooms').fetchone()[0] == 'caller write'
            if failure:
                assert retained(conn) == original
                assert conn.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall() == schema
            conn.rollback()
        assert retained(conn) == original
        if failure or outer:
            assert conn.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall() == schema
    with closing(open_sqlite(source)) as conn:
        assert retained(conn) == original
        conn.execute('BEGIN IMMEDIATE')  # no first-opener lease survived
        conn.rollback()


def test_routing_hint_first_open_preserves_legacy_rows(copying):
    copying.pub._publish_one(KEY)
    with rooms._transaction(copying.source) as conn:
        record = json.loads(conn.execute(f'SELECT record_json FROM {work.PENDING_TABLE}').fetchone()[0])
    install_legacy_source(copying.source, record)
    with closing(open_sqlite(copying.source)) as conn:
        before = retained(conn)
    copying.pub._select_route(copying.pub._load_route(KEY))
    with closing(open_sqlite(copying.source)) as conn:
        assert retained(conn) == before
        conn.execute('BEGIN IMMEDIATE')
        conn.rollback()


@pytest.mark.parametrize('table', [work.SOURCE_TABLE, work.TARGET_TABLE, work.PENDING_TABLE])
@pytest.mark.parametrize('column', ['digest', 'record_json', 'revision'])
@pytest.mark.parametrize('replace', [False, True])
def test_all_snapshot_metadata_mutations_obey_budget(source, tmp_path, monkeypatch, table, column, replace):
    path = source
    if table == work.TARGET_TABLE:
        path = tmp_path / 'target.db'
        legacy_database(source, path, None)
    else:
        capture(source)
    with rooms._transaction(path, immediate=True) as conn:
        work.initialize(conn)
        if table == work.PENDING_TABLE:
            work.prepare_delivery_locked(conn, room_id='room', target_install_id='target',
                route_generation='route', local_gateway_id=HOME, through_seq=1)
    monkeypatch.setattr(work, 'MAX_STORE_BYTES', 1)
    with rooms._transaction(path, immediate=True) as conn:
        work.initialize(conn)  # retain overage, refresh durable limit
        row = dict(conn.execute(f'SELECT * FROM {table}').fetchone())
        changed = {**row, column: str(row[column]) + '9'}
        with pytest.raises(sqlite3.IntegrityError, match='storage is full'):
            if replace:
                conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(changed)}) VALUES ({','.join('?' for _ in changed)})", tuple(changed.values()))
            else:
                conn.execute(f'UPDATE {table} SET {column}=?', (changed[column],))
        assert dict(conn.execute(f'SELECT * FROM {table}').fetchone()) == row


@pytest.mark.parametrize('column', ['target_install_id', 'route_generation', 'status'])
def test_pending_metadata_only_growth_is_guarded_but_shrink_is_allowed(source, monkeypatch, column):
    with rooms._transaction(source, immediate=True) as conn:
        work.prepare_delivery_locked(conn, room_id='room', target_install_id='target',
            route_generation='route', local_gateway_id=HOME, through_seq=1)
    monkeypatch.setattr(work, 'MAX_STORE_BYTES', 1)
    with rooms._transaction(source, immediate=True) as conn:
        work.initialize(conn)
        with pytest.raises(sqlite3.IntegrityError, match='storage is full'):
            conn.execute(f'UPDATE {work.PENDING_TABLE} SET {column}=?', ('x' * 2000,))
        conn.execute(f"UPDATE {work.PENDING_TABLE} SET status='acked'")
        conn.execute(f"UPDATE {work.PENDING_TABLE} SET disposition='invalid'")
        assert conn.execute(f'SELECT status FROM {work.PENDING_TABLE}').fetchone()[0] == 'acked'


@pytest.mark.parametrize('damage', ['digest', 'revision', 'record_json'])
def test_capture_never_reuses_or_overwrites_metadata_invalid_source(source, damage):
    capture(source)
    with sqlite3.connect(source) as conn:
        value = 99 if damage == 'revision' else 'wrong'
        conn.execute(f'UPDATE {work.SOURCE_TABLE} SET {damage}=?', (value,))
        before = conn.execute(f'SELECT revision,digest,record_json FROM {work.SOURCE_TABLE}').fetchone()
    with pytest.raises(work.WorkRecordError):
        capture(source)
    with sqlite3.connect(source) as conn:
        assert conn.execute(f'SELECT revision,digest,record_json FROM {work.SOURCE_TABLE}').fetchone() == before
        assert conn.execute(f'SELECT disposition FROM {work.SOURCE_TABLE}').fetchone()[0] == 'invalid'


def test_invalid_pending_cannot_supply_anchor_or_false_unchanged_ack(copying):
    copying.pub._publish_one(KEY)
    with sqlite3.connect(copying.source) as conn:
        conn.execute(f"UPDATE {work.PENDING_TABLE} SET revision=revision+3,status='acked'")
        before = conn.execute(f'SELECT revision,digest,record_json,status,route_generation FROM {work.PENDING_TABLE}').fetchone()
    for _ in range(3):
        copying.pub._publish_one(KEY)
    assert copying.records == []
    assert copying.pub.status('room')['routes'][0]['work_record_status'] == 'invalid_work_evidence'
    with closing(open_sqlite(copying.source)) as conn:
        assert not work.pending_delivery_is_anchored_locked(conn, room_id='room', target_install_id='install:target', through_seq=99)
        assert tuple(conn.execute(f'SELECT revision,digest,record_json,status,route_generation FROM {work.PENDING_TABLE}').fetchone()) == before
        assert conn.execute(f'SELECT disposition FROM {work.PENDING_TABLE}').fetchone()[0] == 'invalid'

@pytest.mark.parametrize('damage', ['digest', 'revision'])
def test_ingress_refuses_invalid_target_comparison_without_laundering(copying, damage):
    from gateway import hosted_room_peer as peer
    from gateway import hosted_room_links as links
    from tests.tui_gateway.test_hosted_room_replication import SECRET, TARGET
    copying.pub._publish_one(KEY)
    copying.pub._publish_one(KEY)
    record = copying.records[-1]
    with sqlite3.connect(copying.target) as conn:
        conn.execute(f'UPDATE {work.TARGET_TABLE} SET {damage}=?', (0 if damage == 'revision' else 'wrong',))
        before = conn.execute(f'SELECT revision,digest,record_json FROM {work.TARGET_TABLE}').fetchone()
    link = links.load_room_link(copying.source, room_id='room', member_id=KEY[1])
    claims = peer.decode_room_grant(SECRET, link.grant, permission=work.PERMISSION)
    with pytest.raises(work.WorkRecordError):
        work.ingest(copying.target, record=record, token=link.grant, secret=SECRET,
                    target_install_id=TARGET, target_profile=claims['target_profile'])
    with sqlite3.connect(copying.target) as conn:
        assert conn.execute(f'SELECT revision,digest,record_json FROM {work.TARGET_TABLE}').fetchone() == before
        assert conn.execute(f'SELECT disposition FROM {work.TARGET_TABLE}').fetchone()[0] == 'invalid'


def test_outcome_cas_rechecks_frozen_record_bytes(source):
    with rooms._transaction(source, immediate=True) as conn:
        record = work.prepare_delivery_locked(conn, room_id='room', target_install_id='target',
            route_generation='route', local_gateway_id=HOME, through_seq=1)
        conn.execute(f"UPDATE {work.PENDING_TABLE} SET record_json='corrupt after preparation'")
        assert not work.delivery_status_locked(conn, room_id='room', target_install_id='target',
            route_generation='route', record=record, status='acked')
    with sqlite3.connect(source) as conn:
        assert conn.execute(f'SELECT status,disposition,record_json FROM {work.PENDING_TABLE}').fetchone() == (
            'pending', 'invalid', 'corrupt after preparation')

@pytest.mark.parametrize('table', [work.SOURCE_TABLE, work.PENDING_TABLE])
def test_budget_credit_is_exact_producer_and_target_scope(source, monkeypatch, table):
    from gateway import hosted_room_work_storage as storage
    with rooms._transaction(source, immediate=True) as conn:
        work.prepare_delivery_locked(conn, room_id='room', target_install_id='target',
            route_generation='route', local_gateway_id=HOME, through_seq=1)
        proposed = dict(conn.execute(f'SELECT * FROM {table}').fetchone())
        total, count = storage.usage_sql((work.SOURCE_TABLE, work.TARGET_TABLE, work.PENDING_TABLE, storage.INVALID_TABLE))
        size, rows = conn.execute(f'SELECT {total},{count}').fetchone()
    monkeypatch.setattr(work, 'MAX_STORE_BYTES', size)
    monkeypatch.setattr(work, 'MAX_STORE_ROWS', rows)
    with rooms._transaction(source, immediate=True) as conn:
        work.initialize(conn)
        work._budget(conn, table, proposed)  # exact replacement gets its own credit
        conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(proposed)}) VALUES ({','.join('?' for _ in proposed)})", tuple(proposed.values()))
        variants = [{'producer_gateway_id': 'another'}, {'producer_epoch': 2}]
        if table == work.PENDING_TABLE:
            variants.append({'target_install_id': 'another'})
        for change in variants:
            other = {**proposed, **change}
            with pytest.raises(work.WorkRecordCapacityError):
                work._budget(conn, table, other)
            with pytest.raises(sqlite3.IntegrityError, match='storage is full'):
                conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(other)}) VALUES ({','.join('?' for _ in other)})", tuple(other.values()))
        assert dict(conn.execute(f'SELECT * FROM {table}').fetchone()) == proposed

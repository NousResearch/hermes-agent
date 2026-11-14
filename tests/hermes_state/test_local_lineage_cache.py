"""Regression for #143833: reuse reset projections without stale SQLite snapshots."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from hermes_state import SessionDB
from hermes_state_local import (
    LOCAL_LINEAGE_LEN_EXPR, POLICY_PREFIX, _cached_local_lineage_index, local_lineage_index)


@pytest.fixture
def db(tmp_path):
    database = SessionDB(db_path=tmp_path / 'state.db')
    database.create_session('owner', source='cli')
    database.create_session('head', source='cli')
    assert database._conn is not None
    database._conn.execute("UPDATE sessions SET end_reason='session_reset' WHERE id='owner'")
    database._conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (
        POLICY_PREFIX + 'owner', json.dumps({'session_id': 'owner', 'lineage': ['owner', 'head']})))
    database._conn.commit()
    yield database
    database.close()


def _receipt_scans(statements):
    return [sql for sql in statements if 'FROM state_meta' in sql and 'GLOB' in sql and POLICY_PREFIX in sql]


def test_unchanged_session_lists_and_counts_scan_receipts_once(db):
    statements = []
    with db._read_ctx() as conn:
        conn.set_trace_callback(statements.append)
    for _ in range(5):
        rows = db.list_sessions_rich(limit=10)
        assert [row['id'] for row in rows] == ['head']
        assert set(rows[0]['_lineage_ids']) == {'owner', 'head'}
        assert db.session_count(exclude_children=True) == 1
        assert db.session_count_by_source(exclude_children=True) == {'cli': 1}
    assert len(_receipt_scans(statements)) == 1


@pytest.mark.parametrize('mutation', ['extend', 'retire', 'end_reason', 'delete_head'])
def test_committed_writer_changes_refresh_the_pooled_reader(db, mutation):
    assert [row['id'] for row in db.list_sessions_rich()] == ['head']
    assert db._conn is not None
    if mutation == 'extend':
        db.create_session('tail', source='cli')
        db._conn.execute("UPDATE sessions SET end_reason='session_reset' WHERE id='head'")
        db._conn.execute('UPDATE state_meta SET value=? WHERE key=?', (
            json.dumps({'session_id': 'owner', 'lineage': ['owner', 'head', 'tail']}), POLICY_PREFIX + 'owner'))
        expected = {'tail'}
    elif mutation == 'retire':
        db._conn.execute('DELETE FROM state_meta WHERE key=?', (POLICY_PREFIX + 'owner',))
        expected = {'owner', 'head'}
    elif mutation == 'end_reason':
        db._conn.execute("UPDATE sessions SET end_reason='compression' WHERE id='owner'")
        expected = {'owner', 'head'}
    else:
        db._conn.execute("DELETE FROM sessions WHERE id='head'")
        expected = set()
    db._conn.commit()
    assert {row['id'] for row in db.list_sessions_rich()} == expected
    assert db.session_count(exclude_children=True) == len(expected)


@pytest.mark.parametrize('finish', ['commit', 'rollback'])
def test_same_connection_transactions_do_not_publish_intermediate_results(db, finish):
    conn = db._conn
    assert conn is not None
    before = _cached_local_lineage_index(db, conn)
    conn.execute('BEGIN')
    conn.execute('DELETE FROM state_meta WHERE key=?', (POLICY_PREFIX + 'owner',))
    assert _cached_local_lineage_index(db, conn) == (set(), {})
    getattr(conn, finish)()
    expected = (set(), {}) if finish == 'commit' else before
    assert _cached_local_lineage_index(db, conn) == expected
    # Same-connection autocommit writes must invalidate too (data_version alone cannot see them).
    conn.execute('INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (
        POLICY_PREFIX + 'owner', json.dumps({'session_id': 'owner', 'lineage': ['owner', 'head']})))
    conn.commit()
    assert _cached_local_lineage_index(db, conn) == before


def test_read_snapshot_does_not_outlive_its_transaction(db):
    with db._read_ctx() as reader:
        before = _cached_local_lineage_index(db, reader)
        reader.execute('BEGIN')
        assert _cached_local_lineage_index(db, reader) == before
        assert db._conn is not None
        db._conn.execute('DELETE FROM state_meta WHERE key=?', (POLICY_PREFIX + 'owner',))
        db._conn.commit()
        assert _cached_local_lineage_index(db, reader) == before
        reader.rollback()
        assert _cached_local_lineage_index(db, reader) == (set(), {})


def test_commit_during_projection_is_not_cached_as_a_stable_result(db):
    statements = []
    changed = False
    with db._read_ctx() as reader:
        def trace(sql):
            nonlocal changed
            statements.append(sql)
            if not changed and 'SELECT id,end_reason' in sql:
                changed = True
                assert db._conn is not None
                db._conn.execute('DELETE FROM state_meta WHERE key=?', (POLICY_PREFIX + 'owner',))
                db._conn.commit()
        reader.set_trace_callback(trace)
        try:
            _cached_local_lineage_index(db, reader)
            assert changed
            assert _cached_local_lineage_index(db, reader) == (set(), {})
            assert _cached_local_lineage_index(db, reader) == (set(), {})
        finally:
            reader.set_trace_callback(None)
    assert len(_receipt_scans(statements)) == 2


def test_callers_cannot_mutate_the_cached_projection(db):
    with db._read_ctx() as reader:
        superseded, heads = _cached_local_lineage_index(db, reader)
        superseded.clear()
        heads['head'][1].append('foreign-session')
        heads.clear()
        assert _cached_local_lineage_index(db, reader) == ({'owner'}, {'head': ('owner', ['owner', 'head'])})


def test_connection_churn_is_bounded_and_other_databases_stay_isolated(db, tmp_path):
    # Real handles can churn independently of the pool without retaining an unbounded cache.
    conns = [sqlite3.connect(f'file:{db.db_path}?mode=ro', uri=True) for _ in range(12)]
    try:
        for conn in conns:
            assert _cached_local_lineage_index(db, conn)[0] == {'owner'}
        cache, _lock = db._local_lineage_cache
        assert len(cache) <= 8
        other = SessionDB(db_path=tmp_path / 'other.db')
        try:
            with other._read_ctx() as conn:
                assert _cached_local_lineage_index(other, conn) == (set(), {})
        finally:
            other.close()
    finally:
        for conn in conns:
            conn.close()


def test_concurrent_lists_keep_the_same_projection(db):
    barrier = Barrier(6)
    def read(_):
        barrier.wait(timeout=10)
        return [{row['id'] for row in db.list_sessions_rich()} for _ in range(5)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(read, range(6)):
            assert result == [{'head'}] * 5


def test_read_only_legacy_database_needs_no_schema_change(tmp_path):
    path = tmp_path / 'legacy.db'
    conn = sqlite3.connect(path)
    conn.executescript('CREATE TABLE sessions(id TEXT PRIMARY KEY,end_reason TEXT);'
                       'CREATE TABLE state_meta(key TEXT PRIMARY KEY,value TEXT);')
    conn.executemany('INSERT INTO sessions VALUES(?,?)', [('old', 'session_reset'), ('new', None)])
    conn.execute('INSERT INTO state_meta VALUES(?,?)', (
        POLICY_PREFIX + 'old', json.dumps({'session_id': 'old', 'lineage': ['old', 'new']})))
    conn.commit()
    conn.close()
    before = path.read_bytes()
    db = SessionDB(db_path=path, read_only=True)
    try:
        with db._read_ctx() as reader:
            expected = ({'old'}, {'new': ('old', ['old', 'new'])})
            assert _cached_local_lineage_index(db, reader) == expected
            assert _cached_local_lineage_index(db, reader) == expected
    finally:
        db.close()
    assert path.read_bytes() == before


def test_local_lineage_lookup_is_served_by_the_expression_index(db):
    """The projection must seek idx_state_meta_local_lineage, not read every receipt (#152507)."""
    conn = db._conn
    assert conn is not None
    indexes = {row[1] for row in conn.execute("PRAGMA index_list('state_meta')")}
    assert 'idx_state_meta_local_lineage' in indexes
    # Enough rows that the planner's choice reflects the query shape, not a tiny table.
    conn.executemany('INSERT INTO state_meta(key, value) VALUES(?, ?)', (
        (POLICY_PREFIX + f'plain-{i}', json.dumps({
            'session_id': f'plain-{i}', 'lineage': [f'plain-{i}']})) for i in range(200)))
    conn.commit()
    plan = ' | '.join(row[-1] for row in conn.execute(
        f'EXPLAIN QUERY PLAN SELECT value FROM state_meta WHERE {LOCAL_LINEAGE_LEN_EXPR} > 1'))
    assert 'USING INDEX idx_state_meta_local_lineage' in plan, plan
    assert 'SCAN' not in plan, plan


def test_indexed_predicate_matches_the_receipt_scan_it_replaced(db):
    """Rows and answers must stay identical to the old key-range + JSON predicate (#152507)."""
    conn = db._conn
    assert conn is not None
    receipts = {
        'two': {'session_id': 'two', 'lineage': ['two', 'three']},
        'three': {'session_id': 'three', 'lineage': ['three', 'four', 'five']},
        'one': {'session_id': 'one', 'lineage': ['one']},
        'empty': {'session_id': 'empty'},
    }
    conn.executemany('INSERT INTO state_meta(key, value) VALUES(?, ?)',
                     ((POLICY_PREFIX + name, json.dumps(body)) for name, body in receipts.items()))
    conn.execute('INSERT INTO state_meta(key, value) VALUES(?, ?)', (POLICY_PREFIX + 'broken', '{not json'))
    conn.execute('INSERT INTO state_meta(key, value) VALUES(?, ?)',
                 ('gateway.api.binding.v1:x', json.dumps({'lineage': ['a', 'b']})))
    conn.commit()
    indexed = sorted(row[0] for row in conn.execute(
        f'SELECT key FROM state_meta WHERE {LOCAL_LINEAGE_LEN_EXPR} > 1'))
    scanned = sorted(row[0] for row in conn.execute(
        "SELECT key FROM state_meta NOT INDEXED WHERE key GLOB ? AND json_valid(value) "
        "AND json_array_length(value, '$.lineage') > 1", (POLICY_PREFIX + '*',)))
    assert indexed == scanned
    assert indexed == [POLICY_PREFIX + name for name in ('owner', 'three', 'two')]


def test_writable_open_recreates_the_index_on_an_existing_store(tmp_path):
    """An older store gains the index on open, and repeated opens do not fail (#152507)."""
    path = tmp_path / 'state.db'
    db = SessionDB(db_path=path)
    try:
        db.create_session('owner', source='cli')
        db.create_session('head', source='cli')
        assert db._conn is not None
        db._conn.execute("UPDATE sessions SET end_reason='session_reset' WHERE id='owner'")
        db._conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (
            POLICY_PREFIX + 'owner', json.dumps({'session_id': 'owner', 'lineage': ['owner', 'head']})))
        db._conn.commit()
    finally:
        db.close()
    raw = sqlite3.connect(path)
    raw.execute('DROP INDEX idx_state_meta_local_lineage')
    raw.commit()
    raw.close()
    for _ in range(2):
        db = SessionDB(db_path=path)
        try:
            assert db._conn is not None
            indexes = {row[1] for row in db._conn.execute("PRAGMA index_list('state_meta')")}
            assert 'idx_state_meta_local_lineage' in indexes
            with db._read_ctx() as reader:
                assert _cached_local_lineage_index(db, reader) == (
                    {'owner'}, {'head': ('owner', ['owner', 'head'])})
        finally:
            db.close()


def test_local_lineage_work_does_not_scale_with_single_segment_receipts(db):
    """An unchanged lineage answer must not parse every unrelated policy snapshot (#152507)."""
    conn = db._conn
    assert conn is not None

    def lookup_work():
        steps = 0

        def tick():
            nonlocal steps
            steps += 1
            return 0

        conn.set_progress_handler(tick, 1)
        try:
            result = local_lineage_index(conn)
        finally:
            conn.set_progress_handler(None, 0)
        return steps, result

    before_steps, before = lookup_work()
    assert before == ({'owner'}, {'head': ('owner', ['owner', 'head'])})
    config = 'x' * 4096
    conn.executemany(
        'INSERT INTO state_meta(key, value) VALUES(?, ?)',
        ((POLICY_PREFIX + f'ordinary-{i}', json.dumps({
            'session_id': f'ordinary-{i}', 'lineage': [f'ordinary-{i}'],
            'policy': {'config_json': config},
        })) for i in range(400)),
    )
    conn.commit()

    after_steps, after = lookup_work()
    assert after == before
    # SQLite VM steps, not elapsed time: an index seek stays bounded as the
    # store accumulates receipts without resets; the old JSON scan is O(rows).
    assert after_steps - before_steps < 100, (before_steps, after_steps)

"""The SQLite backend's session listings and counts scan reset receipts once while the store is unchanged, and a
read-only open of a legacy database reads its lineage projection without a schema change."""
import json
import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.local import (
    LOCAL_LINEAGE_LEN_EXPR,
    POLICY_PREFIX,
    _cached_local_lineage_index,
    local_lineage_index,
)
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    list_sessions_rich,
    session_count,
    session_count_by_source,
)


@pytest.fixture
def db(tmp_path):
    database = StoreHolder(tmp_path / 'state.db')
    create_session(database, 'owner', source='cli')
    create_session(database, 'head', source='cli')
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
        rows = list_sessions_rich(db, limit=10)
        assert [row['id'] for row in rows] == ['head']
        assert set(rows[0]['_lineage_ids']) == {'owner', 'head'}
        assert session_count(db, exclude_children=True) == 1
        assert session_count_by_source(db, exclude_children=True) == {'cli': 1}
    assert len(_receipt_scans(statements)) == 1


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
    db = StoreHolder(path, read_only=True)
    try:
        with db._read_ctx() as reader:
            expected = ({'old'}, {'new': ('old', ['old', 'new'])})
            assert _cached_local_lineage_index(db, reader) == expected
            assert _cached_local_lineage_index(db, reader) == expected
    finally:
        db.close()
    assert path.read_bytes() == before


def test_local_lineage_lookup_is_served_by_the_expression_index(db):
    """The projection must seek idx_state_meta_local_lineage instead of scanning every receipt."""
    conn = db._conn
    assert conn is not None
    indexes = {row[1] for row in conn.execute("PRAGMA index_list('state_meta')")}
    assert 'idx_state_meta_local_lineage' in indexes
    # Enough rows that the planner's choice reflects the query shape, not a tiny table.
    conn.executemany(
        'INSERT INTO state_meta(key, value) VALUES(?, ?)',
        ((POLICY_PREFIX + f'plain-{i}', json.dumps({
            'session_id': f'plain-{i}', 'lineage': [f'plain-{i}'],
        })) for i in range(200)),
    )
    conn.commit()
    plan = ' | '.join(row[-1] for row in conn.execute(
        f'EXPLAIN QUERY PLAN SELECT value FROM state_meta WHERE {LOCAL_LINEAGE_LEN_EXPR} > 1'
    ))
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
    conn.executemany(
        'INSERT INTO state_meta(key, value) VALUES(?, ?)',
        ((POLICY_PREFIX + name, json.dumps(body)) for name, body in receipts.items()),
    )
    conn.execute('INSERT INTO state_meta(key, value) VALUES(?, ?)', (POLICY_PREFIX + 'broken', '{not json'))
    conn.execute(
        'INSERT INTO state_meta(key, value) VALUES(?, ?)',
        ('gateway.api.binding.v1:x', json.dumps({'lineage': ['a', 'b']})),
    )
    conn.commit()
    indexed = sorted(row[0] for row in conn.execute(
        f'SELECT key FROM state_meta WHERE {LOCAL_LINEAGE_LEN_EXPR} > 1'
    ))
    scanned = sorted(row[0] for row in conn.execute(
        "SELECT key FROM state_meta NOT INDEXED WHERE key GLOB ? AND json_valid(value) "
        "AND json_array_length(value, '$.lineage') > 1",
        (POLICY_PREFIX + '*',),
    ))
    assert indexed == scanned
    assert indexed == [POLICY_PREFIX + name for name in ('owner', 'three', 'two')]


def test_writable_open_recreates_the_index_on_an_existing_store(tmp_path):
    """An older store gains the index on open, and repeated opens do not fail (#152507)."""
    path = tmp_path / 'state.db'
    db = StoreHolder(path)
    try:
        create_session(db, 'owner', source='cli')
        create_session(db, 'head', source='cli')
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
        db = StoreHolder(path)
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
    """Uncached lookups must not parse every unrelated policy snapshot (#152507, #155093)."""
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

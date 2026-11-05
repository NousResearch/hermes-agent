"""The SQLite backend's session listings and counts scan reset receipts once while the store is unchanged, and a
read-only open of a legacy database reads its lineage projection without a schema change."""
import json
import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.local import POLICY_PREFIX, _cached_local_lineage_index
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

"""Legacy previews omit absent ledgers only, without repairing the database."""
import sqlite3

import pytest

from hermes_state import SessionDB
from tests.state.raw_maintenance_helpers import ended, ledger


@pytest.mark.parametrize('preview', ['empty-count', 'prune'])
@pytest.mark.parametrize('schema', ['absent', 'admissions', 'workers', 'uppercase',
                                  'malformed-admissions', 'malformed-workers', 'view', 'query-denied'])
def test_readonly_preview_checks_present_ledger_without_mutating_file(tmp_path, preview, schema):
    path = tmp_path / 'legacy.db'
    with SessionDB(path) as db:
        ended(db, 'protected')
        ended(db, 'legacy')
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE worker_receipts')
        if schema in {'admissions', 'uppercase', 'query-denied'}:
            ledger(conn, 'protected', 'admission-terminal')
            conn.execute('DROP TABLE worker_executions')
            if schema == 'uppercase':
                conn.execute('ALTER TABLE session_admissions RENAME TO temporary_name')
                conn.execute('ALTER TABLE temporary_name RENAME TO SESSION_ADMISSIONS')
        elif schema == 'workers':
            ledger(conn, 'protected', 'worker-terminal')
            conn.execute('DROP TABLE session_admissions')
        else:
            conn.execute('DROP TABLE session_admissions')
            conn.execute('DROP TABLE worker_executions')
            if schema.startswith('malformed'):
                table = 'session_admissions' if schema.endswith('admissions') else 'worker_executions'
                conn.execute('CREATE TABLE ' + table + '(wrong_column TEXT)')
            elif schema == 'view':
                conn.execute('CREATE VIEW session_admissions AS SELECT id AS target_session_id FROM sessions')
    before = path.read_bytes()
    with SessionDB(path, read_only=True) as db:
        if schema == 'query-denied':
            db._conn.set_authorizer(lambda action, name, *args:
                sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and name == 'session_admissions'
                else sqlite3.SQLITE_OK)
        report = {}
        def inspect():
            if preview == 'empty-count':
                return db.count_empty_sessions(report=report)
            return len(db.list_prune_candidates(older_than_days=None, exclude_ledger_owned=True, report=report))
        if schema.startswith('malformed') or schema in {'view', 'query-denied'}:
            with pytest.raises(sqlite3.DatabaseError):
                inspect()
            assert report == {}
        else:
            assert inspect() == (2 if schema == 'absent' else 1)
            assert report == {'skipped_protected': 0 if schema == 'absent' else 1}
    assert path.read_bytes() == before


@pytest.mark.parametrize('operation', ['count', 'preview', 'delete', 'prune'])
def test_writer_missing_runtime_schema_still_refuses(tmp_path, operation):
    with SessionDB(tmp_path / 'state.db') as db:
        ended(db, 'legacy')
        db._execute_write(lambda conn: conn.execute('DROP TABLE session_admissions'))
        before = db.get_session('legacy')
        report = {}
        with pytest.raises(sqlite3.DatabaseError):
            if operation == 'count':
                db.count_empty_sessions(report=report)
            elif operation == 'preview':
                db.list_prune_candidates(older_than_days=None, exclude_ledger_owned=True, report=report)
            elif operation == 'delete':
                db.delete_session('legacy')
            else:
                db.prune_sessions(older_than_days=None, report=report)
        assert report == {} and db.get_session('legacy') == before

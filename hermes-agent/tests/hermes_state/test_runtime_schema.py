"""Real SQLite schema reconciliation and storage-failure controls."""
import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION
import hermes_state_runtime as rt


def test_schema_reconciles_existing_sessions_without_losing_history(tmp_path):
    path = tmp_path / 'state.db'
    db = SessionDB(db_path=path)
    db.create_session('existing', source='test')
    db.append_message('existing', role='user', content='retained')
    db.close()
    with sqlite3.connect(path) as legacy:
        legacy.execute('ALTER TABLE sessions DROP COLUMN runtime_generation')
        legacy.execute('ALTER TABLE sessions DROP COLUMN runtime_revision')
        for table in ('worker_receipts', 'worker_executions', 'session_admissions', 'runtime_epoch'):
            legacy.execute(f'DROP TABLE {table}')
        legacy.execute('UPDATE schema_version SET version=?', (SCHEMA_VERSION - 1,))
    db = SessionDB(db_path=path)
    try:
        assert db.get_session('existing')['runtime_revision'] == 0
        assert db.get_session('existing')['runtime_generation'] == 0
        assert [r['content'] for r in db.get_messages('existing')] == ['retained']
        assert db._conn.execute('SELECT version FROM schema_version').fetchone()[0] == SCHEMA_VERSION
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        assert rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id='existing', request_id='r', payload={'text': 'next'})['status'] == 'queued'
        assert db._conn.execute('PRAGMA foreign_key_check').fetchall() == []
    finally:
        db.close()


def test_real_sqlite_full_aborts_admission_and_epoch_failure_is_atomic(tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('s', source='test')
        epoch = rt.begin_runtime_epoch(db, instance_id='boot')
        db._conn.execute("CREATE TRIGGER refuse_epoch BEFORE UPDATE ON runtime_epoch BEGIN SELECT RAISE(ABORT, 'epoch fixture'); END")
        with pytest.raises(sqlite3.IntegrityError, match='epoch fixture'):
            rt.begin_runtime_epoch(db, instance_id='refused')
        assert tuple(db._conn.execute('SELECT epoch,instance_id FROM runtime_epoch').fetchone()) == (epoch, 'boot')
        db._conn.execute('DROP TRIGGER refuse_epoch')
        pages = db._conn.execute('PRAGMA page_count').fetchone()[0]
        db._conn.execute(f'PRAGMA max_page_count={pages}')
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='full', payload={'text': 'x' * (2 * 1024 * 1024)})
        with pytest.raises(sqlite3.OperationalError) as exc:
            rt.admit_session_input(db, **args)
        assert exc.value.sqlite_errorcode == sqlite3.SQLITE_FULL
        assert rt.list_session_admissions(db, session_id='s') == []
        db._conn.execute('PRAGMA max_page_count=2147483646')
        assert rt.admit_session_input(db, **args)['status'] == 'queued'
    finally:
        db.close()

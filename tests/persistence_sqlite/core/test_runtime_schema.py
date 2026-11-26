"""Real SQLite schema reconciliation and storage-failure controls."""
import sqlite3

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import SCHEMA_VERSION
from persistence.sqlite.authority._admission_store import admit_session_input, list_session_admissions
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.messages import append_message, get_messages
from persistence.sqlite.conversation._store.sessions import create_session, get_session


def test_schema_reconciles_existing_sessions_without_losing_history(tmp_path):
    path = tmp_path / 'state.db'
    holder = StoreHolder(path)
    create_session(holder, 'existing', 'test')
    append_message(holder, 'existing', role='user', content='retained')
    holder.close()
    with sqlite3.connect(path) as legacy:
        legacy.execute('ALTER TABLE sessions DROP COLUMN runtime_generation')
        legacy.execute('ALTER TABLE sessions DROP COLUMN runtime_revision')
        for table in ('worker_receipts', 'worker_executions', 'session_admissions', 'runtime_epoch'):
            legacy.execute(f'DROP TABLE {table}')
        legacy.execute('UPDATE schema_version SET version=?', (SCHEMA_VERSION - 1,))
    holder = StoreHolder(path)
    try:
        assert get_session(holder, 'existing')['runtime_revision'] == 0
        assert get_session(holder, 'existing')['runtime_generation'] == 0
        assert [r['content'] for r in get_messages(holder, 'existing')] == ['retained']
        assert holder._conn.execute('SELECT version FROM schema_version').fetchone()[0] == SCHEMA_VERSION
        epoch = begin_runtime_epoch(holder, instance_id='boot')
        assert admit_session_input(holder, epoch=epoch, principal_id='human', session_id='existing', request_id='r', payload={'text': 'next'})[0]['status'] == 'queued'
        assert holder._conn.execute('PRAGMA foreign_key_check').fetchall() == []
    finally:
        holder.close()


def test_real_sqlite_full_aborts_admission_and_epoch_failure_is_atomic(tmp_path):
    holder = StoreHolder(tmp_path / 'state.db')
    try:
        create_session(holder, 's', 'test')
        epoch = begin_runtime_epoch(holder, instance_id='boot')
        holder._conn.execute("CREATE TRIGGER refuse_epoch BEFORE UPDATE ON runtime_epoch BEGIN SELECT RAISE(ABORT, 'epoch fixture'); END")
        with pytest.raises(sqlite3.IntegrityError, match='epoch fixture'):
            begin_runtime_epoch(holder, instance_id='refused')
        assert tuple(holder._conn.execute('SELECT epoch,instance_id FROM runtime_epoch').fetchone()) == (epoch, 'boot')
        holder._conn.execute('DROP TRIGGER refuse_epoch')
        pages = holder._conn.execute('PRAGMA page_count').fetchone()[0]
        holder._conn.execute(f'PRAGMA max_page_count={pages}')
        args = dict(epoch=epoch, principal_id='human', session_id='s', request_id='full', payload={'text': 'x' * (2 * 1024 * 1024)})
        with pytest.raises(sqlite3.OperationalError) as exc:
            admit_session_input(holder, **args)
        assert exc.value.sqlite_errorcode == sqlite3.SQLITE_FULL
        assert list_session_admissions(holder, session_id='s') == []
        holder._conn.execute('PRAGMA max_page_count=2147483646')
        assert admit_session_input(holder, **args)[0]['status'] == 'queued'
    finally:
        holder.close()

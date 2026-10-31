"""Explicit repair must restore browsing after additive schema drift (#140315)."""
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION
from hermes_cli import sessions_cmd
from hermes_cli.web_server_sessions import _open_session_db_at_path


@pytest.fixture(autouse=True)
def _volume_headroom(monkeypatch):
    """The backup/snapshot guards reserve 2% of the volume: hermetic on a nearly full dev disk."""
    import hermes_state_repair as repair
    monkeypatch.setattr(repair, '_REPAIR_BACKUP_FREE_FRACTION', 0.0)
    # The holder scan sees every process on the host: a developer's running Hermes would hold a
    # real state.db and refuse these tmp_path repairs. Same stub as test_state_db_repair_non_destructive.
    monkeypatch.setattr(repair, '_live_writer_holds_db', lambda _path: False)


def _legacy_store(tmp_path, monkeypatch):
    path = tmp_path / 'state.db'
    with SessionDB(db_path=path) as db:
        db.create_session('retained', source='cli')
        db.append_message('retained', role='user', content='history survives')
    with closing(sqlite3.connect(path)) as conn:
        conn.execute('ALTER TABLE sessions DROP COLUMN runtime_generation')
        conn.execute('ALTER TABLE sessions DROP COLUMN runtime_revision')
        conn.execute('ALTER TABLE async_delegations DROP COLUMN owner_execution_id')
        for table in ('worker_receipts', 'worker_executions', 'session_admissions', 'runtime_epoch'):
            conn.execute(f'DROP TABLE {table}')
        conn.commit()
        assert conn.execute('SELECT version FROM schema_version').fetchone()[0] == SCHEMA_VERSION
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', path)
    monkeypatch.setattr(sessions_cmd, 'get_hermes_home', lambda: tmp_path)
    return path


def test_check_only_reports_drift_without_mutation(tmp_path, monkeypatch):
    path = _legacy_store(tmp_path, monkeypatch)
    before = path.read_bytes()
    # Browsing would catch this unowned store up (#140315); check-only must not.
    assert sessions_cmd._cmd_repair(SimpleNamespace(check_only=True)) == 1
    assert path.read_bytes() == before


@pytest.mark.parametrize('blocker', ['owner', 'ddl', 'incomplete', 'backup'])
def test_refused_reconciliation_preserves_original(tmp_path, monkeypatch, blocker):
    from gateway.runtime_ownership import ProfileOwnership
    import hermes_state_repair as repair
    import hermes_state_schema as schema

    path = _legacy_store(tmp_path, monkeypatch)
    before = path.read_bytes()
    owner = ProfileOwnership()
    if blocker == 'owner':
        owner.reserve([tmp_path])
    elif blocker == 'ddl':
        def fail(cursor):
            cursor.execute('CREATE TABLE only_in_scratch (id INTEGER)')
            raise sqlite3.OperationalError('injected DDL refusal')
        monkeypatch.setattr(schema, 'apply_declared_schema', fail)
    elif blocker == 'incomplete':
        monkeypatch.setattr(schema, 'apply_declared_schema', lambda cursor: None)
    else:
        monkeypatch.setattr(repair, '_backup_db_file', lambda path: (None, 'injected backup refusal'))
    try:
        assert sessions_cmd._cmd_repair(SimpleNamespace(check_only=False, no_backup=False)) == 1
    finally:
        owner.close()
    assert path.read_bytes() == before
    assert not path.with_name(path.name + '.reconcile-scratch').exists()


@pytest.mark.parametrize('with_history', [True, False])
def test_canonical_store_is_unchanged(tmp_path, monkeypatch, with_history):
    path = tmp_path / 'state.db'
    with SessionDB(db_path=path) as db:
        if with_history:
            db.create_session('normal', source='cli')
            db.append_message('normal', role='user', content='normal')
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', path)
    before = path.read_bytes()
    assert sessions_cmd._cmd_repair(SimpleNamespace(check_only=False)) in (None, 0)
    assert path.read_bytes() == before


@pytest.mark.parametrize('missing', ['sessions', 'messages'])
def test_missing_history_table_is_not_recreated_as_empty(tmp_path, monkeypatch, missing):
    from hermes_state_reconcile import repair_declarative_schema
    path = tmp_path / 'state.db'
    kept = 'messages' if missing == 'sessions' else 'sessions'
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f'CREATE TABLE {kept} (id TEXT)')
        conn.execute(f"INSERT INTO {kept} VALUES ('retain-me')")
        conn.commit()
    before = path.read_bytes()
    report = repair_declarative_schema(path)
    assert report['repaired'] is False
    assert 'canonical history tables are missing' in report['error']
    assert path.read_bytes() == before


def test_snapshot_history_loss_is_not_promoted(tmp_path, monkeypatch):
    import hermes_state_repair as repair
    from hermes_state_reconcile import repair_declarative_schema
    path = _legacy_store(tmp_path, monkeypatch)
    before = path.read_bytes()
    copy = repair._copy_database_snapshot

    def changed_snapshot(source, destination, **kwargs):
        copy(source, destination, **kwargs)
        if kwargs.get('source_connection') is not None:
            with closing(sqlite3.connect(destination)) as conn:
                conn.execute('DROP TABLE messages')
                conn.commit()

    monkeypatch.setattr(repair, '_copy_database_snapshot', changed_snapshot)
    report = repair_declarative_schema(path)
    assert report['repaired'] is False
    assert 'canonical history tables are missing' in report['error']
    assert path.read_bytes() == before


def test_repair_restores_missing_indexed_column(tmp_path, monkeypatch):
    path = tmp_path / 'state.db'
    with SessionDB(db_path=path) as db:
        db.create_session('retained', source='cli')
        db.append_message('retained', role='user', content='history survives')
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', path)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute('DROP INDEX idx_async_delegations_delivery')
        conn.execute('ALTER TABLE async_delegations DROP COLUMN completed_at')
        conn.commit()
    assert sessions_cmd._cmd_repair(SimpleNamespace(check_only=False, no_backup=False)) in (None, 0)
    with _open_session_db_at_path(path, read_only=True) as db:
        assert [row['content'] for row in db.get_messages('retained')] == ['history survives']
        assert db._conn.execute("SELECT name FROM sqlite_master WHERE name='idx_async_delegations_delivery'").fetchone()


@pytest.mark.parametrize('backup', [True, False])
def test_explicit_repair_restores_readonly_history(tmp_path, monkeypatch, backup):
    path = _legacy_store(tmp_path, monkeypatch)
    assert sessions_cmd._cmd_repair(SimpleNamespace(check_only=False, no_backup=not backup)) in (None, 0)
    backups = list(tmp_path.glob('state.db.malformed-backup-*'))
    assert bool(backups) is backup
    with _open_session_db_at_path(path, read_only=True) as db:
        assert [row['content'] for row in db.get_messages('retained')] == ['history survives']
        assert db.session_count() == 1
        assert db._conn.execute('SELECT version FROM schema_version').fetchone()[0] == SCHEMA_VERSION
    assert sessions_cmd._cmd_repair(SimpleNamespace(check_only=True)) in (None, 0)

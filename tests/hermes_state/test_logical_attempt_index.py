"""Canonical acceptance projection and bounded explicit inventory preparation."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def payload(task='task', scope='signed-epoch-one'):
    # Indexing is a projection of accepted canonical bytes, not a new grant parser.
    return {'text': 'private prompt', 'api_turn_v1': {'run_owner_scope': scope,
            'settings': {'room_dispatch': {'task_id': task, 'execution_generation': 1}}}}


def accept(db, epoch, *, sid='member', request='run', task='task', scope='signed-epoch-one'):
    db.ensure_session(sid, source='api')
    return rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id=sid,
                                 request_id=request, payload=payload(task, scope))


def settle(db, epoch, sid='member'):
    row = rt.claim_session_input(db, epoch=epoch, session_id=sid)
    return rt.settle_session_input(db, epoch=epoch, admission_id=row['admission_id'],
                                  generation=row['generation'], outcome='completed')


def index_api():
    import hermes_state_logical_attempts as index
    return index


def prepare(db, batch=7):
    index = index_api()
    for _ in range(1000):
        result = index.prepare_logical_attempt_index(db, batch_size=batch)
        assert result['processed'] <= batch
        if result['complete'] or result['error']:
            return result
    pytest.fail('bounded preparation did not finish')


def lookup(db, *, sid='member', scope='signed-epoch-one', task='task'):
    return index_api().lookup_logical_attempt(db, principal_id='api', session_id=sid,
        owner_scope=scope, task_id=task, execution_generation=1)


def test_admission_commits_projection_and_retirement_preserves_exact_identity(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        row = accept(db, epoch)
        table = db._conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logical_attempts'").fetchone()
        assert table is not None, 'canonical admission has no transaction-owned logical projection'
        assert prepare(db)['complete']
        found = lookup(db)
        assert found['admission_id'] == row['admission_id']
        assert found['request_id'] == 'run'
        assert lookup(db, scope='signed-epoch-two') is None
        assert lookup(db, task='other-task') is None
        assert accept(db, epoch)['admission_id'] == row['admission_id']
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            accept(db, epoch, task='changed')
        settle(db, epoch)
        assert db.delete_session('member')
        assert lookup(db) == found
        assert lookup(db, scope='signed-epoch-two') is None
        assert db._conn.execute("SELECT count(*) FROM logical_attempts WHERE admission_id!=''").fetchone()[0] == 1
        assert 'private prompt' not in str(tuple(db._conn.execute("SELECT * FROM logical_attempts WHERE admission_id!=''").fetchone()))
        assert rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='member',
                request_id='run', payload=payload())['admission_id'] == row['admission_id']


def test_projection_failure_rolls_back_actual_admission_and_retirement(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        accept(db, epoch, sid='control')
        assert db._conn.execute("SELECT 1 FROM sqlite_master WHERE name='logical_attempts'").fetchone(), 'missing projection transaction'
        db._execute_write(lambda c: c.execute("CREATE TRIGGER deny_index BEFORE INSERT ON logical_attempts BEGIN SELECT RAISE(ABORT,'projection failed'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='projection failed'):
            accept(db, epoch)
        assert rt.list_session_admissions(db, session_id='member') == []
        db._execute_write(lambda c: c.execute('DROP TRIGGER deny_index'))
        row = accept(db, epoch)
        settle(db, epoch)
        db._execute_write(lambda c: c.execute("CREATE TRIGGER deny_retirement BEFORE INSERT ON state_meta WHEN NEW.key GLOB 'gateway.terminal_admission.v1.*' BEGIN SELECT RAISE(ABORT,'retirement failed'); END"))
        with pytest.raises(sqlite3.IntegrityError, match='retirement failed'):
            db.delete_session('member')
        assert db.get_session('member')
        assert rt.get_session_admission(db, admission_id=row['admission_id'])['payload'] == payload()
        assert prepare(db)['complete']
        assert lookup(db)['admission_id'] == row['admission_id']


def test_unprepared_partial_restart_and_concurrent_forward_admission(tmp_path):
    path = tmp_path / 'state.db'
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        for n in range(12):
            accept(db, epoch, request=f'run-{n}', task=f'task-{n}')
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, task='fresh')
        first = index_api().prepare_logical_attempt_index(db, batch_size=2)
        assert not first['complete'] and first['processed'] == 2
    with SessionDB(path) as db:
        start = Event()
        def writer():
            with SessionDB(path) as other:
                start.wait(5)
                return accept(other, epoch, request='concurrent', task='concurrent')
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(writer)
            start.set()
            assert prepare(db, batch=2)['complete']
            row = future.result(10)
        assert lookup(db, task='concurrent')['admission_id'] == row['admission_id']
        assert lookup(db, task='fresh') is None


def test_erased_legacy_is_physical_hold_not_invented_scope(tmp_path):
    from hermes_state_terminal import ADMISSION_PREFIX, identity_key
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        row = accept(db, epoch)
        settle(db, epoch)
        raw = dict(db._conn.execute('SELECT * FROM session_admissions').fetchone())
        # Faithful old retirement inventory with NO pre-erasure index. The old
        # identity value names the admission, not a reversible physical identity.
        raw['admission_id'] = 'legacy'; raw['request_id'] = 'old-run'
        raw['target_session_id'] = 'legacy-member'; raw['payload_json'] = '{}'
        def seed(c):
            for key, value in [(ADMISSION_PREFIX + 'legacy', json.dumps(raw)),
                    (identity_key('api', 'legacy-member', 'old-run'), json.dumps('legacy')),
                    (RETIRED_PREFIX + 'legacy-member', '{}')]:
                c.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, value))
        db._execute_write(seed)
        assert prepare(db)['complete']
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='legacy-member', scope='new-epoch')
        assert lookup(db, task='fresh') is None
        assert lookup(db)['admission_id'] == row['admission_id']


def test_unclassifiable_legacy_terminal_is_explicit_preparation_failure(tmp_path):
    from hermes_state_terminal import ADMISSION_PREFIX, identity_key
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    with SessionDB(tmp_path / 'state.db') as db:
        def seed(c):
            for key, value in [(ADMISSION_PREFIX + 'legacy', '{malformed'),
                    (identity_key('api', 'erased-member', 'erased-run'), json.dumps('legacy')),
                    (RETIRED_PREFIX + 'erased-member', '{}')]:
                c.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, value))
        db._execute_write(seed)
        result = prepare(db)
        assert not result['complete']
        assert result['error'] == 'unclassifiable_terminal'
        assert result['error_key'] == ADMISSION_PREFIX + 'legacy'
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, task='fresh')


def test_coverage_schema_and_projection_clear_are_not_empty_cache_absence(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        accept(db, epoch)
        assert prepare(db)['complete']
        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(lambda c: c.execute('DELETE FROM logical_attempts'))
        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(lambda c: c.execute("UPDATE logical_attempts SET task_id='foreign'"))
        db._execute_write(lambda c: c.execute('DELETE FROM logical_attempt_coverage'))
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, task='fresh')
        assert prepare(db)['complete']
        db._execute_write(lambda c: c.execute('CREATE TABLE unrelated_schema_change(x)'))
        assert lookup(db, task='fresh') is None
        db._execute_write(lambda c: c.execute('DROP INDEX logical_attempt_exact'))
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, task='fresh')
        db._execute_write(lambda c: c.execute('CREATE INDEX logical_attempt_exact ON logical_attempts(principal_id,session_id,owner_scope,task_id,execution_generation)'))
        assert prepare(db)['complete']
        assert lookup(db, task='fresh') is None

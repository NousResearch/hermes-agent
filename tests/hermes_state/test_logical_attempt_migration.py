"""Preparation consumes actual legacy retirement, never hashed-identity guesses."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt
from tests.hermes_state.test_logical_attempt_index import accept, lookup, payload, prepare, settle


def legacy_accept_and_retire(db, epoch, monkeypatch, *, sid='old-member', retire=True):
    import hermes_state_logical_attempts as index
    # Old owners have no projection writer. Keep the real admission, claim,
    # settlement and SessionDB retirement SQL; disable only the new projection.
    with monkeypatch.context() as old_owner:
        old_owner.setattr(index, 'project_admission', lambda *a, **kw: None)
        row = accept(db, epoch, sid=sid, request=sid, task=sid)
        settle(db, epoch, sid)
        if retire:
            assert db.delete_session(sid)
    return row


def test_actual_preindex_erasure_and_unclassifiable_corruption(tmp_path, monkeypatch):
    from hermes_state_terminal import ADMISSION_PREFIX, identity_key
    from hermes_state_mutation_retirement import RETIRED_PREFIX
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        row = legacy_accept_and_retire(db, epoch, monkeypatch)
        key = ADMISSION_PREFIX + row['admission_id']
        assert db._conn.execute('SELECT value FROM state_meta WHERE key=?',
            (identity_key('api', 'old-member', 'old-member'),)).fetchone()[0] == json.dumps(row['admission_id'])
        assert db._conn.execute('SELECT 1 FROM state_meta WHERE key=?', (RETIRED_PREFIX + 'old-member',)).fetchone()
        assert not db._conn.execute("SELECT 1 FROM logical_attempts WHERE admission_id!=''").fetchone()
        db._execute_write(lambda c: c.execute('UPDATE state_meta SET value=? WHERE key=?', ('{lost-json', key)))
        failed = prepare(db)
        assert failed['error'] == 'unclassifiable_terminal' and not failed['complete']
        assert failed['error_key'] == key
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='new-member')


def test_real_erased_legacy_preserves_physical_ambiguity(tmp_path, monkeypatch):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        legacy_accept_and_retire(db, epoch, monkeypatch)
        assert prepare(db)['complete']
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='old-member', scope='new-home-epoch')
        assert lookup(db, sid='unrelated') is None


@pytest.mark.parametrize('field,bad', [('status', 'queued'), ('payload_digest', 'broken'), ('intent', 'broken')])
def test_independent_legacy_identity_scopes_corrupt_nonidentity_fields(tmp_path, monkeypatch, field, bad):
    from hermes_state_terminal import ADMISSION_PREFIX
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        row = legacy_accept_and_retire(db, epoch, monkeypatch)
        key = ADMISSION_PREFIX + row['admission_id']
        saved = json.loads(db._conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
        saved[field] = bad
        db._execute_write(lambda c: c.execute('UPDATE state_meta SET value=? WHERE key=?', (json.dumps(saved), key)))
        result = prepare(db)
        assert result['error'] == 'scoped_terminal_corruption'
        assert result['coverage_complete'] and not result['complete']
        assert lookup(db, sid='unrelated') is None
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='old-member')


def test_migration_interleaved_retirement_projects_before_erasure(tmp_path, monkeypatch):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        for n in range(5):
            legacy_accept_and_retire(db, epoch, monkeypatch, sid=f'old-{n}', retire=False)
        import hermes_state_logical_attempts as index
        assert not index.prepare_logical_attempt_index(db, batch_size=1)['complete']
        assert db.delete_session('old-4')  # outside the already-scanned cursor
        assert prepare(db, batch=1)['complete']
        assert lookup(db, sid='old-4', task='old-4')['request_id'] == 'old-4'
        assert lookup(db, sid='old-4', scope='new-home-epoch', task='old-4') is None


def test_duplicate_logical_admission_race_is_one_canonical_acceptance(tmp_path):
    path = tmp_path / 'state.db'
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.ensure_session('member', source='api')
        assert prepare(db)['complete']
        barrier = Barrier(2)
        def race(request):
            with SessionDB(path) as other:
                barrier.wait(10)
                try:
                    return rt.admit_session_input(other, epoch=epoch, principal_id='api',
                        session_id='member', request_id=request, payload=payload())['admission_id']
                except rt.RuntimeStoreError as exc:
                    return exc.reason
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(race, ['one', 'two']))
        assert results.count('admission_conflict') == 1
        assert db._conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 1
        assert lookup(db)['admission_id'] in results


def test_duplicate_legacy_attempts_are_ambiguity_not_first_match(tmp_path, monkeypatch):
    import hermes_state_logical_attempts as index
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        with monkeypatch.context() as legacy:
            legacy.setattr(index, 'project_admission', lambda *a, **kw: None)
            accept(db, epoch, request='first')
            accept(db, epoch, request='second')
        assert prepare(db)['complete']
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db)
        assert lookup(db, task='fresh') is None


def test_preparation_does_not_bless_damaged_writer_trigger(tmp_path):
    import hermes_state_logical_attempts as index
    with SessionDB(tmp_path / 'state.db') as db:
        assert prepare(db)['complete']
        def damage(c):
            c.execute('DROP TRIGGER logical_attempt_live_insert')
            c.execute('CREATE TRIGGER logical_attempt_live_insert AFTER INSERT ON session_admissions BEGIN SELECT 1; END')
        db._execute_write(damage)
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            index.prepare_logical_attempt_index(db, batch_size=2)
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db)


@pytest.mark.parametrize('retired', [False, True])
def test_replace_and_drop_cannot_turn_projection_loss_into_new(tmp_path, retired):
    import hermes_state_logical_attempts as index
    path = tmp_path / 'state.db'
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        accepted = accept(db, epoch)
        assert prepare(db)['complete']
        if retired:
            settle(db, epoch)
            assert db.delete_session('member')
        saved = dict(db._conn.execute('SELECT * FROM logical_attempts WHERE admission_id=?',
                                     (accepted['admission_id'],)).fetchone())
        saved['task_id'] = 'forged-task'
        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(lambda c: c.execute(
                f'INSERT OR REPLACE INTO logical_attempts({",".join(saved)}) VALUES({",".join("?" for _ in saved)})',
                tuple(saved.values())))
        db._execute_write(lambda c: c.execute('DROP TABLE logical_attempts'))
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, task='fresh')
    with SessionDB(path) as db:
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, task='fresh')
        assert prepare(db)['complete']
        if retired:
            with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
                lookup(db, task='fresh')
        else:
            assert lookup(db)['admission_id'] == accepted['admission_id']
            assert lookup(db, task='fresh') is None


def test_replaced_database_is_unavailable_not_stale_absence(tmp_path):
    path = tmp_path / 'state.db'
    replacement = tmp_path / 'replacement.db'
    with SessionDB(replacement) as other:
        assert prepare(other)['complete']
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        accept(db, epoch)
        assert prepare(db)['complete']
        replacement.replace(path)
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, task='fresh')


def test_profile_coverage_and_logical_identity_never_cross_databases(tmp_path):
    with SessionDB(tmp_path / 'a.db') as a, SessionDB(tmp_path / 'b.db') as b:
        epoch = rt.begin_runtime_epoch(a, instance_id='a')
        row = accept(a, epoch)
        assert prepare(a)['complete'] and prepare(b)['complete']
        assert lookup(a)['admission_id'] == row['admission_id']
        assert lookup(b) is None
        assert lookup(a)['admission_id'] == row['admission_id']


def test_scoped_live_damage_never_becomes_absence_or_foreign_hold(tmp_path):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        row = accept(db, epoch)
        assert prepare(db)['complete']
        db._execute_write(lambda c: c.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
                                              ('{damaged', row['admission_id'])))
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, task='fresh')
        assert lookup(db, sid='unrelated') is None
        failed = prepare(db)
        assert failed['error'] == 'scoped_live_corruption' and not failed['complete']
        assert lookup(db, sid='unrelated') is None

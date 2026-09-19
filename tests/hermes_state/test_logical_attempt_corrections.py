"""Review regressions: owned evidence loss, fair repair, exact generations."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt
import hermes_state_logical_attempts as index
from hermes_state_terminal import ADMISSION_PREFIX
from tests.hermes_state.test_logical_attempt_index import accept, lookup, payload, prepare, settle
from tests.hermes_state.test_logical_attempt_migration import legacy_accept_and_retire


def damage_terminal(db, row, *, unknown=False):
    key = ADMISSION_PREFIX + row['admission_id']
    saved = json.loads(db._conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
    damaged = '{lost-json' if unknown else json.dumps(dict(saved, payload_digest='broken'))
    db._execute_write(lambda c: c.execute('UPDATE state_meta SET value=? WHERE key=?', (damaged, key)))


@pytest.mark.parametrize('kind', ['scoped-terminal', 'unknown-terminal', 'scoped-live'])
@pytest.mark.parametrize('first_opener', ['session', 'raw'])
@pytest.mark.parametrize('lost', [
    'TABLE logical_attempt_dirty', 'TABLE logical_attempt_coverage',
    'INDEX logical_attempt_dirty_scope', 'INDEX logical_attempt_exact',
    'TRIGGER logical_attempt_terminal_update', 'TRIGGER logical_attempt_live_update',
])
def test_owned_evidence_loss_reopen_requires_bounded_reinventory(tmp_path, monkeypatch, kind, first_opener, lost):
    path = tmp_path / 'state.db'
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        row = legacy_accept_and_retire(db, epoch, monkeypatch, retire=kind != 'scoped-live')
        if kind == 'scoped-live':
            db._execute_write(lambda c: c.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
                                                 ('{lost-json', row['admission_id'])))
        else:
            damage_terminal(db, row, unknown=kind == 'unknown-terminal')
        # Enough intact inventory to distinguish bounded reconstruction from
        # silently blessing the previous completed cursors on normal reopen.
        for n in range(3):
            accept(db, epoch, sid=f'control-{n}', request=f'control-{n}')
        result = prepare(db)
        assert not result['complete']
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='old-member')
        db._execute_write(lambda c: c.execute('DROP ' + lost))
    if first_opener == 'raw':
        from hermes_state_schema import reconcile_state_schema
        # Async delegation opens through this shared initializer with tuple rows.
        # It must invalidate before repairing an owned guard/index, not conceal
        # that repair from the later SessionDB opener.
        with sqlite3.connect(path) as conn:
            assert conn.row_factory is None
            reconcile_state_schema(conn)
            assert conn.execute('SELECT count(*) FROM logical_attempt_coverage').fetchone()[0] == 0
    with SessionDB(path) as db:
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='unrelated')
        step = index.prepare_logical_attempt_index(db, batch_size=1)
        assert step['processed'] <= 1 and not step['coverage_complete']
        for _ in range(12):
            step = index.prepare_logical_attempt_index(db, batch_size=1)
            assert step['processed'] <= 1
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='old-member')
        if kind == 'unknown-terminal':
            assert not step['coverage_complete']
            with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
                lookup(db, sid='unrelated')
        else:
            assert step['coverage_complete']
            assert lookup(db, sid='unrelated') is None
        assert not step['complete'] and step['pending']


def test_corrupt_first_hold_cannot_starve_repair_or_interleaved_work_across_restart(tmp_path, monkeypatch):
    path = tmp_path / 'state.db'
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        evidence = []
        for sid in ('left', 'right'):
            row = legacy_accept_and_retire(db, epoch, monkeypatch, sid=sid, retire=False)
            raw = dict(db._conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (row['admission_id'],)).fetchone())
            with monkeypatch.context() as legacy:
                legacy.setattr(index, 'project_admission', lambda *a, **kw: None)
                assert db.delete_session(sid)
            evidence.append(raw)
        poison, repairable = sorted(evidence, key=lambda row: row['admission_id'])
        for row in evidence:
            damage_terminal(db, row)
        newcomer = accept(db, epoch, sid='newcomer', request='newcomer')
        assert prepare(db)['coverage_complete']
        db._execute_write(lambda c: c.execute('UPDATE state_meta SET value=? WHERE key=?',
            (json.dumps(repairable), ADMISSION_PREFIX + repairable['admission_id'])))
        # First cursor step must spend only its one-record budget on the poison.
        step = index.prepare_logical_attempt_index(db, batch_size=1)
        assert step['processed'] == 1
        start = Event()
        def concurrent_obligation():
            with SessionDB(path) as other:
                start.wait(5)
                other._execute_write(lambda c: c.execute(
                    'UPDATE session_admissions SET payload_json=payload_json WHERE admission_id=?', (newcomer['admission_id'],)))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(concurrent_obligation)
            start.set()
            future.result(10)
        # New live work sorts BEFORE the terminal cursor. Do not reset the
        # cursor in the test; a durable wrap must eventually revisit it.
    for _ in range(8):
        with SessionDB(path) as db:
            step = index.prepare_logical_attempt_index(db, batch_size=1)
            assert step['processed'] <= 1
    with SessionDB(path) as db:
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid=poison['target_session_id'])
        restored = lookup(db, sid=repairable['target_session_id'], task=repairable['target_session_id'])
        assert restored['admission_id'] == repairable['admission_id']
        assert lookup(db, sid='newcomer')['admission_id'] == newcomer['admission_id']
        assert db._conn.execute('SELECT session_id FROM logical_attempt_dirty WHERE source=? AND admission_id=?',
                               ('terminal', poison['admission_id'])).fetchone()[0] == poison['target_session_id']
        assert step['coverage_complete'] and step['pending'] and not step['complete']


@pytest.mark.parametrize('generation', [2**63 - 1, 2**63, 2**63 + 1, 10**700 + 1],
                         ids=['sqlite-max', 'sqlite-overflow', 'adjacent-overflow', 'large-decimal'])
def test_exact_large_generation_new_admission_retirement_and_reopen(tmp_path, generation):
    path = tmp_path / 'state.db'
    def find(db, value=generation, scope='signed-epoch-one', task='task'):
        return index.lookup_logical_attempt(db, principal_id='api', session_id='member',
            owner_scope=scope, task_id=task, execution_generation=value)
    value = payload()
    value['api_turn_v1']['settings']['room_dispatch']['execution_generation'] = generation
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        db.ensure_session('member', source='api')
        assert prepare(db)['complete'] and find(db) is None
        row = rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='member', request_id='large', payload=value)
        assert find(db)['execution_generation'] == generation
        assert find(db, generation + 1) is None
        assert find(db, scope='different') is None and find(db, task='different') is None
        with pytest.raises(rt.RuntimeStoreError, match='admission_conflict'):
            rt.admit_session_input(db, epoch=epoch, principal_id='api', session_id='member', request_id='duplicate', payload=value)
        settle(db, epoch)
        assert db.delete_session('member')
    with SessionDB(path) as db:
        assert prepare(db)['complete']
        found = find(db)
        assert found['admission_id'] == row['admission_id'] and found['execution_generation'] == generation
        assert find(db, generation + 1) is None


def test_prior_index_version_reprepares_without_rewriting_erased_exact_identity(tmp_path):
    path = tmp_path / 'state.db'
    with SessionDB(path) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        accepted = accept(db, epoch)
        settle(db, epoch)
        assert db.delete_session('member')
        for n in range(3):
            accept(db, epoch, sid=f'live-{n}', request=f'live-{n}')
        assert prepare(db)['complete']
        before = tuple(db._conn.execute('SELECT * FROM logical_attempts WHERE admission_id=?',
                                        (accepted['admission_id'],)).fetchone())
        # v1's original tables/columns are unchanged: remove only the v2
        # auxiliary cursor/anchor and mark the previous preparation version.
        def old_local_index(c):
            c.execute('DROP TABLE logical_attempt_reconcile')
            c.execute("DELETE FROM logical_attempt_dirty WHERE source=''")
            c.execute('UPDATE logical_attempt_coverage SET version=1')
        db._execute_write(old_local_index)
    with SessionDB(path) as db:
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db)
        step = index.prepare_logical_attempt_index(db, batch_size=1)
        assert step['processed'] == 1 and not step['coverage_complete']
        assert prepare(db, batch=1)['complete']
        assert lookup(db)['admission_id'] == accepted['admission_id']
        assert tuple(db._conn.execute('SELECT * FROM logical_attempts WHERE admission_id=?',
                                     (accepted['admission_id'],)).fetchone()) == before
        assert db._conn.execute('SELECT version FROM logical_attempt_coverage').fetchone()[0] == index.VERSION
        assert db._conn.execute("SELECT count(*) FROM logical_attempt_dirty WHERE source!=''").fetchone()[0] == 0


def test_dirty_recreation_anchor_invalidates_even_before_another_normal_open(tmp_path, monkeypatch):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        row = legacy_accept_and_retire(db, epoch, monkeypatch)
        damage_terminal(db, row, unknown=True)
        assert not prepare(db)['coverage_complete']
        db._execute_write(lambda c: c.execute('DROP TABLE logical_attempt_dirty'))
        # Execute the actual DDL, not normal open's preinstallation invalidation.
        # Its new dirty-store anchor must independently invalidate the old cookie.
        db._conn.executescript(index.SCHEMA_SQL)
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='unrelated')
        assert not prepare(db)['coverage_complete']
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid='unrelated')


def test_unknown_hold_does_not_starve_scoped_repair(tmp_path, monkeypatch):
    with SessionDB(tmp_path / 'state.db') as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        evidence = []
        for sid in ('unknown-first', 'repair-second'):
            row = legacy_accept_and_retire(db, epoch, monkeypatch, sid=sid, retire=False)
            raw = dict(db._conn.execute('SELECT * FROM session_admissions WHERE admission_id=?', (row['admission_id'],)).fetchone())
            with monkeypatch.context() as legacy:
                legacy.setattr(index, 'project_admission', lambda *a, **kw: None)
                assert db.delete_session(sid)
            evidence.append(raw)
        poison, repairable = sorted(evidence, key=lambda row: row['admission_id'])
        damage_terminal(db, poison, unknown=True)
        damage_terminal(db, repairable)
        assert not prepare(db)['coverage_complete']
        db._execute_write(lambda c: c.execute('UPDATE state_meta SET value=? WHERE key=?',
            (json.dumps(repairable), ADMISSION_PREFIX + repairable['admission_id'])))
        for _ in range(6):
            step = index.prepare_logical_attempt_index(db, batch_size=1)
            assert step['processed'] <= 1 and not step['coverage_complete']
        assert db._conn.execute('SELECT 1 FROM logical_attempts WHERE admission_id=?',
                                (repairable['admission_id'],)).fetchone()
        assert db._conn.execute('SELECT session_id FROM logical_attempt_dirty WHERE source=? AND admission_id=?',
                                ('terminal', poison['admission_id'])).fetchone()[0] is None
        with pytest.raises(rt.RuntimeStoreError, match='storage_unavailable'):
            lookup(db, sid=repairable['target_session_id'], task=repairable['target_session_id'])
        db._execute_write(lambda c: c.execute('UPDATE state_meta SET value=? WHERE key=?',
            (json.dumps(poison), ADMISSION_PREFIX + poison['admission_id'])))
        assert prepare(db, batch=1)['complete']
        assert lookup(db, sid=repairable['target_session_id'], task=repairable['target_session_id'])['admission_id'] == repairable['admission_id']

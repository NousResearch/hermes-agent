"""Explicit deletes retire terminals atomically; busy/unknown rows refuse the batch."""
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tests.hermes_state.raw_maintenance_helpers import ended, ledger, snapshot


@pytest.mark.parametrize('kind', ['admission-live', 'admission-unknown', 'worker-live', 'worker-unknown'])
@pytest.mark.parametrize('operation', ['single', 'bulk', 'verified', 'delegate', 'delegate-bulk', 'writer-arrival'])
def test_explicit_delete_refuses_atomically_with_ledger_reference(tmp_path, monkeypatch, kind, operation):
    with SessionDB(tmp_path / 'state.db') as db:
        ended(db, 'protected')
        ended(db, 'legacy')
        target = 'protected'
        if operation.startswith('delegate'):
            ended(db, 'delegate', parent='protected', delegate=True)
            ended(db, 'grandchild', parent='delegate', delegate=True)
            target = 'grandchild'
            db._execute_write(lambda conn: ledger(conn, 'protected', 'admission-terminal'))
        if operation in {'bulk', 'delegate-bulk'}:
            db._execute_write(lambda conn: ledger(conn, 'legacy', 'worker-terminal'))
        for sid in ('protected', 'legacy', target):
            db.append_message(sid, 'user', 'history fixture')
            (tmp_path / (sid + '.json')).write_text('transcript fixture', encoding='utf-8')
        # Opaque pre-existing metadata is not an acceptance/closing-result simulation.
        db.set_meta('fixture-do-not-change', 'original')
        expected = db.get_session_delete_targets('protected')
        if operation != 'writer-arrival':
            db._execute_write(lambda conn: ledger(conn, target, kind))
        before = snapshot(db)
        if operation == 'writer-arrival':
            write = db._execute_write
            def arriving(callback, **kwargs):
                def joined(conn):
                    ledger(conn, target, kind)
                    return callback(conn)
                return write(joined, **kwargs)
            monkeypatch.setattr(db, '_execute_write', arriving)
        reason = 'unknown_execution' if kind.endswith('unknown') else 'session_busy'
        with pytest.raises(RuntimeStoreError, match=reason) as refused:
            if operation in {'bulk', 'delegate-bulk'}:
                db.delete_sessions(['legacy', 'protected', 'missing'], sessions_dir=tmp_path)
            else:
                db.delete_session('protected', sessions_dir=tmp_path,
                    expected_delete_ids=expected if operation == 'verified' else None)
        assert refused.value.reason == reason
        assert snapshot(db) == before
        assert all((tmp_path / (sid + '.json')).read_text(encoding='utf-8') == 'transcript fixture'
                   for sid in ('protected', 'legacy', target))


@pytest.mark.parametrize('kind', ['admission-live', 'worker-live', 'admission-unknown', 'worker-unknown'])
@pytest.mark.parametrize('operation', ['prune', 'empty', 'ghost', 'if-empty'])
def test_cleanup_skips_and_reports_ledger_rows_but_collects_legacy(tmp_path, monkeypatch, kind, operation):
    with SessionDB(tmp_path / 'state.db') as db:
        ended(db, 'protected')
        ended(db, 'legacy')
        db._execute_write(lambda conn: ledger(conn, 'protected', kind))
        before = db.get_session('protected')
        untouched = {key: value for key, value in snapshot(db).items() if key != 'sessions'}
        for sid in ('protected', 'legacy'):
            (tmp_path / (sid + '.json')).write_text('transcript fixture', encoding='utf-8')
        preview = {}
        assert db.count_empty_sessions(report=preview) == 1
        assert preview == {'skipped_protected': 1}
        assert [r['id'] for r in db.list_prune_candidates(
            older_than_days=None, exclude_ledger_owned=True, report=preview)] == ['legacy']
        assert preview == {'skipped_protected': 1}
        # Archive/listing candidates keep their existing non-destructive semantics.
        assert {r['id'] for r in db.list_prune_candidates(older_than_days=None)} == {'protected', 'legacy'}
        report = {}
        if operation == 'prune':
            count = db.prune_sessions(older_than_days=None, sessions_dir=tmp_path, report=report)
        elif operation == 'empty':
            count = db.delete_empty_sessions(sessions_dir=tmp_path, report=report)
        elif operation == 'ghost':
            count = db.prune_empty_ghost_sessions(sessions_dir=tmp_path, report=report)
        else:
            reason = 'unknown_execution' if kind.endswith('unknown') else 'session_busy'
            with pytest.raises(RuntimeStoreError, match=reason):
                db.delete_session_if_empty('protected', sessions_dir=tmp_path, report=report)
            assert report == {}
            count = db.delete_session_if_empty('legacy', sessions_dir=tmp_path)
        assert count == 1
        if operation != 'if-empty':
            assert report == {'removed': 1, 'skipped_protected': 1}
        assert db.get_session('protected') == before and db.get_session('legacy') is None
        assert {key: value for key, value in snapshot(db).items() if key != 'sessions'} == untouched
        assert (tmp_path / 'protected.json').read_text(encoding='utf-8') == 'transcript fixture'
        assert not (tmp_path / 'legacy.json').exists()
        assert db._read_all('PRAGMA foreign_key_check') == []


@pytest.mark.parametrize('kind', ['admission-terminal', 'worker-terminal'])
@pytest.mark.parametrize('operation', ['single', 'bulk', 'verified', 'if-empty'])
def test_explicit_delete_retires_exact_cascade_and_preserves_branches(tmp_path, kind, operation):
    import json
    from hermes_state_terminal import ADMISSION_PREFIX, WORKER_PREFIX, identity_key
    with SessionDB(tmp_path / 'state.db') as db:
        ended(db, 'root')
        targets = ['root']
        if operation != 'if-empty':
            ended(db, 'delegate', parent='root', delegate=True)
            ended(db, 'grandchild', parent='delegate', delegate=True)
            ended(db, 'branch', parent='delegate')
            ended(db, 'branch-delegate', parent='branch', delegate=True)
            db._execute_write(lambda conn: ledger(conn, 'branch', 'worker-unknown'))
            targets += ['delegate', 'grandchild']
        if operation == 'bulk':
            ended(db, 'independent')
            targets += ['independent']
        for sid in targets:
            db._execute_write(lambda conn, sid=sid: ledger(conn, sid, kind))
            (tmp_path / (sid + '.json')).write_text('transcript fixture')
        report = {}
        if operation == 'verified':
            before = snapshot(db)
            assert not db.delete_session('root', expected_delete_ids=['root'], sessions_dir=tmp_path)
            assert snapshot(db) == before
        if operation == 'if-empty':
            assert db.delete_session_if_empty('root', sessions_dir=tmp_path, report=report)
            assert report == {'removed': 1, 'skipped_protected': 0}
        elif operation == 'bulk':
            assert db.delete_sessions(['root', 'independent', 'root', 'missing'], sessions_dir=tmp_path) == 2
        else:
            assert db.delete_session('root', sessions_dir=tmp_path,
                expected_delete_ids=targets if operation == 'verified' else None)
        for sid in targets:
            assert db.get_session(sid) is None
            assert not (tmp_path / (sid + '.json')).exists()
            key = ADMISSION_PREFIX + 'a-' + sid if kind.startswith('admission') else WORKER_PREFIX + 'w-' + sid
            tombstone = json.loads(db.get_meta(key))
            assert tombstone['status'] == 'terminal'
            if kind.startswith('admission'):
                assert tombstone['payload_json'] == '{}' and tombstone['lineage_json'] == '[]'
                assert json.loads(db.get_meta(identity_key('fixture', sid, 'r-' + sid))) == 'a-' + sid
        if operation != 'if-empty':
            assert db.get_session('branch')['parent_session_id'] is None
            assert db.get_session('branch-delegate')['parent_session_id'] == 'branch'
            assert db._read_one("SELECT status FROM worker_executions WHERE session_id='branch'")[0] == 'unknown'
        assert db._read_all('PRAGMA foreign_key_check') == []


@pytest.mark.parametrize('kind', ['admission-terminal', 'worker-terminal'])
@pytest.mark.parametrize('operation', ['prune', 'empty', 'ghost'])
def test_sweep_retires_terminal_rows_and_reports_only_busy_skips(tmp_path, kind, operation):
    """Preconstructed metadata exercises retirement, never a turn or worker lifecycle."""
    import json
    from hermes_state_terminal import ADMISSION_PREFIX, WORKER_PREFIX
    with SessionDB(tmp_path / 'state.db') as db:
        for sid in ('terminal', 'busy', 'legacy'):
            ended(db, sid)
        db._execute_write(lambda conn: ledger(conn, 'terminal', kind))
        db._execute_write(lambda conn: ledger(conn, 'busy', kind.replace('terminal', 'live')))
        report = {}
        assert db.count_empty_sessions(report=report) == 2
        assert report == {'skipped_protected': 1}
        assert {r['id'] for r in db.list_prune_candidates(
            older_than_days=None, exclude_ledger_owned=True)} == {'terminal', 'legacy'}
        operations = {'prune': lambda: db.prune_sessions(older_than_days=None, report=report),
                      'empty': lambda: db.delete_empty_sessions(report=report),
                      'ghost': lambda: db.prune_empty_ghost_sessions(report=report)}
        assert operations[operation]() == 2
        assert report == {'removed': 2, 'skipped_protected': 1}
        assert db.get_session('terminal') is None and db.get_session('busy') is not None
        key = ADMISSION_PREFIX + 'a-terminal' if kind.startswith('admission') else WORKER_PREFIX + 'w-terminal'
        tombstone = json.loads(db.get_meta(key))
        assert tombstone['status'] == 'terminal'
        assert db._read_all('PRAGMA foreign_key_check') == []

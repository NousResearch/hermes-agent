"""Raw maintenance skips obligations; it never retires canonical ledgers."""
import pytest

from hermes_state import SessionDB
from tests.state.raw_maintenance_helpers import ended, ledger, snapshot


@pytest.mark.parametrize('kind', ['admission-live', 'admission-terminal', 'worker-live', 'worker-terminal'])
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
        with pytest.raises(ValueError, match='owning gateway') as refused:
            if operation in {'bulk', 'delegate-bulk'}:
                db.delete_sessions(['legacy', 'protected', 'missing'], sessions_dir=tmp_path)
            else:
                db.delete_session('protected', sessions_dir=tmp_path,
                    expected_delete_ids=expected if operation == 'verified' else None)
        assert refused.value.reason == 'runtime_coordination_required'
        assert snapshot(db) == before
        assert all((tmp_path / (sid + '.json')).read_text(encoding='utf-8') == 'transcript fixture'
                   for sid in ('protected', 'legacy', target))


@pytest.mark.parametrize('kind', ['admission-live', 'admission-terminal', 'worker-live', 'worker-terminal'])
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
            assert not db.delete_session_if_empty('protected', sessions_dir=tmp_path, report=report)
            assert report == {'removed': 0, 'skipped_protected': 1}
            count = db.delete_session_if_empty('legacy', sessions_dir=tmp_path)
        assert count == 1
        if operation != 'if-empty':
            assert report == {'removed': 1, 'skipped_protected': 1}
        assert db.get_session('protected') == before and db.get_session('legacy') is None
        assert {key: value for key, value in snapshot(db).items() if key != 'sessions'} == untouched
        assert (tmp_path / 'protected.json').read_text(encoding='utf-8') == 'transcript fixture'
        assert not (tmp_path / 'legacy.json').exists()
        assert db._read_all('PRAGMA foreign_key_check') == []

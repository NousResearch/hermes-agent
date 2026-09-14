"""Caller refusal/reporting uses real stores without runtime execution."""
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from tests.state.raw_maintenance_helpers import ended, ledger, snapshot


@pytest.mark.parametrize('action', ['delete', 'export', 'prune'])
def test_cli_reports_protected_history_without_partial_delete(tmp_path, monkeypatch, capsys, action):
    from hermes_cli import sessions_cmd
    from hermes_cli.session_filters import build_prune_filters
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        ended(db, 'protected')
        ended(db, 'legacy')
        db._execute_write(lambda conn: ledger(conn, 'protected', 'admission-terminal'))
        before = snapshot(db)
        close = db.close
        monkeypatch.setattr('hermes_state.SessionDB', lambda *args, **kwargs: db)
        monkeypatch.setattr(db, 'close', lambda: None)
        # The real command dispatcher owns the clean refusal, including verified export.
        if action == 'export':
            args = SimpleNamespace(sessions_action='export', session_id='protected', format='md',
                output=str(tmp_path / 'export'), delete_after_verified=True, yes=True,
                lineage='single', force=True, dry_run=False)
            monkeypatch.setattr(sessions_cmd, '_DB_HANDLERS', {'export': lambda store, ignored:
                sessions_cmd._export_markdown_single(store, args,
                    lambda sid, include_lineage: (store.export_session(sid), tmp_path / 'export.md'),
                    tmp_path, False)})
            monkeypatch.setattr('hermes_cli.session_export_md.verify_export_file', lambda *args: (True, ''))
        elif action == 'delete':
            args = SimpleNamespace(sessions_action='delete', session_id='protected', yes=True)
        else:
            args = SimpleNamespace(sessions_action='prune', yes=True, dry_run=False, source='tui',
                                   older_than=None, never_active=False, include_pinned=False)
            # Parse the ordinary filter contract; optional CLI attributes are absent.
            assert build_prune_filters(args)['source'] == 'tui'
        result = sessions_cmd.cmd_sessions(args)
        output = capsys.readouterr().out
        if action == 'prune':
            assert 'Pruned 1 session(s).' in output and 'Skipped 1 session(s)' in output
            assert db.get_session('protected') is not None and db.get_session('legacy') is None
        else:
            assert result == 1 and 'Refused:' in output and 'owning gateway' in output
            assert snapshot(db) == before
        monkeypatch.setattr(db, 'close', close)


@pytest.mark.asyncio
async def test_http_components_report_same_transaction_counts_and_bulk_refusal(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from hermes_cli import web_server_sessions
    from hermes_cli.web_routers import sessions
    from hermes_cli.web_models import BulkDeleteSessions, SessionPrune
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        for sid in ('protected', 'legacy'):
            ended(db, sid)
        db._execute_write(lambda conn: ledger(conn, 'protected', 'worker-terminal'))
        close = db.close
        monkeypatch.setattr(web_server_sessions, '_open_session_db_for_profile', lambda *args, **kwargs: db)
        monkeypatch.setattr(db, 'close', lambda: None)
        before = snapshot(db)
        with pytest.raises(HTTPException) as refused:
            await sessions.bulk_delete_sessions_endpoint(BulkDeleteSessions(ids=['legacy', 'protected']))
        assert refused.value.status_code == 409
        assert refused.value.detail['code'] == 'runtime_coordination_required'
        assert snapshot(db) == before
        assert await sessions.count_empty_sessions_endpoint() == {'count': 1, 'skipped_protected': 1}
        dry = sessions._prune_sessions(SessionPrune(older_than_days=1, dry_run=True))
        assert dry['matched'] == 1 and dry['skipped_protected'] == 1
        result = await sessions.delete_empty_sessions_endpoint()
        assert result == {'ok': True, 'deleted': 1, 'skipped_protected': 1}
        assert db.get_session('protected') == before['sessions'][0]
        monkeypatch.setattr(db, 'close', close)

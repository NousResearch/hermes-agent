"""``session.list`` carries what a session picker shows, and filters by folder with ``hermes -c``'s rule (#150091)."""
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_mutation_model_busy import _owner


@pytest.mark.asyncio
async def test_live_rows_carry_picker_fields_and_filter_by_workspace(tmp_path, monkeypatch):
    from hermes_agent.gateway.session_local_recovery import transcript_target

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        db.append_message(target, 'user', 'where is the bug?')
        db.append_message(target, 'assistant', 'in the parser')
        db.append_message(target, 'assistant', '', tool_calls=[{'id': 'c1', 'type': 'function',
                          'function': {'name': 'read_file', 'arguments': '{}'}}])
        from hermes_agent.agent.context_compressor import SUMMARY_PREFIX
        db.append_message(target, 'assistant', f'{SUMMARY_PREFIX}\n## Goal\nnot a reply')

        async def listing(**params):
            reply = await owner.dispatch({'id': 1, 'method': 'session.list', 'params': params})
            assert 'result' in reply, reply
            return reply['result']['sessions']

        [bare] = await listing()
        assert 'preview' not in bare and 'last_reply' not in bare  # a live-list poll pays for none of it
        [row] = await listing(picker=True)
        assert row['preview'] == 'where is the bug?'
        assert row['last_reply'] == 'in the parser'
        assert (row['cwd'], row['workspace'], row['pinned']) == (str(tmp_path), str(tmp_path), False)
        assert row['last_active_at'] >= db.get_session(target)['started_at']
        assert [r['session_id'] for r in await listing(workspace=str(tmp_path), picker=True)] == [ref.session_id]
        assert await listing(workspace=str(tmp_path / 'elsewhere')) == []
        assert 'error' in await owner.dispatch({'id': 2, 'method': 'session.list',
                                                'params': {'title': 'x', 'workspace': str(tmp_path)}})
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_stored_rows_carry_picker_fields_and_filter_by_repo_root(tmp_path, monkeypatch):
    from hermes_agent.gateway import run
    from hermes_agent.gateway.config import GatewayConfig
    from hermes_agent.gateway.session import SessionStore
    from hermes_agent.gateway.session_authority import initialize_session_authority
    from hermes_agent.gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    repo = tmp_path / 'repo'
    for sid, cwd in (('in-repo', repo / 'pkg'), ('elsewhere', tmp_path / 'other')):
        db.create_session(sid, source='cli', model='m', cwd=str(cwd))
        db.append_message(sid, 'user', f'{sid} question')
        db.append_message(sid, 'assistant', f'{sid} answer')
        db.end_session(sid, 'cli_exit')
    db._write_sql('UPDATE sessions SET git_repo_root=?, git_branch=? WHERE id=?', (str(repo), 'main', 'in-repo'))
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'owner'})
    try:
        reply = await native.dispatch({'id': 1, 'method': 'session.list',
                                       'params': {'scope': 'stored', 'workspace': str(repo), 'picker': True}})
        assert 'result' in reply, reply
        [row] = reply['result']['sessions']
        assert (row['id'], row['workspace'], row['git_branch'], row['cwd']) == (
            'in-repo', str(repo), 'main', str(repo / 'pkg'))
        assert (row['preview'], row['last_reply']) == ('in-repo question', 'in-repo answer')
        everything = await native.dispatch({'id': 2, 'method': 'session.list', 'params': {'scope': 'stored'}})
        assert {r['id'] for r in everything['result']['sessions']} == {'in-repo', 'elsewhere'}
    finally:
        await native.close()
        store.close_all_db_handles()

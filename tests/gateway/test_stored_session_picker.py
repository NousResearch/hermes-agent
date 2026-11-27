"""The native picker enumerates old transcripts without adopting them on read."""
from types import SimpleNamespace

import pytest
from tests.fixtures.storage_holder import history_binding_key, raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.mark.asyncio
async def test_stored_picker_survives_upgrade_and_resumes_without_copying(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    for sid, source in [('old-cli', 'cli'), ('old-tui', 'tui'), ('old-gui', 'gui'),
                        ('internal', 'tool'), ('bot', 'telegram'), ('hidden', 'cli'),
                        ('archived', 'tui'), ('foreign', 'cli'), ('imported-other', 'cli')]:
        seed_session(db, sid, source, model='old-model', cwd=str(tmp_path),
                     profile_name='other' if sid == 'foreign' else None)
        seed_messages(db, sid, {'role': 'user', 'content': sid + ' question'},
                      {'role': 'assistant', 'content': sid + ' answer'})
        db.sessions.end(sid, 'cli_exit')
    db.sessions.reopen('old-tui')
    assert db.coordination.try_acquire_compression_lock(session_id='old-tui', holder='fixture')
    try:
        db.lineage.publish_compression_child('old-tui', CompressionChild(session_id='tui-tip', source='tui', model='old-model'),
            [NewMessage.from_live(m) for m in ({'role': 'user', 'content': 'compressed question'},
                                               {'role': 'assistant', 'content': 'compressed answer'})],
            compression_lock_holder='fixture')
    finally:
        db.coordination.release_compression_lock(session_id='old-tui', holder='fixture')
    db.sessions.set_hidden('hidden', True)
    db.sessions.set_archived('archived', True)
    raw_holder(db)._write_sql('INSERT INTO state_meta(key,value) VALUES(?,?)',
                              (history_binding_key('imported-other'), '["other-profile","other-user"]'))
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'owner'})

    async def listing(connection, **params):
        return await connection.dispatch({'id': 1, 'method': 'session.list', 'params': params})

    before = raw_holder(db)._conn.total_changes
    result = await listing(native, scope='stored')
    assert 'result' in result, result
    assert result['result']['scope'] == 'stored'
    assert {r['id'] for r in result['result']['sessions']} == {'old-cli', 'tui-tip', 'old-gui'}
    assert raw_holder(db)._conn.total_changes == before
    assert authority.sessions == {}
    assert (await listing(native))['result'] == {'scope': 'live', 'sessions': []}
    assert len((await listing(native, scope='stored', limit=1))['result']['sessions']) == 1

    remote = AuthorityConnection(authority, object(), {'provider': 'remote', 'user_id': 'remote',
        'profile_id': str(tmp_path), 'instance_id': 'owner'}, operator=True)
    assert (await listing(remote, scope='stored'))['result']['sessions'] == []
    denied = await listing(native, scope='stored', profile='other')
    assert 'error' in denied

    resumed = await native.dispatch({'id': 2, 'method': 'session.resume', 'params': {'session_id': 'old-cli'}})
    assert 'result' in resumed, resumed
    assert [m['content'] for m in resumed['result']['messages']] == ['old-cli question', 'old-cli answer']
    assert raw_holder(db)._read_one('SELECT COUNT(*) FROM messages WHERE session_id=?', ('old-cli',))[0] == 2
    assert 'old-cli' not in {r['id'] for r in (await listing(native, scope='stored'))['result']['sessions']}
    assert 'old-cli' in {r['id'] for r in (await listing(native))['result']['sessions']}
    compressed = await native.dispatch({'id': 3, 'method': 'session.resume', 'params': {'session_id': 'tui-tip'}})
    assert 'result' in compressed, compressed
    messages = compressed['result']['messages']
    assert [m['content'] for m in messages if not m.get('archived')] == ['compressed question', 'compressed answer']
    assert [m['content'] for m in messages if m.get('archived')] == ['old-tui question', 'old-tui answer']

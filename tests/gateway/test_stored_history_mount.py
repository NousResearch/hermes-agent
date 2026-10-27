"""Old stored chats no loader restores open read-only for their owner (#143547 RT-30b, RT-48, RT-49, RT-54).

Pre-update cron runs, subagent transcripts, API rows without a binding, WebUI rows and rows the
foreign-session importer wrote answered ``not_found`` on ``session.resume`` and on delete. They
now mount read-only: the profile's owner reads and deletes them, a submit is refused by name, and
a caller that is not the owner still finds nothing.
"""
import json
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from hermes_state_runtime import RuntimeStoreError

ROWS = {
    'cron_job1_20261001_120000': ('cron', {}),
    'sub-1': ('subagent', {}),
    'api-1': ('api_server', {'session_key': 'agent:main:api_server:dm:api-1', 'chat_id': 'api-1'}),
    'webui-1': ('webui', {'session_key': 'webui-1', 'origin_json': json.dumps({'platform': 'webui'})}),
    'cc-1': ('claude-code', {'origin_json': json.dumps({'imported_from': {'tool': 'claude-code', 'path': '/x'}})}),
    'hook-1': ('webhook', {}),
}


async def _connect(tmp_path, monkeypatch, sid, **identity):
    """A fresh gateway over a state.db holding *sid*, and a native Desktop connection to it."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'm1', 'provider': 'openrouter'}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    source, columns = ROWS[sid]
    db.create_session(sid, source=source, model='m-old')
    db.append_message(sid, role='user', content='q')
    db.append_message(sid, role='assistant', content='a')
    for column, value in columns.items():
        db._write_sql(f'UPDATE sessions SET {column}=? WHERE id=?', (value, sid))
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _adapter_for_source=lambda source: None, _cached_agent_for=lambda route: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='i')
    native = {'user_id': 'owner', 'provider': 'local', 'profile_id': str(tmp_path), 'instance_id': 'i',
              'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control'],
              'native_bootstrap': True}
    return AuthorityConnection(authority, object(), {**native, **identity}), SessionRef(str(tmp_path), sid)


@pytest.mark.asyncio
@pytest.mark.parametrize('sid', sorted(ROWS))
async def test_old_stored_history_opens_read_only_for_its_owner(tmp_path, monkeypatch, sid):
    conn, ref = await _connect(tmp_path, monkeypatch, sid)

    opened = await conn.resume(ref, {})

    assert [m['content'] for m in opened['messages']] == ['q', 'a']
    with pytest.raises(RuntimeStoreError, match='session_read_only'):
        await conn.submit(ref, {'text': 'more', 'submission_id': 's1'})
    assert conn.authority.db.get_session(sid)['message_count'] == 2


@pytest.mark.asyncio
async def test_a_caller_that_is_not_the_owner_still_finds_nothing(tmp_path, monkeypatch):
    """A remote login is not evidence of ownership, before or after the owner opened the row."""
    from gateway.session_contract import Principal
    conn, ref = await _connect(tmp_path, monkeypatch, 'cron_job1_20261001_120000')
    login = Principal('basic:alice', ref.profile_id, frozenset({'session:read'}), 'web-socket')
    with pytest.raises(RuntimeStoreError, match='not_found'):
        conn.authority.authorize(login, ref, 'session:read')
    assert ref.session_id not in conn.authority.sessions
    await conn.resume(ref, {})
    with pytest.raises(RuntimeStoreError, match='not_found'):
        conn.authority.authorize(login, ref, 'session:read')


@pytest.mark.asyncio
async def test_the_owner_deletes_a_completed_subagent_history(tmp_path, monkeypatch):
    conn, ref = await _connect(tmp_path, monkeypatch, 'sub-1')
    row = conn.authority.db.get_session('sub-1')
    await conn.mutate(ref, {'session_id': 'sub-1', 'request_id': 'delete-1', 'expected_revision': row['runtime_revision'],
                            'expected_generation': row['runtime_generation'], 'operation': 'delete', 'payload': {}})
    assert conn.authority.db.get_session('sub-1') is None


async def _mutate(conn, ref, request_id, operation, payload):
    row = conn.authority.db.get_session(ref.session_id)
    return await conn.mutate(ref, {'session_id': ref.session_id, 'request_id': request_id, 'operation': operation,
                                   'payload': payload, 'expected_revision': row['runtime_revision'],
                                   'expected_generation': row['runtime_generation']})


@pytest.mark.asyncio
async def test_the_owner_archives_unarchives_and_pins_old_stored_history(tmp_path, monkeypatch):
    """RT-54/RT-56: these answered ``session_read_only`` on the mount, as a moved-on chat's old
    conversation answered ``session_superseded``."""
    conn, ref = await _connect(tmp_path, monkeypatch, 'cron_job1_20261001_120000')
    await _mutate(conn, ref, 'archive', 'archive', {'archived': True})
    assert conn.authority.db.get_session(ref.session_id)['archived'] == 1
    await _mutate(conn, ref, 'unarchive', 'archive', {'archived': False})
    await _mutate(conn, ref, 'pin', 'sidebar', {'pinned': True})
    row = conn.authority.db.get_session(ref.session_id)
    assert (row['archived'], row['pinned']) == (0, 1)


@pytest.mark.asyncio
async def test_a_stored_history_with_child_sessions_is_not_deleted(tmp_path, monkeypatch):
    """The delete took the run's delegate child with it and detached any other child, unseen."""
    conn, ref = await _connect(tmp_path, monkeypatch, 'cron_job1_20261001_120000')
    db = conn.authority.db
    db.create_session('sub-child', source='subagent', model='m-old')
    db._write_sql('UPDATE sessions SET parent_session_id=?, model_config=? WHERE id=?',
                  (ref.session_id, json.dumps({'_delegate_from': ref.session_id}), 'sub-child'))
    with pytest.raises(RuntimeStoreError, match='session_has_children'):
        await _mutate(conn, ref, 'delete', 'delete', {})
    assert db.get_session('sub-child')['parent_session_id'] == ref.session_id


@pytest.mark.asyncio
async def test_a_childless_continuation_whose_chain_has_a_delegate_is_not_deleted(tmp_path, monkeypatch):
    """The delete walks back to the compression parent and took that parent's delegate child too."""
    conn, root = await _connect(tmp_path, monkeypatch, 'api-1')
    db = conn.authority.db
    db._write_sql("UPDATE sessions SET end_reason='compression', ended_at=1 WHERE id=?", (root.session_id,))
    db.create_session('api-1b', source='api_server', model='m-old', parent_session_id=root.session_id)
    db.create_session('sub-child', source='subagent', model='m-old')
    db._write_sql('UPDATE sessions SET session_key=?, chat_id=? WHERE id=?',
                  (ROWS['api-1'][1]['session_key'], 'api-1', 'api-1b'))
    db._write_sql('UPDATE sessions SET parent_session_id=?, model_config=? WHERE id=?',
                  (root.session_id, json.dumps({'_delegate_from': root.session_id}), 'sub-child'))
    tip = SessionRef(root.profile_id, 'api-1b')
    await conn.resume(tip, {})
    with pytest.raises(RuntimeStoreError, match='session_has_children'):
        await _mutate(conn, tip, 'delete', 'delete', {})
    assert db.get_session('sub-child')['parent_session_id'] == root.session_id


@pytest.mark.asyncio
async def test_the_reads_a_viewer_sends_for_an_open_chat_answer_on_a_mount(tmp_path, monkeypatch):
    """Desktop polls the session-control store and loads its slash palette and model picker for the
    open chat; the mount has no source, and each of these answered ``internal error``."""
    conn, ref = await _connect(tmp_path, monkeypatch, 'cron_job1_20261001_120000')
    await conn.resume(ref, {})
    sid = {'session_id': ref.session_id}
    for method, params in (('session.control.read', sid), ('commands.catalog', sid),
                           ('config.get', {**sid, 'key': 'mtime'}), ('model.options', sid)):
        reply = await conn.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, (method, reply)
    from gateway.run import GatewayRunner
    runner = conn.authority.runner
    for name in ('_voice_key_for_source', '_voice_key', '_resolve_session_service_tier', '_resolve_session_key_or_none',
                 '_peek_session_state', '_load_service_tier'):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner))
    runner._adapter_profile_for_source = lambda source: None  # no adapters here
    # Desktop's automatic voice read on open, and the TUI's /fast status read.
    replies = {key: await conn.dispatch({'id': 1, 'method': 'config.get', 'params': {**sid, 'key': key}})
               for key in ('voice', 'fast')}
    assert {key: reply.get('result', {}).get('value') for key, reply in replies.items()} == {
        'voice': 'off', 'fast': 'normal'}, replies
    for command in ('fast status', 'model'):
        reply = await conn.dispatch({'id': 1, 'method': 'slash.exec', 'params': {**sid, 'command': command}})
        assert reply['error']['message'] == 'session_read_only', (command, reply)


@pytest.mark.asyncio
async def test_an_api_chat_opened_read_only_still_continues_through_the_api(tmp_path, monkeypatch):
    """Binding the viewed row replaced nothing: the read-only mount had no source to check. The viewer
    that opened it keeps receiving the bound conversation's events and detaches normally."""
    import asyncio
    import queue
    from gateway.config import Platform
    from gateway.session_api import bind_api_session
    conn, ref = await _connect(tmp_path, monkeypatch, 'api-1')
    frames = queue.Queue()
    conn.authority.events[conn.actor.transport_id] = SimpleNamespace(write=lambda frame: frames.put(frame) or True)
    opened = await conn.resume(ref, {})

    assert bind_api_session(conn.authority, 'api-1', caller_key='agent:main:api_server:dm:api-1') == ref

    live = conn.authority.sessions['api-1']
    assert live.source.platform == Platform.API_SERVER and not live.superseded
    live.event_stream.publish('api-1', {'text': 'bound'})
    frame = await asyncio.to_thread(frames.get, True, 5)
    assert frame['params']['payload'] == {'text': 'bound'}
    detached = await conn.detach(ref, {'session_id': 'api-1', 'subscription_id': opened['subscription_id']})
    assert detached['detached'] is True


@pytest.mark.asyncio
@pytest.mark.parametrize('sid', sorted(ROWS))
async def test_session_info_answers_on_a_mount(tmp_path, monkeypatch, sid):
    """The classic CLI's /new and /tools read session.info first; on a mount it answered ``internal error``."""
    conn, ref = await _connect(tmp_path, monkeypatch, sid)
    await conn.resume(ref, {})
    reply = await conn.dispatch({'id': 1, 'method': 'session.info', 'params': {'session_id': sid}})
    assert reply.get('result', {}).get('source') == ROWS[sid][0], reply


@pytest.mark.asyncio
@pytest.mark.parametrize('selected', ['tip', 'root'])
async def test_a_compressed_cron_history_deletes_as_one_chat(tmp_path, monkeypatch, selected):
    """Deleting one segment of an old cron run's compression chain removed that segment only: the
    other came back as its own chat, or stayed as a detached one."""
    conn, ref = await _connect(tmp_path, monkeypatch, 'cron_job1_20261001_120000')
    db, root = conn.authority.db, ref.session_id
    db._write_sql("UPDATE sessions SET end_reason='compression', ended_at=1 WHERE id=?", (root,))
    db.create_session('cron-tip', source='cron', model='m-old', parent_session_id=root)
    target = SessionRef(ref.profile_id, 'cron-tip' if selected == 'tip' else root)
    await conn.resume(target, {})
    await _mutate(conn, target, 'delete', 'delete', {})
    assert [sid for sid in (root, 'cron-tip') if db.get_session(sid) is not None] == []


@pytest.mark.asyncio
async def test_push_to_talk_reads_voice_off_on_a_mount(tmp_path, monkeypatch):
    """The TUI's push-to-talk gate on an opened stored chat raised AttributeError on the source-less mount."""
    from gateway import run_api
    from gateway.run import GatewayRunner
    from gateway.session_config import local_voice_enabled
    conn, ref = await _connect(tmp_path, monkeypatch, 'cron_job1_20261001_120000')
    await conn.resume(ref, {})
    runner = conn.authority.runner
    for name in ('_voice_key_for_source', '_voice_key'):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner))
    runner._adapter_profile_for_source = lambda source: None
    runner.session_authority = conn.authority
    monkeypatch.setattr(run_api, '_ACTIVE_RUNNER', runner)
    assert local_voice_enabled(ref.session_id) is False

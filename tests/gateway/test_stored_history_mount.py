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

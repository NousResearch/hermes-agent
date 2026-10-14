"""The owner runs a session's end-of-conversation lifecycle when its last viewer closes it.

Regression for #134889 LC-1/LC-2/LC-6. Main's classic CLI and TUI ran the memory providers'
``on_session_end`` commit and plugin ``on_session_finalize`` (which closes the Relay scope) in their own
process at exit and ``/new``. Under the gateway the client is a viewer with no memory manager, so
nothing ran until idle eviction; and a close sent while a turn was live left ``ended_at`` unset for good.
"""
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until


async def _owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    calls = []

    async def finalize(**kwargs):
        calls.append(('finalize', kwargs))
        return []

    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None,
                             _evict_cached_agent_at_boundary=lambda key: calls.append(('commit', key)),
                             _finalize_session_off_loop=finalize)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    return store._db, authority, calls


def _viewer(authority, tmp_path):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {
        'provider': 'local', 'user_id': 'native', 'profile_id': str(tmp_path), 'instance_id': 'owner'})


async def _call(conn, method, **params):
    reply = await conn.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return reply['result']


async def _open(db, authority, tmp_path):
    db.create_session('chat', source='cli', model='fake-model', cwd=str(tmp_path))
    db.append_message('chat', 'user', 'q')
    db.append_message('chat', 'assistant', 'a')
    viewer = _viewer(authority, tmp_path)
    resumed = await _call(viewer, 'session.resume', session_id='chat')
    db.reopen_session('chat')
    return viewer, resumed['subscription_id']


@pytest.mark.asyncio
async def test_last_viewer_close_commits_memory_and_finalizes_on_the_owner(tmp_path, monkeypatch):
    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    other = _viewer(authority, tmp_path)
    second = await _call(other, 'session.resume', session_id='chat')
    route = authority.sessions['chat'].route

    # Another viewer still has the conversation open: nothing ends.
    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='new_session')
    assert calls == []
    # A plain detach (a viewer leaving, not closing) never ends it either.
    await _call(other, 'session.detach', session_id='chat', subscription_id=second['subscription_id'])
    assert calls == [] and db.get_session('chat')['end_reason'] is None

    again = await _call(other, 'session.resume', session_id='chat')
    await _call(other, 'session.detach', session_id='chat', subscription_id=again['subscription_id'],
                end_reason='cli_close')
    assert db.get_session('chat')['end_reason'] == 'cli_close'
    assert calls == [('commit', route), ('finalize', {
        'session_id': 'chat', 'platform': 'cli', 'reason': 'cli_close', 'session_key': route})]


@pytest.mark.asyncio
async def test_close_while_a_turn_is_queued_completes_when_the_fifo_goes_idle(tmp_path, monkeypatch):
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import admit_session_input, cancel_session_input

    db, authority, calls = await _owner(tmp_path, monkeypatch)
    viewer, subscription = await _open(db, authority, tmp_path)
    row = admit_session_input(db, epoch=authority.epoch, principal_id='native', session_id='chat',
                              request_id='r1', payload={'text': 'still running'})

    await _call(viewer, 'session.detach', session_id='chat', subscription_id=subscription, end_reason='cli_close')
    assert db.get_session('chat')['ended_at'] is None and calls == []

    # The turn ends (here: cancelled) and the drain reaches its idle point.
    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
    authority._schedule(SessionRef(authority.profile_id, 'chat'))
    await async_wait_until(lambda: len(calls) == 2)
    assert db.get_session('chat')['end_reason'] == 'cli_close'
    assert [name for name, _ in calls] == ['commit', 'finalize']


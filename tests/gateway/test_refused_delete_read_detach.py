"""A refused picker delete must not leave the closed session in the live list (#134889 TUI-19).

The Ink picker and the classic CLI read a delete's CAS tuple with ``session.resume``, which attaches a
viewer. When the owner refuses the delete (stale revision, read-only grant) nothing retired the
session, so ``closed_and_unviewed`` stayed false while that viewer was subscribed and the closed chat
reappeared in the live section until the client exited. The clients now detach the read's
subscription on a refusal; this pins the owner side of that contract over the real dispatch: the
refused delete reopens the session in the live list, and a plain ``session.detach`` of the read's
subscription puts it back as a closed, resumable history row with its original end stamp.
"""
from types import SimpleNamespace

import pytest


async def _owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    return store._db, authority


def _viewer(authority, tmp_path, user='native'):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {
        'provider': 'local', 'user_id': user, 'profile_id': str(tmp_path), 'instance_id': 'owner'})


async def _call(conn, method, **params):
    reply = await conn.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return reply['result']


async def _live_ids(conn):
    return [row['id'] for row in (await _call(conn, 'session.list'))['sessions']]


@pytest.mark.asyncio
async def test_detaching_the_reads_subscription_after_a_refused_delete_restores_the_closed_row(tmp_path, monkeypatch):
    db, authority = await _owner(tmp_path, monkeypatch)
    db.create_session('closed', source='tui', model='fake-model', cwd=str(tmp_path))
    db.append_message('closed', 'user', 'q')
    db.append_message('closed', 'assistant', 'a')
    viewer = _viewer(authority, tmp_path)
    first = await _call(viewer, 'session.resume', session_id='closed')
    await _call(viewer, 'session.detach', session_id='closed', subscription_id=first['subscription_id'],
                end_reason='tui_close')
    stamped = db.get_session('closed')
    assert stamped['end_reason'] == 'tui_close'
    picker = _viewer(authority, tmp_path)
    assert await _live_ids(picker) == []

    # The picker's sequence: a CAS read, then a delete the owner refuses (stale revision).
    read = await _call(picker, 'session.resume', session_id='closed')
    refused = await picker.dispatch({'id': 2, 'method': 'session.mutate', 'params': {
        'session_id': 'closed', 'request_id': 'req-1', 'expected_revision': read['revision'] + 1,
        'expected_generation': read['execution_generation'], 'operation': 'delete', 'payload': {}}})
    assert refused['error']['data']['reason'] == 'revision_conflict', refused
    assert await _live_ids(picker) == ['closed']  # the bug: the read's viewer reopened it

    out = await _call(picker, 'session.detach', session_id='closed', subscription_id=read['subscription_id'])
    assert out['detached'] is True
    assert await _live_ids(picker) == []
    row = db.get_session('closed')
    assert (row['ended_at'], row['end_reason']) == (stamped['ended_at'], 'tui_close')
    stored = await _call(picker, 'session.list', scope='stored')
    assert [r['id'] for r in stored['sessions']] == ['closed']




@pytest.mark.asyncio
async def test_the_release_detaches_by_the_id_the_read_resolved_to(tmp_path, monkeypatch):
    """A compressed session: the picker's history row names a segment (or a legacy root), and the
    owner's read resolves it elsewhere and subscribes under the RESOLVED id. The clients release by
    ``snapshot.session_id``; a detach by the id they sent is a no-op that leaves the viewer attached."""
    from gateway.session_controls import AuthorityConnection
    from tests.gateway.test_local_resume_compression_child import _compressed
    compressed = await _compressed(tmp_path, monkeypatch)
    authority = compressed.warm
    picker = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {
        'provider': 'local', 'user_id': 'owner', 'profile_id': 'fixture', 'instance_id': authority.instance_id,
        'capabilities': ['session:create', 'session:read', 'session:control']})
    read = await _call(picker, 'session.resume', session_id='child-1')  # the id the client holds
    assert read['session_id'] == compressed.owner                      # the owner resolved it elsewhere
    by_sent = await _call(picker, 'session.detach', session_id='child-1', subscription_id=read['subscription_id'])
    assert by_sent['detached'] is False
    by_resolved = await _call(picker, 'session.detach', session_id=read['session_id'],
                              subscription_id=read['subscription_id'])
    assert by_resolved['detached'] is True, by_resolved

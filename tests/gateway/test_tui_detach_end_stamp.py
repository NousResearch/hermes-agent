"""The TUI's close of a canonical session stamps the row as main's sidecar did (PR #106742 rows 28/35/38).

Main's TUI closed its session on ``/new`` / resume-switch (``tui_close``) and on exit (``tui_shutdown``),
and the picker then listed it as resumable rather than live. On the canonical owner the TUI detaches with
that ``end_reason``; the owner stamps it only when no other viewer remains and the FIFO is idle, and the
picker's live feed drops a closed, idle, unviewed session into the stored (resumable) feed.
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


def _native(authority, tmp_path):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {
        'provider': 'local', 'user_id': 'native', 'profile_id': str(tmp_path), 'instance_id': 'owner'})


async def _call(conn, method, **params):
    reply = await conn.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return reply['result']


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['tui_close', 'tui_shutdown'])
async def test_tui_detach_stamps_end_reason_and_moves_row_to_resumable(tmp_path, monkeypatch, reason):
    db, authority = await _owner(tmp_path, monkeypatch)
    db.create_session('old-tui', source='tui', model='fake-model', cwd=str(tmp_path))
    db.append_message('old-tui', 'user', 'q')
    db.append_message('old-tui', 'assistant', 'a')
    viewer = _native(authority, tmp_path)
    resumed = await _call(viewer, 'session.resume', session_id='old-tui')
    # Opening a closed chat reopens it only on the first turn; reopen to model an active session.
    db.reopen_session('old-tui')
    live = await _call(viewer, 'session.list')
    assert [(r['id'], r['model']) for r in live['sessions']] == [('old-tui', 'fake-model')]

    # A second viewer holds the session open: no stamp.
    other = _native(authority, tmp_path)
    second = await _call(other, 'session.resume', session_id='old-tui')
    out = await _call(viewer, 'session.detach', session_id='old-tui',
                      subscription_id=resumed['subscription_id'], end_reason=reason)
    assert out['detached'] is True
    assert db.get_session('old-tui')['end_reason'] is None

    await _call(other, 'session.detach', session_id='old-tui',
                subscription_id=second['subscription_id'], end_reason=reason)
    row = db.get_session('old-tui')
    assert (row['end_reason'], row['ended_at'] is not None, row['source']) == (reason, True, 'tui')

    fresh = _native(authority, tmp_path)
    assert (await _call(fresh, 'session.list'))['sessions'] == []
    stored = await _call(fresh, 'session.list', scope='stored')
    assert [r['id'] for r in stored['sessions']] == ['old-tui']


@pytest.mark.asyncio
async def test_plain_detach_never_stamps(tmp_path, monkeypatch):
    db, authority = await _owner(tmp_path, monkeypatch)
    db.create_session('old-tui', source='tui', model='fake-model', cwd=str(tmp_path))
    db.append_message('old-tui', 'user', 'q')
    viewer = _native(authority, tmp_path)
    resumed = await _call(viewer, 'session.resume', session_id='old-tui')
    await _call(viewer, 'session.detach', session_id='old-tui', subscription_id=resumed['subscription_id'])
    assert db.get_session('old-tui')['end_reason'] is None

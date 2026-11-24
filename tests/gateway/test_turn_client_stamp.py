"""The owner stamps each admitted turn with the client its connection announced (``surface_v1.client``).

Driven through the native socket's real dispatch: the hello (``client.capabilities``), then
``prompt.submit``; the admission row and the running turn are read back.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fixtures.storage_seed import seed_session

DESKTOP = {'name': 'hermes-desktop', 'version': '2.1.0', 'kind': 'desktop'}


def _peer():
    """A real ``WSTransport`` (its own attributes) whose frames are collected instead of sent."""
    from tui_gateway.ws import WSTransport

    class Peer(WSTransport):
        def write(self, obj):
            self.frames.append(obj)
            return True

    peer = Peer(object(), asyncio.get_running_loop(), auth_identity={})
    peer.frames = []
    return peer


async def _authority(home, monkeypatch, answer):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority

    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('HERMES_HOME', str(home))
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_session_db=store.persistence, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='owner')
    seed_session(authority.persistence, 's', 'telegram')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
    return authority


class _Client:
    def __init__(self, authority, identity=None):
        from gateway.session_controls import AuthorityConnection
        self.transport = _peer()
        self.connection = AuthorityConnection(authority, self.transport, {'user_id': 'owner', **(identity or {})})
        self._rid = 0

    async def call(self, method, **params):
        from tui_gateway.ws import _dispatch_request
        self._rid += 1
        return await _dispatch_request(self.connection, {'jsonrpc': '2.0', 'id': self._rid, 'method': method,
                                                         'params': params}, method, self.transport)


def _rows(authority):
    return {row.request_id: row.payload for row in authority.persistence.admissions.list(session_id='s', pending_only=False)}


def _recorder(seen):
    from gateway.session_surface import surface_turn_client, surface_turn_note

    async def answer(event):
        seen.append((event.text, surface_turn_client(), surface_turn_note(SimpleNamespace(valid_tool_names=set()))))
        return 'ok'
    return answer


@pytest.mark.asyncio
async def test_an_announced_connection_stamps_its_turn_and_an_unannounced_one_commits_today_s_payload(
        tmp_path, monkeypatch):
    seen = []
    authority = await _authority(tmp_path / 'home', monkeypatch, _recorder(seen))
    plain, desktop = _Client(authority), _Client(authority)
    try:
        assert (await desktop.call('client.capabilities', client=DESKTOP))['result']['client_activity'] is True
        for client, request_id in ((plain, 'typed'), (desktop, 'stamped')):
            await client.call('session.resume', session_id='s')
            reply = await client.call('prompt.submit', session_id='s', submission_id=request_id, text='same words')
            assert reply['result']['status'] == 'queued', reply
            await authority.sessions['s'].task
    finally:
        await plain.connection.close()
        await desktop.connection.close()

    rows = _rows(authority)
    assert rows['stamped']['surface_v1'] == {'client': DESKTOP}
    assert 'surface_v1' not in rows['typed'], 'an unannounced connection commits exactly what it did before'
    assert {key: value for key, value in rows['stamped'].items() if key != 'surface_v1'} == rows['typed']
    # The running turn reads the stamp; the model-bound note stays empty, as for a plain window turn.
    assert seen == [('same words', None, ''), ('same words', DESKTOP, '')]


@pytest.mark.asyncio
async def test_a_client_sent_on_prompt_submit_is_refused(tmp_path, monkeypatch):
    authority = await _authority(tmp_path / 'home', monkeypatch, _recorder([]))
    client = _Client(authority)
    try:
        await client.call('client.capabilities', client=DESKTOP)
        await client.call('session.resume', session_id='s')
        refused = await client.call('prompt.submit', session_id='s', submission_id='forged', text='x',
                                    client={'name': 'someone-else'})
        assert refused['error']['message'] == 'invalid_params', refused
    finally:
        await client.connection.close()
    assert 'forged' not in _rows(authority)


@pytest.mark.asyncio
async def test_a_sibling_profile_turn_on_the_same_socket_carries_the_same_stamp(tmp_path, monkeypatch):
    """A host-scoped socket (the Desktop's shared primary) routes ``profile=beta`` to beta's
    authority through a sibling connection on the SAME transport, so beta's row is stamped too."""
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    root = tmp_path / '.hermes'
    seen = []
    beta = await _authority(root / 'profiles' / 'beta', monkeypatch, _recorder(seen))
    launch = await _authority(root, monkeypatch, _recorder(seen))
    launch.runner.session_authorities = [launch, beta]
    client = _Client(launch, {'profile_scope': 'host'})
    try:
        await client.call('client.capabilities', client=DESKTOP)
        resumed = await client.call('session.resume', session_id='s', profile='beta')
        assert 'result' in resumed, resumed
        reply = await client.call('prompt.submit', session_id='s', submission_id='routed', text='hi', profile='beta')
        assert reply['result']['status'] == 'queued', reply
        await beta.sessions['s'].task
    finally:
        await client.connection.close()
    assert _rows(beta)['routed']['surface_v1'] == {'client': DESKTOP}
    assert 'routed' not in _rows(launch)
    assert seen == [('hi', DESKTOP, '')]

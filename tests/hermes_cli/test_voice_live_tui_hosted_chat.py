"""Standalone serve records its Desktop calls through the TUI gateway's store."""
import asyncio
from contextlib import asynccontextmanager
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from gateway.session_contract import SessionActor
from gateway.session_controls import AuthorityConnection
from gateway.session_voice_capture import VoiceCaptures
from hermes_cli import web_voice_owner
from hermes_cli.gateway_client import rpc_error
from hermes_cli.web_voice_owner import start_via_voice_owner
from persistence import SessionId
from persistence.conversation.live_voice import VoiceFragment
from persistence.conversation.sessions import NewSession
from persistence.core.errors import AuthorityRefusal
from tests.waiting import async_wait_until
from tools.gpt_live_sideband import ClientDelegation
from tools.voice_live import VOICE_LIVE_TURN_NOTE
from tui_gateway import server


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(server, '_hermes_home', tmp_path)
    monkeypatch.setattr(server, '_db', None)
    monkeypatch.setattr(server, '_db_error', None)
    monkeypatch.setattr(server, '_sessions', {
        'draft-runtime': {
            'session_key': 'fresh-draft', 'source': 'desktop', 'auth_user_id': 'login:fixture',
            'model_override': {'model': 'voice-fixture'},
        },
        'foreign-runtime': {
            'session_key': 'foreign-draft', 'source': 'desktop', 'profile_home': str(tmp_path / 'foreign'),
        },
    })
    store = server._get_db()
    for sid, source in [('desktop-chat', 'desktop'), ('owner-chat', 'local')]:
        store.sessions.create(NewSession(SessionId(sid), source))
    actor = SessionActor('login:fixture', str(tmp_path), frozenset({'session:control'}), 'http')
    calls = {'owner': 0, 'negotiate': 0}
    finished = asyncio.Event()

    async def owner(*_args, **_kwargs):
        calls['owner'] += 1
        return {'session': {'id': 'recorded'}, 'transport': {'type': 'webrtc', 'sdp': 'a'}, 'capture_status': 'recording'}

    def negotiate(sdp, _history, **_kwargs):
        assert sdp == 'v=0\r\n'
        calls['negotiate'] += 1
        return {'session': {'id': 'provider-call'}, 'transport': {'type': 'webrtc', 'sdp': 'a'}}

    @asynccontextmanager
    async def sideband(call_id, *, api_key, base_url):
        assert (call_id, api_key, base_url) == ('provider-call', 'project-fixture', 'https://example.invalid/v1')

        async def events():
            yield VoiceFragment('first', 'user', 'Bring tea.', None, None)
            yield VoiceFragment('second', 'assistant', 'Of course.', None, None)
            yield None

        try:
            yield events()
        finally:
            finished.set()

    async def scoped(_profile, fn):
        return fn()

    monkeypatch.setattr('hermes_cli.web_server_sessions._session_mutation_context', lambda *_: (None, actor))
    monkeypatch.setattr('hermes_cli.web_voice_owner.start_via_voice_owner', owner)
    monkeypatch.setattr('hermes_cli.web_routers.audio._run_config_scoped', scoped)
    monkeypatch.setattr('tools.voice_live.create_webrtc_session', negotiate)
    monkeypatch.setattr('tools.voice_live._resolve_credentials', lambda _: ('project-fixture', 'https://example.invalid/v1'))
    monkeypatch.setattr('gateway.session_voice_capture.voice_capture_socket', sideband)
    try:
        yield SimpleNamespace(calls=calls, store=store, finished=finished, actor=actor)
    finally:
        store.close()


async def _start(session_id):
    from hermes_cli.web_models import VoiceLiveSessionRequest
    from hermes_cli.web_routers.audio import create_voice_live_session
    return await create_voice_live_session(VoiceLiveSessionRequest(sdp='v=0\r\n', session_id=session_id),
                                           Request({'type': 'http'}), None)




@pytest.mark.asyncio
@pytest.mark.parametrize('session_id', ['desktop-chat', 'fresh-draft'])
async def test_tui_hosted_chat_records_in_its_transcript(wired, session_id):
    if session_id == 'fresh-draft':
        assert wired.store.sessions.get(session_id) is None
    answer = await _start(session_id)
    assert wired.calls == {'owner': 0, 'negotiate': 1}
    assert answer['capture_status'] == 'recording'
    assert answer['session']['id'] == 'provider-call'
    await asyncio.wait_for(wired.finished.wait(), timeout=5)
    assert [(row.role, row.content) for row in wired.store.messages.read_transcript(session_id)] == [
        ('user', 'Bring tea.'), ('assistant', 'Of course.'),
    ]
    row = wired.store.sessions.get(session_id)
    assert row.source == 'desktop'
    if session_id == 'fresh-draft':
        assert (row.model, row.user_id) == ('voice-fixture', 'login:fixture')


@pytest.mark.asyncio
async def test_owner_hosted_chat_still_goes_to_the_owner(wired):
    answer = await _start('owner-chat')
    assert answer['capture_status'] == 'recording'
    assert wired.calls == {'owner': 1, 'negotiate': 0}


@pytest.mark.asyncio
@pytest.mark.parametrize('session_id', ['never-stored', 'foreign-draft'])
async def test_unknown_chat_keeps_the_owner_path(wired, session_id):
    await _start(session_id)
    assert wired.calls == {'owner': 1, 'negotiate': 0}


@pytest.mark.asyncio
@pytest.mark.parametrize('refuse_before_negotiation', [True, False])
async def test_owner_not_found_falls_back_only_before_negotiation(wired, monkeypatch, refuse_before_negotiation):
    def authorize(*_args):
        if refuse_before_negotiation or wired.calls['negotiate']:
            raise AuthorityRefusal('not_found')

    owner = SimpleNamespace(profile_id=wired.actor.profile_id, instance_id='fixture', events={}, authorize=authorize)
    connection = AuthorityConnection(owner, object(), {'user_id': 'fixture', 'provider': 'basic'})

    class Socket:
        async def rpc(self, method, *, _timeout, **params):
            wired.calls['owner'] += 1
            reply = await connection.dispatch({'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params})
            raise rpc_error(reply['error'])

    @asynccontextmanager
    async def connect(**_kwargs):
        yield Socket()

    monkeypatch.setattr(web_voice_owner, 'start_via_voice_owner', start_via_voice_owner)
    monkeypatch.setattr(web_voice_owner, 'discover_gateway_endpoint', lambda *_args, **_kwargs: SimpleNamespace(
        endpoint=SimpleNamespace(control_home=wired.actor.profile_id)))
    monkeypatch.setattr(web_voice_owner, 'connect_gateway', connect)
    if refuse_before_negotiation:
        answer = await _start('never-stored')
        assert answer['capture_status'] == 'unavailable'
    else:
        with pytest.raises(HTTPException) as error:
            await _start('never-stored')
        assert (error.value.status_code, error.value.detail) == (404, 'not_found')
    assert wired.calls == {'owner': 1, 'negotiate': 1}
    assert wired.store.sessions.get('never-stored') is None


@pytest.mark.asyncio
async def test_other_owner_refusals_still_fail(wired, monkeypatch):
    async def busy(*_args, **_kwargs):
        raise HTTPException(status_code=503, detail='owner_unreachable')

    monkeypatch.setattr('hermes_cli.web_voice_owner.start_via_voice_owner', busy)
    with pytest.raises(HTTPException) as error:
        await _start('never-stored')
    assert error.value.status_code == 503
    assert wired.calls['negotiate'] == 0


@pytest.mark.asyncio
async def test_local_socket_failure_returns_the_same_call_unsaved(wired, monkeypatch):
    @asynccontextmanager
    async def unavailable(*_args, **_kwargs):
        raise OSError('sideband disconnected')
        yield

    monkeypatch.setattr('gateway.session_voice_capture.voice_capture_socket', unavailable)
    answer = await _start('desktop-chat')
    assert answer == {'ok': True, 'session': {'id': 'provider-call'}, 'transport': {'type': 'webrtc', 'sdp': 'a'},
                      'capture_status': 'unavailable'}
    assert wired.calls == {'owner': 0, 'negotiate': 1}
    assert not wired.store.messages.read_transcript('desktop-chat')


@pytest.mark.asyncio
async def test_local_capture_finishes_after_concurrent_turn_releases_its_lease(wired):
    session_id = SessionId('desktop-chat')
    lease = {'session_id': session_id, 'holder': 'test-turn'}
    assert wired.store.coordination.try_acquire_turn_lease(**lease)
    try:
        answer = await _start(session_id)
        assert answer['capture_status'] == 'recording'
        await asyncio.wait_for(wired.finished.wait(), timeout=5)
        assert not wired.store.messages.read_transcript(session_id)
        assert len(wired.store.messages.pending_voice_fragments('provider-call')) == 2
    finally:
        wired.store.coordination.release_turn_lease(**lease)
    rows = await async_wait_until(lambda: wired.store.messages.read_transcript(session_id),
                                  what='voice transcript publication after the text turn')
    assert [row.content for row in rows] == ['Bring tea.', 'Of course.']

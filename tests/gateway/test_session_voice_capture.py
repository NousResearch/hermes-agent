"""The capture worker writes the real store without calling prompt submission."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from starlette.requests import Request

from gateway.session_voice_capture import record_voice_events, recover_voice_calls
from persistence import ProfileScope, SessionId
from persistence.authority.admissions import VOICE_CLAIM_V1
from persistence.conversation.live_voice import VoiceFragment
from persistence.conversation.sessions import NewSession
from persistence.sqlite import SQLiteBackend
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from tools.gpt_live_sideband import ClientDelegation


@pytest.mark.asyncio
@pytest.mark.parametrize('closed', [True, False])
async def test_voice_capture_projects_two_speakers_into_real_history(tmp_path: Path, closed: bool):
    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    try:
        store = backend.open(scope)
        session = store.sessions.create(NewSession(SessionId('synthetic'), 'desktop'))
        call = store.messages.start_voice_capture(session, 'test-call')

        async def events():
            user = VoiceFragment('1', 'user', 'Please bring ', 300, 1000)
            yield user
            yield user  # back-to-back provider replay
            yield VoiceFragment('2', 'user', 'tea.', 100, 500)  # starts earlier, arrives later
            yield VoiceFragment('3', 'assistant', 'Of ', 1200, 1500)
            assert updates == [('Please bring tea.',)]
            yield VoiceFragment('4', 'assistant', 'course.', None, None)
            yield VoiceFragment('5', 'user', 'Thanks.', 1700, 1800)
            assert updates == [('Please bring tea.',), ('Please bring tea.', 'Of course.')]
            assert [f.event_id for f in store.messages.pending_voice_fragments(call)] == ['5']
            if closed:
                yield None

        updates = []

        def published():
            # The event is an invalidation, not a substitute for the store write.
            updates.append(tuple(row.content for row in store.messages.read_conversation(session)))

        assert await record_voice_events(store.messages, call, events(), published=published) is closed
        assert updates[-1] == ('Please bring tea.', 'Of course.', 'Thanks.')
        backend.reopen()
        store = backend.open(scope)
        rows = store.messages.read_conversation(session)
        assert [(row.role, row.content) for row in rows] == [
            ('user', 'Please bring tea.'), ('assistant', 'Of course.'), ('user', 'Thanks.'),
        ]
        assert [row.display_metadata['voice_live']['partial'] for row in rows] == [False, False, not closed]
        assert store.messages.pending_voice_fragments(call) == ()
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_projection_retries_when_claim_consumes_a_pending_prefix(tmp_path):
    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    try:
        store = backend.open(scope)
        epoch = begin_runtime_epoch(store.holder(), instance_id='voice-owner')
        chat = store.sessions.create(NewSession(SessionId('voice'), 'desktop'))
        call = store.messages.start_voice_capture(chat, 'call')

        async def events():
            yield VoiceFragment('request', 'user', 'Bring tea.', 0, 300)
            yield ClientDelegation(call_id=call, delegation_id='d', event_id='marker', offset_ms=300)
            yield VoiceFragment('late', 'user', ' No sugar.', 300, 500)
            yield VoiceFragment('answer', 'assistant', 'Of course.', 500, 800)
            assert [row.content for row in store.messages.read_transcript(chat)] == ['Bring tea.', ' No sugar.']
            yield None

        def write_with_race(write, *args, **kwargs):
            if write == store.messages.publish_voice_turns and args[1][0].event_ids == ('request', 'late'):
                # The claim commits after projection has read the combined run.
                payload = {'text': 'Bring tea.', VOICE_CLAIM_V1: {
                    'call_id': call, 'delegation_id': 'd', 'event_ids': ['request'],
                }}
                admitted, _ = store.admissions.admit(epoch=epoch, principal_id='owner', session_id=chat,
                                                      request_id='voice-request', payload=payload)
                claimed = store.admissions.claim(epoch=epoch, session_id=chat, head=admitted.admission_id)
                assert claimed.voice_input.text == 'Bring tea.'
            return write(*args, **kwargs)

        async def write(fn, *args, **kwargs):
            return await asyncio.to_thread(write_with_race, fn, *args, **kwargs)

        assert await record_voice_events(store.messages, call, events(), write=write)
        assert [(row.role, row.content) for row in store.messages.read_transcript(chat)] == [
            ('user', 'Bring tea.'), ('user', ' No sugar.'), ('assistant', 'Of course.'),
        ]
        assert not store.messages.pending_voice_fragments(call)
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_sideband_failure_publishes_incomplete_to_the_original_chat(tmp_path):
    from gateway.session_events import SessionEvents
    from gateway.session_voice_capture import VoiceCaptures

    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    store = backend.open(scope)
    session = store.sessions.create(NewSession(SessionId('recording'), 'desktop'))
    events = SessionEvents()
    frames = []
    events.observers.add(frames.append)
    authority = SimpleNamespace(sessions={session: SimpleNamespace(event_stream=events)})
    captures = VoiceCaptures(store.messages, authority=authority)

    @asynccontextmanager
    async def socket():
        async def fragments():
            yield VoiceFragment('e1', 'user', 'Last words', 0, 500)
            raise OSError('transient sideband disconnect')
        yield fragments()

    try:
        assert await captures.start(session, 'call', socket())
        await captures.tasks['call']
        assert [(frame['params']['type'], frame['params']['session_id'], frame['params']['payload']) for frame in frames] == [
            ('voice.live.capture', session, {'call_id': 'call', 'state': 'recording'}),
            ('voice.live.capture', session, {'call_id': 'call', 'state': 'updated'}),
            ('voice.live.capture', session, {'call_id': 'call', 'state': 'incomplete'}),
        ]
        assert store.messages.read_conversation(session)[0].display_metadata['voice_live']['partial']
    finally:
        await captures.close()
        backend.close()


@pytest.mark.asyncio
async def test_provider_close_after_rewind_reports_incomplete(tmp_path):
    from gateway.session_events import SessionEvents
    from gateway.session_voice_capture import VoiceCaptures
    from persistence.conversation.messages import NewMessage

    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    store = backend.open(scope)
    session = store.sessions.create(NewSession(SessionId('rewound'), 'desktop'))
    store.messages.append(session, NewMessage('user', 'Undo this.'))
    stream = SessionEvents()
    frames = []
    stream.observers.add(frames.append)
    captures = VoiceCaptures(store.messages, authority=SimpleNamespace(
        sessions={session: SimpleNamespace(event_stream=stream)}))

    @asynccontextmanager
    async def socket():
        async def fragments():
            yield VoiceFragment('pending', 'user', 'Do not restore me.', 0, 200)
            await asyncio.to_thread(store.messages.rewind_user_turn, session, 0)
            yield None
        yield fragments()

    try:
        assert await captures.start(session, 'call', socket())
        await captures.tasks['call']
        assert not store.messages.read_transcript(session)
        assert frames[-1]['params']['payload'] == {'call_id': 'call', 'state': 'incomplete'}
    finally:
        await captures.close()
        backend.close()


@pytest.mark.asyncio
async def test_restart_recovers_the_unfinished_tail_without_minting_a_turn(tmp_path):
    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    try:
        store = backend.open(scope)
        session = store.sessions.create(NewSession(SessionId('recover'), 'desktop'))
        call = store.messages.start_voice_capture(session, 'unfinished-call')
        store.messages.ingest_voice_fragment(call, VoiceFragment('e1', 'user', 'The last words', None, None))
        backend.reopen()  # no finish or close signal was committed
        store = backend.open(scope)
        await recover_voice_calls(store.messages)
        await recover_voice_calls(store.messages)
        rows = store.messages.read_conversation(session)
        assert [(r.content, r.display_metadata['voice_live']['partial']) for r in rows] == [('The last words', True)]
        assert not store.messages.pending_voice_calls()
        assert not store.admissions.list(session_id=session, pending_only=False)
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_standalone_dashboard_uses_the_selected_profiles_owner(tmp_path, monkeypatch):
    from gateway.session_contract import SessionActor
    from hermes_cli.web_models import VoiceLiveSessionRequest
    from hermes_cli.web_routers.audio import create_voice_live_session

    home = tmp_path / 'secondary'
    actor = SessionActor('login:fixture', str(home), frozenset({'session:control'}), 'http')
    endpoint = SimpleNamespace(control_home=str(home))
    reply = {'session': {'id': 'test-call'}, 'transport': {'type': 'webrtc', 'sdp': 'answer'},
             'capture_status': 'recording'}

    class Socket:
        async def rpc(self, method, **params):
            assert method == 'voice.live.start'
            assert params == {'session_id': 'book-debrief', 'sdp': 'v=0\r\n', 'history': None, '_timeout': 45}
            return reply

    @asynccontextmanager
    async def gateway(*, endpoint: SimpleNamespace, subject: str):
        assert endpoint.control_home == str(home)
        assert subject == actor.subject
        yield Socket()

    def discover(target, *, timeout):
        assert target == home
        return SimpleNamespace(endpoint=endpoint)

    def must_not_negotiate(*_args, **_kwargs):
        pytest.fail('The dashboard must not open a second provider call')

    monkeypatch.setattr('hermes_cli.web_server_sessions._session_mutation_context', lambda *_: (None, actor))
    monkeypatch.setattr('hermes_cli.web_voice_owner.discover_gateway_endpoint', discover)
    monkeypatch.setattr('hermes_cli.web_voice_owner.connect_gateway', gateway)
    monkeypatch.setattr('tools.voice_live.create_webrtc_session', must_not_negotiate)
    answer = await create_voice_live_session(VoiceLiveSessionRequest(sdp='v=0\r\n', session_id='book-debrief'),
                                             Request({'type': 'http'}), 'secondary')
    assert answer == {'ok': True, **reply}


@pytest.mark.asyncio
async def test_ambiguous_owner_failure_does_not_open_a_second_provider_call(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from gateway.session_contract import SessionActor
    from hermes_cli.web_models import VoiceLiveSessionRequest
    from hermes_cli.web_routers.audio import create_voice_live_session

    actor = SessionActor('login:fixture', str(tmp_path), frozenset({'session:control'}), 'http')
    endpoint = SimpleNamespace(control_home=str(tmp_path))

    @asynccontextmanager
    async def lost_reply(**_kwargs):
        class Socket:
            async def rpc(self, *_args, **_kwargs):
                raise TimeoutError('The owner may have already opened the call')

        yield Socket()

    def must_not_negotiate(*_args, **_kwargs):
        pytest.fail('Ambiguous owner acknowledgement must not fall through to another paid call')

    monkeypatch.setattr('hermes_cli.web_server_sessions._session_mutation_context', lambda *_: (None, actor))
    monkeypatch.setattr('hermes_cli.web_voice_owner.discover_gateway_endpoint', lambda *_a, **_kw: SimpleNamespace(endpoint=endpoint))
    monkeypatch.setattr('hermes_cli.web_voice_owner.connect_gateway', lost_reply)
    monkeypatch.setattr('tools.voice_live.create_webrtc_session', must_not_negotiate)
    with pytest.raises(HTTPException) as error:
        await create_voice_live_session(VoiceLiveSessionRequest(sdp='v=0\r\n', session_id='book-debrief'),
                                       Request({'type': 'http'}), 'secondary')
    assert (error.value.status_code, error.value.detail) == (503, 'owner_unreachable')


@pytest.mark.asyncio
async def test_wait_for_passive_marker_is_scoped_and_wakes_after_commit(tmp_path):
    from gateway.session_voice_capture import VoiceCaptures
    from tools.gpt_live_sideband import ClientDelegation

    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    store = backend.open(scope)
    chat = store.sessions.create(NewSession(SessionId('voice'), 'desktop'))
    captures = VoiceCaptures(store.messages)
    release_marker, close_stream = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def socket():
        async def events():
            yield VoiceFragment('ev', 'user', 'Hello.', 0, 300)
            await release_marker.wait()
            yield ClientDelegation(call_id='call', delegation_id='d', event_id='marker', offset_ms=300)
            await close_stream.wait()
            yield None

        yield events()

    try:
        assert await captures.start(chat, 'call', socket())
        waiter = asyncio.create_task(captures.wait_for_marker(chat, 'call', 'd'))
        await asyncio.sleep(0)
        assert not waiter.done()
        assert not await captures.wait_for_marker('another-chat', 'call', 'd')
        release_marker.set()
        assert await asyncio.wait_for(waiter, timeout=1)
        assert not store.messages.record_voice_marker('call', 'd', 'marker', 300)
        assert await captures.wait_for_marker(chat, 'call', 'd')
        assert not await captures.wait_for_marker(chat, 'call', 'missing', timeout=0)
        waiter = asyncio.create_task(captures.wait_for_marker(chat, 'call', 'closing'))
        close_stream.set()
        assert not await asyncio.wait_for(waiter, timeout=1)
    finally:
        await captures.close()
        backend.close()


@pytest.mark.asyncio
async def test_retirement_retains_owner_while_a_cancelled_capture_write_finishes(tmp_path, monkeypatch):
    from gateway.run_runtime import _retire_profile_authority
    from gateway.session_authority import SessionAuthority

    monkeypatch.setattr('gateway.run_runtime.stop_authority_turns', lambda *_a, **_kw: None)
    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    store = backend.open(scope)
    session = store.sessions.create(NewSession(SessionId('retiring'), 'desktop'))
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False), _running_agents={})
    authority = SessionAuthority(runner, profile_id='p', instance_id='test', epoch=1, persistence=store)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    ingest = store.messages.ingest_voice_fragment

    def wait_then_write(call, fragment):
        loop.call_soon_threadsafe(started.set)
        if not release.wait(10):
            raise RuntimeError('fixture never released the SQLite writer')
        return ingest(call, fragment)

    monkeypatch.setattr(type(store.messages), 'ingest_voice_fragment', lambda _self, *args: wait_then_write(*args))

    @asynccontextmanager
    async def socket():
        async def events():
            yield VoiceFragment('e1', 'user', 'Last words', 0, 400)
            await asyncio.Future()
        yield events()

    try:
        assert await authority.voice_captures.start(session, 'call', socket())
        await asyncio.wait_for(started.wait(), timeout=5)
        assert not await _retire_profile_authority(authority, timeout=0)
        await asyncio.sleep(0)  # The capture waiter may cancel; SQLite still owns the write.
        assert not await _retire_profile_authority(authority, timeout=0.05)
        assert not authority.retired
        release.set()
        assert await _retire_profile_authority(authority, timeout=5)
        pending = store.messages.pending_voice_fragments('call')
        rows = store.messages.read_conversation(session)
        assert [f.text for f in pending] + [r.content for r in rows] == ['Last words']
    finally:
        release.set()
        await authority.voice_captures.close()
        backend.close()


@pytest.mark.asyncio
async def test_idle_profile_can_retry_retirement_at_zero_budget(tmp_path, monkeypatch):
    from gateway.run_runtime import _retire_profile_authority
    from gateway.session_authority import SessionAuthority

    # No agent is running. Keep the unrelated turn-stop bootstrap out of this
    # fixture, but exercise the real retirement and mutation inventory checks.
    monkeypatch.setattr('gateway.run_runtime.stop_authority_turns', lambda *_a, **_kw: None)
    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    try:
        store = backend.open(scope)
        runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False), _running_agents={})
        authority = SessionAuthority(runner, profile_id='p', instance_id='test', epoch=1, persistence=store)
        assert await _retire_profile_authority(authority, timeout=0)
        assert authority.retired
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_discovery_refusal_cannot_fall_back_to_an_unsaved_call(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from gateway.session_contract import SessionActor
    from hermes_cli.gateway_runtime import GatewayDiscovery
    from hermes_cli.web_voice_owner import start_via_voice_owner

    actor = SessionActor('login:fixture', str(tmp_path), frozenset({'session:control'}), 'http')
    for reason, status in [('authorization', 403), ('profile_mismatch', 403), ('control_timeout', 503)]:
        monkeypatch.setattr('hermes_cli.web_voice_owner.discover_gateway_endpoint',
                            lambda *_a, **_kw: GatewayDiscovery('inaccessible', reason_code=reason))
        with pytest.raises(HTTPException) as error:
            await start_via_voice_owner(tmp_path, actor, 'book-debrief', 'sdp', None)
        assert error.value.status_code == status
    monkeypatch.setattr('hermes_cli.web_voice_owner.discover_gateway_endpoint',
                        lambda *_a, **_kw: GatewayDiscovery('absent'))
    assert await start_via_voice_owner(tmp_path, actor, 'book-debrief', 'sdp', None) is None


@pytest.mark.asyncio
async def test_capable_owner_refuses_to_start_an_unsaved_call(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from gateway.session_contract import SessionActor
    from gateway.session_voice_capture import VoiceCaptures
    from hermes_cli.web_models import VoiceLiveSessionRequest
    from hermes_cli.web_routers.audio import create_voice_live_session

    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    store = backend.open(scope)
    session = store.sessions.create(NewSession(SessionId('synthetic'), 'desktop'))
    captures = VoiceCaptures(store.messages)
    owner = SimpleNamespace(profile_id='p', authorize=Mock(), voice_captures=captures, sessions={})
    actor = SessionActor('test-user', 'p', frozenset({'session:control'}), 'http')

    @asynccontextmanager
    async def unavailable(*_args, **_kwargs):
        raise RuntimeError('provider unavailable')
        yield  # pragma: no cover -- the handshake itself always fails

    monkeypatch.setattr('tools.voice_live.create_webrtc_session', Mock(return_value={
        'session': {'id': 'test-call'}, 'transport': {'type': 'webrtc', 'sdp': 'answer'},
    }))
    monkeypatch.setattr('tools.voice_live._resolve_credentials', lambda _: ('project-test', 'https://example.invalid/v1'))
    monkeypatch.setattr('hermes_cli.web_server_sessions._session_mutation_context', lambda *_: (owner, actor))
    monkeypatch.setattr('gateway.session_voice_capture.voice_capture_socket', unavailable)
    try:
        with pytest.raises(HTTPException) as error:
            await create_voice_live_session(VoiceLiveSessionRequest(sdp='v=0\r\n', session_id=session),
                                            Request({'type': 'http'}), 'secondary')
        assert (error.value.status_code, error.value.detail) == (503, 'voice_capture_unavailable')
        assert not store.messages.pending_voice_calls()
    finally:
        await captures.close()
        backend.close()


@pytest.mark.asyncio
async def test_audio_session_starts_capture_before_returning_sdp(tmp_path, monkeypatch):
    from gateway.session_contract import SessionActor, SessionRef
    from gateway.session_voice_capture import VoiceCaptures
    from hermes_cli.web_models import VoiceLiveSessionRequest
    from hermes_cli.web_routers.audio import create_voice_live_session

    scope = ProfileScope(profile_dir=tmp_path, name='default')
    backend = SQLiteBackend({scope.hermes_home_key: tmp_path / 'state.db'})
    store = backend.open(scope)
    session = store.sessions.create(NewSession(SessionId('synthetic'), 'desktop'))
    captures = VoiceCaptures(store.messages)
    owner = SimpleNamespace(profile_id='p', authorize=Mock(), voice_captures=captures, sessions={})
    actor = SessionActor('test-user', 'p', frozenset({'session:control'}), 'http')

    @asynccontextmanager
    async def socket(call_id, *, api_key, base_url):
        assert (call_id, api_key, base_url) == ('test-call', 'project-test', 'https://example.invalid/v1')

        async def events():
            yield VoiceFragment('1', 'user', 'Please bring tea.', 0, 1000)
            yield VoiceFragment('2', 'assistant', 'Of course.', 1200, 1600)
            yield None

        yield events()

    @asynccontextmanager
    async def capture():
        try:
            yield
        finally:
            await captures.close()
            backend.close()

    create = Mock(return_value={'session': {'id': 'test-call', 'diagnostic': 'private'},
                                'transport': {'type': 'webrtc', 'sdp': 'answer', 'diagnostic': 'private'},
                                'diagnostic': 'private'})
    monkeypatch.setattr('tools.voice_live.create_webrtc_session', create)
    monkeypatch.setattr('tools.voice_live._resolve_credentials', lambda _live: ('project-test', 'https://example.invalid/v1'))
    monkeypatch.setattr('hermes_cli.web_server_sessions._session_mutation_context', lambda *_: (owner, actor))
    monkeypatch.setattr('gateway.session_voice_capture.voice_capture_socket', socket)

    async with capture():
        answer = await create_voice_live_session(VoiceLiveSessionRequest(sdp='v=0\r\n', session_id=session),
                                                 Request({'type': 'http'}), 'secondary')
        assert answer == {'ok': True, 'session': {'id': 'test-call'},
                          'transport': {'type': 'webrtc', 'sdp': 'answer'}, 'capture_status': 'recording'}
        create.assert_called_once_with('v=0\r\n', None, credentials=('project-test', 'https://example.invalid/v1'))
        owner.authorize.assert_called_with(actor, SessionRef('p', session), 'session:control')
        await captures.tasks['test-call']
        assert [(r.role, r.content) for r in store.messages.read_conversation(session)] == [
            ('user', 'Please bring tea.'), ('assistant', 'Of course.'),
        ]

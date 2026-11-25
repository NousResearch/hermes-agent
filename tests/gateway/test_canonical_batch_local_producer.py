"""Native local GUI admission reaches the real batch producer without a provider or credentials."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
from gateway.run_turn_runner import TurnRunner
from gateway import session_server_requests as requests
from tests.gateway.fixtures.canonical_batch_owner import Peer, QUESTIONS, opening, rpc


@pytest.mark.asyncio
async def test_native_local_gui_admission_invokes_batch_producer(tmp_path, monkeypatch):
    from gateway import run
    monkeypatch.setattr(run, '_hermes_home', tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('platform_toolsets: {cli: []}\nmodel: {provider: custom, default: inert}\n')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    seen = []
    async def answer(event):
        assert event.source.platform == Platform.LOCAL
        generation = owner.executing[ref.session_id]
        turn = SimpleNamespace(_approval_owner=(owner, ref.session_id, generation), _ctx=SimpleNamespace(
            source=event.source, inbound_message_id=event.message_id,
            agent_holder=[SimpleNamespace(_interrupt_requested=False)]))
        seen.append(await asyncio.to_thread(TurnRunner._clarify_callback_sync, turn, QUESTIONS))
        return 'ok'
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _handle_message=answer, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None, _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='fixture')
    peer = Peer()
    connection = AuthorityConnection(owner, peer, {'user_id': 'owner'})
    ref = create_local_session(owner, connection.actor, {'request_id': 'gui', 'source': 'gui',
                                                       'cwd': str(tmp_path), 'toolsets': []})
    try:
        await rpc(connection, 'client.capabilities', server_requests=True)
        await rpc(connection, 'client.capabilities', canonical_clarify_projection_v1=True)
        resume = await rpc(connection, 'session.resume', session_id=ref.session_id)
        assert 'result' in resume, resume
        submitted = await rpc(connection, 'prompt.submit', session_id=ref.session_id, submission_id='local-gui',
                              text='inert batch producer', clarify_batch=True, clarify_projection='canonical_batch_v1')
        assert submitted['result']['status'] == 'queued', submitted
        batch = await opening(peer)
        assert batch['identity']['profile_id'] == str(tmp_path)
        assert batch['questions'] == QUESTIONS
        settled = await rpc(connection, 'clarify.batch.respond', session_id=ref.session_id,
                            request_id=batch['request_id'], identity=batch['identity'],
                            result={'answers': {'q0': '', 'q1': None}, 'outcome': 'submitted'})
        assert settled['result'] == {'status': 'ok'}, settled
        await owner.sessions[ref.session_id].task
        assert seen == [{'answers': {'q0': '', 'q1': None}, 'outcome': 'submitted'}]
    finally:
        requests.withdraw(owner.sessions[ref.session_id].event_stream, ref.session_id, 'clarify')
        if owner.sessions[ref.session_id].task:
            await owner.sessions[ref.session_id].task
        await connection.close()

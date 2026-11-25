"""Batch outcomes preserve native semantics through selected canonical production."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session_controls import AuthorityConnection
from gateway.run_turn_runner import TurnRunner
from gateway import session_server_requests as requests
from tests.gateway.fixtures.canonical_batch_owner import Peer, QUESTIONS, make_owner, opening, rpc


@pytest.mark.asyncio
@pytest.mark.parametrize('ending', ['cancel', 'deadline', 'interrupt', 'legacy_error', 'legacy_last_lock', 'canonical_last_lock'])
async def test_selected_batch_native_outcomes_and_legacy_settlers(tmp_path, monkeypatch, ending):
    from tools import clarify_gateway
    monkeypatch.setattr(clarify_gateway, 'get_clarify_timeout', lambda: .5 if ending == 'deadline' else 5)
    seen = []
    agent = SimpleNamespace(_interrupt_requested=False)
    async def answer(event):
        turn = SimpleNamespace(_approval_owner=(owner, 's', owner.executing['s']), _ctx=SimpleNamespace(
            inbound_message_id=event.message_id, source=SimpleNamespace(platform=Platform.LOCAL), agent_holder=[agent]))
        seen.append(await asyncio.to_thread(TurnRunner._clarify_callback_sync, turn, QUESTIONS))
        return 'ok'
    owner = await make_owner(tmp_path, monkeypatch, answer)
    canonical = AuthorityConnection(owner, Peer(), {'user_id': 'owner'})
    legacy = AuthorityConnection(owner, Peer(), {'user_id': 'owner'})
    try:
        await rpc(canonical, 'client.capabilities')
        await rpc(canonical, 'client.capabilities', canonical_clarify_projection_v1=True)
        await rpc(canonical, 'session.resume', session_id='s')
        await rpc(legacy, 'session.resume', session_id='s')
        admitted = await rpc(canonical, 'prompt.submit', session_id='s', submission_id='modes', text='ask',
                             clarify_batch=True, clarify_projection='canonical_batch_v1')
        assert 'result' in admitted, admitted
        batch = await opening(canonical.transport)
        params = {'session_id': 's', 'request_id': batch['request_id'], 'identity': batch['identity']}
        first = await rpc(canonical, 'clarify.batch.lock', **params, question_id='q0', answer=None)
        assert first['result'] == {'status': 'ok', 'remaining': ['q1']}
        async def cancel():
            reply = await rpc(canonical, 'clarify.batch.respond', **params, result={'answers': {}, 'outcome': 'cancelled'})
            assert reply['result'] == {'status': 'ok'}
        async def interrupt():
            agent._interrupt_requested = True
        async def legacy_error():
            await legacy.dispatch({'id': batch['request_id'], 'error': {'code': -32601, 'message': 'unsupported'}})
        async def legacy_last_lock():
            reply = await rpc(legacy, 'clarify.lock', request_id=batch['request_id'], question_id='q1', answer='')
            assert reply['result'] == {'status': 'ok', 'remaining': []}
        async def canonical_last_lock():
            reply = await rpc(canonical, 'clarify.batch.lock', **params, question_id='q1', answer='')
            assert reply['result'] == {'status': 'ok', 'remaining': []}
        async def deadline():
            pass  # The producer's real timeout settles this case.
        await {'cancel': cancel, 'interrupt': interrupt, 'legacy_error': legacy_error,
               'legacy_last_lock': legacy_last_lock, 'canonical_last_lock': canonical_last_lock,
               'deadline': deadline}[ending]()
        await asyncio.wait_for(asyncio.shield(owner.sessions['s'].task), timeout=10)
        expected = {'answers': {'q0': None}, 'outcome': 'timed_out' if ending == 'deadline' else 'cancelled'}
        if ending.endswith('last_lock'):
            expected = {'answers': {'q0': None, 'q1': ''}, 'outcome': 'submitted'}
        assert seen == [expected]
        duplicate = await rpc(canonical, 'clarify.batch.respond', **params, result={'answers': {'q0': 'late'}, 'outcome': 'submitted'})
        assert duplicate['result'] == {'status': 'expired'}
        assert requests.canonical_batches(owner.sessions['s'].event_stream, 's') == []
    finally:
        requests.withdraw(owner.sessions['s'].event_stream, 's', 'clarify')
        if owner.sessions['s'].task:
            await owner.sessions['s'].task
        await canonical.close()
        await legacy.close()

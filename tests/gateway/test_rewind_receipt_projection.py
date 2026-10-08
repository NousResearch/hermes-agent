"""Receipt retries preserve transcript boundaries, successor ownership and media identity."""



from pathlib import Path



from types import SimpleNamespace



import pytest



from gateway.session_authority import LiveSession



from gateway.session_contract import Principal, SessionRef, Submission



from gateway.session_results import admission_result, finish_result



from hermes_state_runtime import (RuntimeStoreError, admit_session_input, begin_runtime_epoch,
    claim_session_input, get_session_admission, recover_session_inputs)



@pytest.mark.asyncio
async def test_rewind_retry_does_not_evict_successor_agent(owner):
    from gateway.session_mutations import mutate_session
    owner.db.create_session('s', source='test')
    owner.db.append_message('s', 'user', 'keep')
    owner.db.append_message('s', 'assistant', 'discard')
    owner.db.append_message('s', 'user', 'rewind this')
    owner.sessions['s'] = LiveSession(None, 'route')
    agent = SimpleNamespace(interrupts=0)
    cached = {'route': agent}
    agent.interrupt = lambda: setattr(agent, 'interrupts', agent.interrupts + 1)
    owner.runner._evict_cached_agent = lambda route: cached.pop(route, None)
    owner.runner._cached_agent_for = cached.get
    actor = Principal('human', owner.profile_id, frozenset({'session:submit', 'session:control'}), 't')
    ref = SessionRef(owner.profile_id, 's')
    params = dict(session_id='s', request_id='rewind', expected_revision=0, expected_generation=0,
                  operation='rewind', payload={'target_message_id': owner.db.get_messages('s')[2]['id']})
    result = await mutate_session(owner, actor, ref, params)
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='human', session_id='s',
                        request_id='next', payload={'text': 'next'})
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    cached['route'] = agent
    assert await mutate_session(owner, actor, ref, params) == result
    await owner.interrupt(actor, ref, started['generation'])
    assert cached.get('route') is agent and agent.interrupts == 1

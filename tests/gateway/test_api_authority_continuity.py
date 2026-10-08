"""Canonical API turns retain logical identity, reachable controls and owned image bytes."""



import asyncio



import base64



from pathlib import Path



import pytest



from gateway.session_api import restore_api_session



from gateway.session_api_turn import admit_api_turn



from hermes_state_runtime import RuntimeStoreError, claim_session_input, settle_session_input



PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')



def test_declared_api_root_restores_compression_tip_and_retries_same_admission(api, owner):
    kwargs = dict(session_id='s', request_id='one', user_message='hello', conversation_history=[],
                  gateway_session_key='conversation', bind_declared_conversation=True)
    _, ref, first = admit_api_turn(api, **kwargs)
    _, _, second = admit_api_turn(api, **{**kwargs, 'request_id': 'two', 'user_message': 'next'})
    owner.db.publish_compression_child(parent_session_id='s', child_session_id='child', source='api_server',
        messages=[{'role': 'assistant', 'content': 'summary'}], require_compression_lease=False)
    route = owner.sessions['s'].route
    owner.runner.session_store.advance_compression_session(route, 's', 'child')
    assert restore_api_session(owner, 's') == ref
    assert owner.runner.session_store.peek_session_id(route) == 'child'
    assert admit_api_turn(api, **kwargs)[2]['admission_id'] == first['admission_id']
    assert admit_api_turn(api, **{**kwargs, 'request_id': 'two', 'user_message': 'next'})[2]['admission_id'] == second['admission_id']

    # A restart may restore a persisted routing entry from before the final compression
    # publication. Any genuine ancestor may catch up, not only the original root.
    owner.db.publish_compression_child(parent_session_id='child', child_session_id='tip', source='api_server',
        messages=[{'role': 'assistant', 'content': 'new summary'}], require_compression_lease=False)
    assert restore_api_session(owner, 's') == ref
    assert owner.runner.session_store.peek_session_id(route) == 'tip'



@pytest.mark.asyncio
@pytest.mark.parametrize('public_run', ['chatcmpl-public', 'session-public'])
async def test_stream_approval_public_run_controls_exact_canonical_admission(api, owner, monkeypatch, public_run):
    from gateway.platforms.api_server_authority_runs import respond_run, run_admission
    received = asyncio.Event()
    answered = asyncio.Event()
    events = []
    scope = 'a' * 64
    api._run_owners[public_run] = scope
    def notify(data):
        events.append(data)
        received.set()
    async def execute(authority, ref, row):
        authority.register_approval(ref.session_id, row['generation'], 'route',
                                    {'request_id': 'approval', 'command': 'controlled'})
        authority.sessions[ref.session_id].controls.remote_responders['approval'] = (
            lambda kind, prompt, response: answered.set())
        await answered.wait()
        authority.pending_results[row['admission_id']] = {
            'result': {'final_response': 'done', 'messages': [], 'completed': True}, 'usage': {}}
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    task = asyncio.create_task(api._run_agent(user_message='hello', conversation_history=[],
        session_id='s', request_id='durable-id', approval_session_key=public_run,
        approval_notify_callback=notify))
    try:
        await asyncio.wait_for(received.wait(), 10)
        authority, row = run_admission(api, public_run)
        assert row['request_id'] == 'durable-id'
        prompt = events[0]
        assert prompt['request_id'] == 'approval'
        assert prompt['execution_generation'] == row['generation']
        result = await respond_run(api, public_run, {'request_id': prompt['request_id'],
            'execution_generation': prompt['execution_generation'], 'choice': 'once'}, kind='approval')
        assert result['status'] == 'resolved'
        assert (await asyncio.wait_for(task, 10))[0]['completed'] is True
    finally:
        answered.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)



@pytest.mark.asyncio
async def test_observer_reads_terminal_receipt_after_admission_snapshot_settles(api, owner):
    from gateway.session_api_turn import observe_api_turn
    admitted = admit_api_turn(api, session_id='s', request_id='fast', user_message='hello', conversation_history=[])
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'], generation=row['generation'],
        outcome='completed', result={'result': {'final_response': 'done', 'completed': True}, 'usage': {}})
    result, _ = await asyncio.wait_for(observe_api_turn(admitted), 10)
    assert result['final_response'] == 'done'

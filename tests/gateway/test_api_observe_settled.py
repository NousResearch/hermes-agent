"""An API observer never waits on an admission the FIFO already settled or cancelled."""
import asyncio

import pytest

from gateway.session_api_turn import admit_api_turn, observe_api_turn


@pytest.mark.asyncio
async def test_observer_arriving_after_a_fast_settle_returns_the_committed_result(api, owner):
    """A drain already running ahead claims and settles a successor without yielding (its turn
    awaits nothing that suspends), so the successor's observer starts with a stale ``queued``
    snapshot and nobody left to resolve a new waiter."""
    admitted = admit_api_turn(api, session_id='fast', user_message='hello', conversation_history=[])
    _, ref, row = admitted
    assert row['status'] == 'queued'

    async def handle(event):
        from gateway.session_results import execution_result
        execution_result.get()['result'] = {'final_response': 'settled'}
        return 'settled'
    owner.runner._handle_message = handle
    await owner._drain(ref)
    deltas = []
    result, _ = await asyncio.wait_for(observe_api_turn(admitted, stream_delta_callback=deltas.append), 5)
    assert result['final_response'] == 'settled' and deltas == ['settled']
    assert row['admission_id'] not in owner.waiters


@pytest.mark.asyncio
async def test_observer_arriving_after_a_cancel_reports_the_cancellation(api, owner):
    from gateway.session_contract import Principal
    admitted = admit_api_turn(api, session_id='gone', user_message='hello', conversation_history=[])
    authority, ref, row = admitted
    actor = Principal('api', authority.profile_id, frozenset({'session:submit', 'session:control'}), 'api-run:x')
    await authority.cancel_queued(actor, ref, row['admission_id'])
    result, usage = await asyncio.wait_for(observe_api_turn(admitted), 5)
    assert result == {'final_response': '', 'interrupted': True, 'completed': False} and usage == {}
    assert row['admission_id'] not in owner.waiters

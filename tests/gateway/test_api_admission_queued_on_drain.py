"""An API caller whose input is still queued when the gateway retires is told it is saved (#134889).

Retirement releases every observer with runtime_draining, which the HTTP layer answered with a
retryable 503 + Retry-After: the answer a refused request gets. The row stayed queued and ran
exactly once under the next owner, so a client that retried on the 503 submitted the work twice.
"""
import asyncio
import json

import pytest

from gateway import run_runtime
from gateway.platforms.api_server import _openai_error
from gateway.platforms.api_server_authority_runs import store_error_response
from gateway.session_api_turn import AdmissionQueued, observe_api_turn
from gateway.session_contract import Submission
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.test_profile_retire_stops_turns import ACTOR, REF, _Agent, _authority
from tests.waiting import async_wait_until


async def _observe_queued_then(tmp_path, monkeypatch, release):
    """One running turn holds the authority busy; a second session's API admission is queued and
    observed. ``release(authority, admission_id)`` ends the observation; returns (outcome, row)."""
    from gateway import session_finite
    agent = _Agent()
    db, authority, execute, _returned = _authority(tmp_path, agent)
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 1.0, raising=False)
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda authority: None)
    try:
        with db:
            await authority.submit(ACTOR, Submission(request_id='running', ref=REF, payload={'text': 'a'}, intent='queue'))
            await asyncio.wait_for(asyncio.to_thread(agent.started.wait, 5), 6)
            authority.sessions['t'] = type(authority.sessions['s'])(authority.sessions['s'].source, 'route-t')
            db.create_session('t', source='test')
            monkeypatch.setattr(authority, '_schedule', lambda ref: None)
            ref = REF.__class__('owned', 't')
            receipt = await authority.submit(ACTOR, Submission(
                request_id='queued-api', ref=ref, payload={'text': 'b'}, intent='queue'))
            observer = asyncio.ensure_future(observe_api_turn(
                (authority, ref, get_session_admission(db, admission_id=receipt.admission_id))))
            await async_wait_until(lambda: receipt.admission_id in authority.waiters,
                                   timeout=5, what='the API observer registered')
            await release(authority, receipt.admission_id)
            outcome = (await asyncio.gather(observer, return_exceptions=True))[0]
            return outcome, get_session_admission(db, admission_id=receipt.admission_id)
    finally:
        agent.release.set()
        await async_wait_until(lambda: all(task.done() for task in run_runtime._authority_tasks(authority)),
                               timeout=10, what="the authority's tasks finished")
        db.close()


@pytest.mark.asyncio
async def test_retirement_tells_a_queued_api_observer_its_input_is_saved(tmp_path, monkeypatch):
    async def retire(authority, _admission_id):
        assert await run_runtime._retire_profile_authority(authority) is True

    outcome, row = await _observe_queued_then(tmp_path, monkeypatch, retire)
    assert isinstance(outcome, AdmissionQueued), outcome
    assert outcome.reason == 'admission_queued' and 'do not resend' in str(outcome)
    assert row['status'] == 'queued'  # durable: the next owner runs it once


@pytest.mark.asyncio
async def test_other_observer_failures_keep_their_reason(tmp_path, monkeypatch):
    async def busy(authority, admission_id):
        authority.waiters[admission_id].set_exception(RuntimeStoreError('session_busy'))

    outcome, _row = await _observe_queued_then(tmp_path, monkeypatch, busy)
    assert type(outcome) is RuntimeStoreError and outcome.reason == 'session_busy'


def test_admission_queued_is_a_503_the_sdks_do_not_retry():
    response = store_error_response(_openai_error, AdmissionQueued())
    assert response.status == 503
    assert response.headers.get('x-should-retry') == 'false' and 'Retry-After' not in response.headers
    error = json.loads(response.body)['error']
    assert error['code'] == 'admission_queued' and error['message'] == AdmissionQueued.MESSAGE


def test_a_refused_request_stays_a_retryable_503():
    response = store_error_response(_openai_error, RuntimeStoreError('runtime_draining'))
    assert response.status == 503 and response.headers['Retry-After'] == '1'
    assert 'x-should-retry' not in response.headers

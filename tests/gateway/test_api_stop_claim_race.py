"""Stop of a queued API run that the drain claims mid-stop interrupts the claim, never 409s."""
from types import SimpleNamespace

import pytest

from gateway.session_contract import SessionRef
from hermes_state_runtime import claim_session_input, list_session_admissions
from tests.gateway.test_api_unknown_resolution import _AUTH_HEADERS, _runs_client


@pytest.mark.asyncio
async def test_stop_of_queued_run_claimed_before_cancel_commit_interrupts_the_claim(api, owner):
    interrupted = []
    agent = SimpleNamespace(interrupt=lambda *a: interrupted.append(True))
    owner.runner._cached_agent_for = lambda route: agent
    owner._schedule = lambda ref: None
    client = _runs_client(api)
    await client.start_server()
    try:
        accepted = await client.post('/v1/runs', json={'input': 'HEAD', 'session_id': 'api-session'},
                                     headers={**_AUTH_HEADERS, 'Idempotency-Key': 'race-run'})
        assert accepted.status == 202, await accepted.text()
        run_id = (await accepted.json())['run_id']
        ref = SessionRef(owner.profile_id, 'api-session')
        receipt = owner.receipt

        async def drain_claims_after_stop_read(actor, target, admission_id):
            # The drain claims the head after stop_run read it `queued`, before the cancel commits.
            seen = await receipt(actor, target, admission_id)
            claim_session_input(owner.db, epoch=owner.epoch, session_id='api-session')
            return seen
        owner.receipt = drain_claims_after_stop_read

        stopped = await client.post(f'/v1/runs/{run_id}/stop', headers=_AUTH_HEADERS)
        body = await stopped.json()
        assert stopped.status == 200, body
        assert body['status'] == 'stopping', body
        row = list_session_admissions(owner.db, session_id='api-session')[0]
        assert row['status'] == 'started'
        assert interrupted == [True]  # the claimed generation was told to stop
        assert owner.pending_stops.get(ref.session_id) == row['generation']
        assert row['admission_id'] not in owner.waiters or not owner.waiters[row['admission_id']].done()
    finally:
        for task in list(api._active_run_tasks.values()):
            task.cancel()
        await client.close()

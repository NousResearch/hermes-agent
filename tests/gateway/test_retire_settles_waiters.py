"""A successful profile retirement settles every observer still waiting on a queued admission:
API/webhook/Bot futures fail with runtime_draining instead of hanging, the native set is cleared,
and the durable row stays queued for the profile's next owner (never resent, never dropped)."""
import asyncio

import pytest

from gateway import run_runtime
from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_profile_retire_stops_turns import ACTOR, REF, _Agent, _authority
from gateway.session_contract import Submission


@pytest.mark.asyncio
async def test_successful_retirement_fails_pending_waiters_and_keeps_rows_queued(tmp_path, monkeypatch):
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
            # A second session whose drain paused 'session_busy' (a live worker held its head):
            # its API/webhook/Bot observer deliberately keeps waiting, and no drain is left to
            # release it when retirement refuses the next claim.
            authority.sessions['t'] = type(authority.sessions['s'])(authority.sessions['s'].source, 'route-t')
            db.create_session('t', source='test')
            monkeypatch.setattr(authority, '_schedule', lambda ref: None)
            follower = await authority.submit(ACTOR, Submission(
                request_id='follower', ref=REF.__class__('owned', 't'), payload={'text': 'b'}, intent='queue'))
            waiter = authority.waiters.setdefault(follower.admission_id, asyncio.get_running_loop().create_future())
            authority.native_waiters.add('messaging-delivery')

            assert await run_runtime._retire_profile_authority(authority) is True

            assert waiter.done(), 'observer left pending after retirement'
            with pytest.raises(RuntimeStoreError, match='runtime_draining'):
                waiter.result()
            assert not authority.waiters and not authority.native_waiters
            rows = {r['request_id']: r for r in list_session_admissions(db, session_id='t', pending_only=False)}
            assert rows['follower']['status'] == 'queued'  # durable: the next owner runs it
    finally:
        agent.release.set()
        await asyncio.sleep(.05)
        db.close()

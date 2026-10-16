"""A Stop accepted before the claimed turn adopts its agent reaches the agent that turn runs."""
import asyncio

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent


class Agent:
    def __init__(self):
        self.interrupted = False

    def interrupt(self, *args, **kwargs):
        self.interrupted = True

    def clear_interrupt(self, *args, **kwargs):
        self.interrupted = False


@pytest.mark.asyncio
async def test_stop_before_adoption_reaches_the_rebuilt_agent_not_only_the_stale_cached_one(tmp_path, monkeypatch):
    from gateway import session_finite

    db, authority = _authority(tmp_path, monkeypatch)
    stale, fresh = Agent(), Agent()
    cache = {'route': stale}  # the previous turn's reusable agent
    authority.runner._cached_agent_for = cache.get
    claimed, stopped = asyncio.Event(), asyncio.Event()
    ran = {}

    async def execute(authority, ref, row):
        # Pre-turn hygiene: compression runs on the cached agent, then evicts it; the turn rebuilds.
        claimed.set()
        await asyncio.wait_for(stopped.wait(), 5)
        cache['route'] = fresh
        _wire_turn_agent(authority, row['generation'], fresh)
        ran[row['request_id']] = fresh.interrupted
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'compressing')
        await asyncio.wait_for(claimed.wait(), 5)
        handle = await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        assert handle.execution_state == 'running'
        assert stale.interrupted, 'a resident agent (e.g. one compressing) still gets the Stop'
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert ran['compressing'] is True, 'the agent the stopped turn adopts must start interrupted'

        # The latch was this generation's: the next turn on the same fresh agent runs normally.
        fresh.clear_interrupt()
        await _submit(authority, 'next')
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert ran['next'] is False
        assert not authority.pending_stops

"""A Stop after the turn adopted its agent reaches that agent even if the cache no longer holds it."""
import asyncio

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent
from tests.gateway.test_session_authority_preadopt_stop import Agent


@pytest.mark.asyncio
async def test_stop_between_adoption_and_promotion_reaches_the_adopted_agent_after_cache_eviction(tmp_path, monkeypatch):
    """adopt_agent ran, the route's running slot is still the claim's pending sentinel, and the
    agent cache entry was evicted (pressure / idle sweep): the Stop must still interrupt the agent
    this generation adopted, not resolve nobody through the cache and latch nothing."""
    from gateway import session_finite

    db, authority = _authority(tmp_path, monkeypatch)
    fresh = Agent()
    cache = {}
    authority.runner._cached_agent_for = cache.get
    adopted, stopped = asyncio.Event(), asyncio.Event()
    ran = {}

    async def execute(authority, ref, row):
        cache['route'] = fresh
        _wire_turn_agent(authority, row['generation'], fresh)
        cache.pop('route')  # evicted before the running slot is promoted
        adopted.set()
        await asyncio.wait_for(stopped.wait(), 5)
        ran[row['request_id']] = fresh.interrupted
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'adopted')
        await asyncio.wait_for(adopted.wait(), 5)
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert ran['adopted'] is True, 'the Stop reached nobody: the adopted agent ran on'
        assert not authority.adopted, 'settlement must drop the adopted agent reference'


@pytest.mark.asyncio
async def test_retire_stop_between_adoption_and_promotion_reaches_the_adopted_agent(tmp_path, monkeypatch):
    """The same window on the retire/shutdown path: stop_authority_turns(in_process=True) used to
    re-resolve the running slot (still the pending sentinel) and latch a generation adopt_agent had
    already consumed, so the adopted agent ran on through the profile stop."""
    from gateway import session_finite
    from gateway.run_runtime import stop_authority_turns

    db, authority = _authority(tmp_path, monkeypatch)
    fresh = Agent()
    authority.runner._cached_agent_for = {}.get
    adopted, stopped = asyncio.Event(), asyncio.Event()
    ran = {}

    async def execute(authority, ref, row):
        _wire_turn_agent(authority, row['generation'], fresh)
        adopted.set()
        await asyncio.wait_for(stopped.wait(), 5)
        ran[row['request_id']] = fresh.interrupted
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'adopted')
        await asyncio.wait_for(adopted.wait(), 5)
        stop_authority_turns(authority, in_process=True)
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert ran['adopted'] is True, 'the profile Stop reached nobody: the adopted agent ran on'

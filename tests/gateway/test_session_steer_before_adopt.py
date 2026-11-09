"""A steer accepted while the claimed turn is still building its agent reaches that agent."""
import asyncio
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.session_busy_controls import correct
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent


class Agent(InterruptControlMixin):
    def __init__(self):
        self._pending_steer = None
        self._interrupt_requested = False

    def interrupt(self, *args, **kwargs):
        self._interrupt_requested = True


@pytest.mark.asyncio
async def test_steer_before_adoption_reaches_the_rebuilt_agent_not_the_stale_cached_one(tmp_path, monkeypatch):
    from gateway import session_finite

    db, authority = _authority(tmp_path, monkeypatch)
    stale, fresh = Agent(), Agent()
    cache = {'route': stale}  # the previous turn's reusable agent; this turn rebuilds
    authority.runner._cached_agent_for = cache.get
    claimed, steered = asyncio.Event(), asyncio.Event()
    seen = {}

    async def execute(authority, ref, row):
        claimed.set()
        await asyncio.wait_for(steered.wait(), 5)
        fresh._pending_steer = None
        cache['route'] = fresh
        # The real per-turn wiring runs on the executor thread and calls adopt_agent.
        await asyncio.to_thread(_wire_turn_agent, authority, row['generation'], fresh)
        seen[row['request_id']] = fresh._drain_pending_steer()
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        await _submit(authority, 'building')
        await asyncio.wait_for(claimed.wait(), 5)
        generation = authority._handle(REF).execution_generation
        for text in ('use the staging db', 'and skip tests'):
            reply = await correct(connection, REF, {'session_id': 's', 'text': text,
                                                    'execution_generation': generation}, verb='steer')
            assert reply['status'] == 'queued'
        steered.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert seen['building'] == 'use the staging db\nand skip tests', 'the acknowledged steer was lost'
        assert stale._pending_steer is None, 'the replaced cached agent must not swallow it'

        # The latch was that generation's: the next turn's agent starts with nothing pending.
        await _submit(authority, 'next')
        steered.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert seen['next'] is None
        assert not authority.pending_steers

"""A Stop for a finishing turn reaches only that turn, never the next one on the cached agent."""
import asyncio

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent


class Agent:
    """The session's cached agent: one interrupt flag shared by every turn that reuses it."""

    def __init__(self):
        self.interrupted = False

    def interrupt(self, *args, **kwargs):
        self.interrupted = True

    def clear_interrupt(self, *args, **kwargs):
        self.interrupted = False


@pytest.mark.asyncio
async def test_a_stop_after_the_turn_finished_does_not_cancel_the_next_turn(tmp_path, monkeypatch):
    from gateway import session_finite

    db, authority = _authority(tmp_path, monkeypatch)
    agent = Agent()
    authority.runner._cached_agent_for = lambda route: agent
    finished, settle, ready, wired = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    started = {}

    async def execute(authority, ref, row):
        if row['request_id'] == 'stopped-early':
            ready.set()
            await asyncio.wait_for(wired.wait(), 5)  # Stop lands before this turn wires its agent.
        _wire_turn_agent(authority, row['generation'], agent)
        started[row['request_id']] = agent.interrupted
        if row['request_id'] == 'finishing':
            agent.clear_interrupt()  # The turn's finalizer.
            finished.set()
            await asyncio.wait_for(settle.wait(), 5)  # Its tail, before the claim is settled.
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'finishing')
        await asyncio.wait_for(finished.wait(), 5)
        # A repeated Stop for that turn arrives while it is still settling.
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        assert agent.interrupted
        settle.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)

        await _submit(authority, 'next')
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert started['next'] is False, 'the late Stop must not start the next turn interrupted'

        # A Stop for a turn that has not wired the cached agent yet still reaches it.
        await _submit(authority, 'stopped-early')
        await asyncio.wait_for(ready.wait(), 5)
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        wired.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        assert started['stopped-early'] is True

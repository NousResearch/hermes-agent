"""A Stop for a turn whose execution already returned never reaches the next turn's agent (#143290)."""
import asyncio
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit


@pytest.mark.asyncio
async def test_stop_while_the_finished_turn_settles_spares_the_next_turns_agent(tmp_path, monkeypatch):
    from gateway import session_finite, session_settlement_recovery

    db, authority = _authority(tmp_path, monkeypatch)
    stops = []
    next_agent = SimpleNamespace(interrupt=lambda *a, **k: stops.append('next'))
    # Own-agent runs (/v1/runs) install the next turn's agent on the route before it is claimed.
    authority.runner._running_agents = {'route': next_agent}
    authority.runner._resident_agent_for = lambda route: next_agent
    returned, settle = asyncio.Event(), asyncio.Event()
    settle_with_retry = session_settlement_recovery.settle_with_retry

    async def lagging_settlement(*args):
        returned.set()
        await asyncio.wait_for(settle.wait(), 5)
        return await settle_with_retry(*args)
    monkeypatch.setattr(session_settlement_recovery, 'settle_with_retry', lagging_settlement)
    monkeypatch.setattr(session_finite, 'execute_finite_admission',
                        lambda authority, ref, row: asyncio.sleep(0, result='done'))

    with db:
        await _submit(authority, 'finished')
        await asyncio.wait_for(returned.wait(), 5)
        handle = authority._handle(REF)
        assert handle.execution_state == 'running'  # the lagging settlement still reads as its claim
        await authority.interrupt(ACTOR, REF, handle.execution_generation)
        settle.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
    assert stops == [], 'the Stop for the returned turn must not cancel the next one'

"""prompt.cancel on a started admission stops it in the same RPC (#143547 ACP-23): an editor that
exits right after its cancel never sends the follow-up ``session.interrupt``."""
import asyncio

import pytest

# Imported before the home-I/O guard: the drain's lazy import of it probes the checkout root.
import gateway.session_managed_worker

from tests.gateway.test_canonical_stop_hard_cancel import Agent
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit, _wire_turn_agent


@pytest.mark.asyncio
async def test_prompt_cancel_stops_a_started_turn_without_a_follow_up_interrupt(tmp_path, monkeypatch):
    from gateway import session_finite
    from gateway.session_controls import AuthorityConnection

    db, authority = _authority(tmp_path, monkeypatch)
    agent = Agent()
    authority.runner._cached_agent_for = {}.get
    claimed, stopped = asyncio.Event(), asyncio.Event()

    async def execute(authority, ref, row):
        _wire_turn_agent(authority, row['generation'], agent)
        claimed.set()
        await asyncio.wait_for(stopped.wait(), 5)
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        receipt = await _submit(authority, 'turn')
        await asyncio.wait_for(claimed.wait(), 5)
        conn = AuthorityConnection(authority, object(), {'user_id': 'human'})
        reply = await conn.dispatch({'id': 1, 'method': 'prompt.cancel', 'params': {
            'session_id': REF.session_id, 'admission_id': receipt.admission_id}})
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
    assert 'error' not in reply, reply
    assert reply['result']['admission_id'] == receipt.admission_id
    assert agent.stops == ['hard'], 'the started turn kept running'

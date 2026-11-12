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


@pytest.mark.asyncio
async def test_prompt_cancel_of_a_transferred_steer_is_refused_and_leaves_its_receiving_turn_running(
        tmp_path, monkeypatch):
    """B was steered into running A and A never read its text. B reads started (as A's claim), but
    cancelling it must not stop A: it is refused, and B goes back to its queue place and runs once."""
    from tests.gateway.test_prompt_queued_controls import _call, _connection, _ledger, interactive
    from tests.gateway.test_prompt_queued_transfer import BufferedAgent, _drained
    from tests.waiting import async_wait_until
    from gateway import session_finite

    class StoppableAgent(BufferedAgent):  # the real steer buffer, plus the stops a Stop would make
        def __init__(self):
            super().__init__()
            self.stops = []

        def interrupt(self, *args, **kwargs):
            self.stops.append('soft')
            return True

        def hard_interrupt(self, *args, **kwargs):
            self.stops.append('hard')

    db, authority = _authority(tmp_path, monkeypatch)
    interactive(db)
    agent, claimed, release, ran = StoppableAgent(), asyncio.Event(), asyncio.Event(), []

    async def execute(authority, ref, row):
        ran.append(row['payload']['text'])
        authority.adopt_agent(ref.session_id, row['generation'], agent)
        claimed.set()
        await asyncio.wait_for(release.wait(), 30)
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    with db:
        await _submit(authority, 'running')
        await asyncio.wait_for(claimed.wait(), 30)
        queued = await _submit(authority, 'follow-up')
        conn = _connection(authority)
        assert 'error' not in await _call(conn, 'prompt.steer_queued', admission_id=queued.admission_id)
        reply = await _call(conn, 'prompt.cancel', admission_id=queued.admission_id)
        release.set()
        await _drained(authority)
        await async_wait_until(lambda: len(ran) == 2, timeout=30, what='the returned admission ran')
        await _drained(authority)
    assert agent.stops == [], 'cancelling the transferred prompt stopped the turn it was steered into'
    assert reply.get('error', {}).get('data', {}).get('reason') == 'stale_generation', reply
    assert _ledger(db) == [('running', 'terminal', 'completed'), ('follow-up', 'terminal', 'completed')]

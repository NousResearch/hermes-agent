"""prompt.cancel on a STARTED admission stops the running turn in the same RPC.

The old contract answered stale_generation for a started admission and left the session.interrupt
to the cancelling client's follow-up RPC. A client that dies before sending that follow-up - the
normal shape for an editor that aborts a run by killing its agent process - therefore lost the
interrupt, and the gateway-side turn kept running with nobody left to stop it.
"""
import asyncio

import pytest

# Pre-import at collection time, before the home-I/O guard fixture installs: the drain's first
# lazy `from gateway.session_managed_worker import ManagedExecutionUnknown` otherwise imports
# hermes_bootstrap mid-test, whose module-level _settle_interrupted_update() probes the checkout
# root — inside a linked worktree that resolves into the guarded real home and kills the drain
# (this is also why tests/gateway/test_canonical_stop_hard_cancel.py fails when run alone).
import gateway.session_managed_worker

from tests.gateway.test_canonical_stop_hard_cancel import Agent
from tests.gateway.test_session_authority_cancel_settlement import (
    REF, _authority, _submit, _wire_turn_agent)


@pytest.mark.asyncio
async def test_prompt_cancel_on_a_started_turn_stops_it_in_the_same_rpc(tmp_path, monkeypatch):
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
        # The interrupt reached the running turn in this same RPC: no stale_generation error
        # asking the (already-dying) client to finish the job.
        assert 'error' not in reply, f'prompt.cancel refused the started turn: {reply}'
        assert agent.stops == ['hard'], f'the started turn kept running: {agent.stops}'
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        final = reply['result']['status']
    assert final in ('started', 'terminal'), f'unexpected receipt after stop: {final}'


@pytest.mark.asyncio
async def test_prompt_cancel_after_the_turn_settled_does_not_error(tmp_path, monkeypatch):
    """A started row that settles between the receipt and the interrupt must not surface
    stale_generation: the cancel already succeeded in the only sense that matters."""
    from gateway import session_finite
    from gateway.session_controls import AuthorityConnection

    db, authority = _authority(tmp_path, monkeypatch)
    agent = Agent()
    authority.runner._cached_agent_for = {}.get
    claimed = asyncio.Event()

    async def execute(authority, ref, row):
        _wire_turn_agent(authority, row['generation'], agent)
        claimed.set()
        await asyncio.sleep(0.05)
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        receipt = await _submit(authority, 'turn')
        await asyncio.wait_for(claimed.wait(), 5)
        # The turn finishes (drain returns) BEFORE the cancel is dispatched; the row may still
        # read started or already terminal, and either way prompt.cancel must not raise.
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        conn = AuthorityConnection(authority, object(), {'user_id': 'human'})
        reply = await conn.dispatch({'id': 1, 'method': 'prompt.cancel', 'params': {
            'session_id': REF.session_id, 'admission_id': receipt.admission_id}})
    assert 'error' not in reply, f'prompt.cancel raised on a settled turn: {reply}'

"""A correction for a claimed turn with no live agent is refused, not errored (#141644).

When a Desktop pane keeps a stale busy belief after it rebinds, the user's next message goes
out as ``session.steer`` / ``session.redirect`` for a generation whose agent is gone or not
wired yet. The handler used to answer that with a JSON-RPC ``execution_not_ready`` error, and
every client turns an error into "the send failed": the Desktop composer discards the
optimistic row and drops the text, so the message is unrecoverable.

'Rejected' is the reply both ends already define for "not corrected — queue it as the next
prompt" (``SessionRedirectResponse`` in the Desktop, and tui_gateway's idle-steer answer), so
the text survives as the next turn. A genuinely stale generation and an agent that cannot
carry the verb keep their errors.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_busy_controls import correct
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_authority_cancel_settlement import (
    ACTOR, REF, _authority, _submit, _wire_turn_agent)


async def _claim_without_agent(authority, monkeypatch, request_id='building'):
    """Submit a turn and hold it at the claim, before the turn wires its agent."""
    from gateway import session_finite

    claimed, release = asyncio.Event(), asyncio.Event()

    async def execute(authority, ref, row):
        claimed.set()
        await asyncio.wait_for(release.wait(), 5)
        return 'done'

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    await _submit(authority, request_id)
    await asyncio.wait_for(claimed.wait(), 5)
    return release


@pytest.mark.asyncio
async def test_redirect_for_a_claimed_turn_with_no_agent_is_rejected_not_errored(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation

        reply = await correct(connection, REF, {'session_id': 's', 'text': 'second thought',
                                               'execution_generation': generation}, verb='redirect')

        assert reply == {'status': 'rejected', 'text': 'second thought',
                         'execution_generation': generation, 'authority_epoch': authority.epoch}
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)


@pytest.mark.asyncio
async def test_steer_in_the_build_window_still_latches_for_the_adopting_agent(tmp_path, monkeypatch):
    """The latch path is untouched: a steer that CAN reach the turn is not refused."""
    db, authority = _authority(tmp_path, monkeypatch)
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation

        reply = await correct(connection, REF, {'session_id': 's', 'text': 'use the staging db',
                                                'execution_generation': generation}, verb='steer')

        assert reply['status'] == 'queued'
        assert authority.pending_steers[REF.session_id][0] == generation
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)


@pytest.mark.asyncio
async def test_a_stale_generation_still_raises(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation

        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            await correct(connection, REF, {'session_id': 's', 'text': 'late',
                                            'execution_generation': generation + 1}, verb='redirect')
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)


@pytest.mark.asyncio
async def test_an_adopted_agent_that_cannot_carry_the_verb_still_raises(tmp_path, monkeypatch):
    """A live agent without ``steer`` is a capability error, not a stale busy belief."""
    db, authority = _authority(tmp_path, monkeypatch)
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation
        _wire_turn_agent(authority, generation, SimpleNamespace(interrupted=0))

        with pytest.raises(RuntimeStoreError, match='execution_not_ready'):
            await correct(connection, REF, {'session_id': 's', 'text': 'steer me',
                                            'execution_generation': generation}, verb='steer')
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)

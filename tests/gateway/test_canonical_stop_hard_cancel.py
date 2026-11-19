"""Canonical Stop (Desktop, Ink, classic CLI view, ACP) hard-cancels the turn and leaves background delegations
running: a background subagent is independent of the chat that spawned it (Teknium ruling); `/stop` ends them."""
import asyncio
from unittest.mock import MagicMock

import pytest

from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit, _wire_turn_agent


class Agent:
    def __init__(self):
        self.stops = []

    def interrupt(self, *args, **kwargs):
        self.stops.append('hard' if kwargs.get('hard_cancel') else 'soft')

    def hard_interrupt(self, message=None, *, tool_reason=None):
        self.stops.append('hard')

    def clear_interrupt(self, *args, **kwargs):
        pass


def _delegation(ad, monkeypatch, parent):
    fn = MagicMock()
    monkeypatch.setitem(ad._records, 'd-' + parent, {
        'delegation_id': 'd-' + parent, 'status': 'running', 'session_key': 'other-route',
        'origin_ui_session_id': 'other', 'parent_session_id': parent, 'interrupt_fn': fn})
    return fn


@pytest.mark.asyncio
@pytest.mark.parametrize('when', ['adopted', 'before_adoption'])
async def test_canonical_stop_hard_cancels_the_turn_and_keeps_background_delegations(tmp_path, monkeypatch, when):
    from gateway import session_finite
    from tools import async_delegation as ad

    db, authority = _authority(tmp_path, monkeypatch)
    authority.runner._cached_agent_for = {}.get
    agent = Agent()
    own, foreign = _delegation(ad, monkeypatch, 's'), _delegation(ad, monkeypatch, 'unrelated')
    claimed, stopped = asyncio.Event(), asyncio.Event()

    async def execute(authority, ref, row):
        if when == 'adopted':
            _wire_turn_agent(authority, row['generation'], agent)
        claimed.set()
        await asyncio.wait_for(stopped.wait(), 5)
        if when == 'before_adoption':  # the Stop latched; the agent this generation adopts fires it hard
            _wire_turn_agent(authority, row['generation'], agent)
        return 'done'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await _submit(authority, 'turn')
        await asyncio.wait_for(claimed.wait(), 5)
        await authority.interrupt(ACTOR, REF, authority._handle(REF).execution_generation)
        stopped.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)
    assert agent.stops == ['hard'], f'canonical Stop delivered {agent.stops}; main hard-cancels'
    assert not own.called, "Stop of the turn ended the session's background delegation"
    assert not foreign.called, "Stop ended another session's delegation"

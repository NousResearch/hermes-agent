"""A canonical ``session.steer`` reaches the subagents the running turn is waiting on (#141973).

A parent blocked inside ``delegate_task`` reads its own steer only after the child finishes, so
the correction also goes to the parent's live children, as the messaging gateway already does.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from agent.interrupt_control import InterruptControlMixin
from gateway.session_busy_controls import correct
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _wire_turn_agent
from tests.gateway.test_session_steer_stale_busy_refusal import _claim_without_agent


def _agent(children=()):
    agent = InterruptControlMixin()
    agent._active_children = list(children)
    agent._active_children_lock = threading.Lock()
    return agent


@pytest.mark.asyncio
async def test_an_accepted_steer_also_reaches_the_running_subagent(tmp_path, monkeypatch):
    child = _agent()
    parent = _agent(children=[child])
    db, authority = _authority(tmp_path, monkeypatch)
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation
        _wire_turn_agent(authority, generation, parent)

        reply = await correct(connection, REF, {'session_id': 's', 'text': 'skip the tests',
                                               'execution_generation': generation}, verb='steer')

        assert reply['status'] == 'queued', reply
        assert child._drain_pending_steer() == 'skip the tests'
        assert parent._drain_pending_steer() == 'skip the tests'
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)

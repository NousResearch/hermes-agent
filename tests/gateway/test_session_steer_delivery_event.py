"""An accepted steer tells the client whether ``steer.delivered`` will follow.

Desktop holds a steer under the live reply until the agent appends it after the running tool
batch, which it learns from ``steer.delivered``. The flag lives on the ACK so a client of an
agent with no delivery sink (or an older gateway) keeps sealing at arrival.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_busy_controls import correct
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _wire_turn_agent
from tests.gateway.test_session_steer_stale_busy_refusal import _claim_without_agent


@pytest.mark.asyncio
@pytest.mark.parametrize('announces', [True, False])
async def test_steer_ack_carries_delivery_event_only_with_a_delivery_sink(tmp_path, monkeypatch, announces):
    db, authority = _authority(tmp_path, monkeypatch)
    connection = SimpleNamespace(authority=authority, actor=ACTOR)
    with db:
        release = await _claim_without_agent(authority, monkeypatch)
        generation = authority._handle(REF).execution_generation
        agent = SimpleNamespace(interrupted=0, steer=lambda text: True,
                                steer_delivered_callback=(lambda texts: None) if announces else None)
        _wire_turn_agent(authority, generation, agent)

        reply = await correct(connection, REF, {'session_id': 's', 'text': 'prefer B',
                                               'execution_generation': generation}, verb='steer')

        assert reply['status'] == 'queued'
        assert reply.get('delivery_event') is (True if announces else None)
        release.set()
        await asyncio.wait_for(authority.sessions['s'].task, 5)

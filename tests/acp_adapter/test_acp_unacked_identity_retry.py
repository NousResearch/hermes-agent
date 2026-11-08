"""ACP keeps a retained submission identity through its own retry: a failed reconnect keeps an
un-acked prompt's input_id, and a slash control's retry keeps its prepared request_id."""
import asyncio

import acp
import pytest

from hermes_cli.gateway_client import GatewayClientError
from hermes_state_runtime import list_session_admissions
from tests.acp_adapter import test_retained_identity_retirement as base
from tests.acp_adapter.test_retained_identity_retirement import _acp, _model, _settled

owner = base.owner  # the shared real-authority fixture


@pytest.mark.asyncio
async def test_failed_reconnect_keeps_the_unacked_input_id(owner):
    agent = await _acp(owner)
    try:
        owner.wire.lose = ('prompt.submit', lambda params: True, 'after')
        with pytest.raises(GatewayClientError, match='outcome is unknown'):
            await agent.prompt([acp.text_block('yes')], owner.sid)
        await _settled(owner, 1)
        reconnect = agent._client

        async def unreachable():
            agent._client = reconnect
            raise GatewayClientError('gateway unreachable')
        agent._client = unreachable
        with pytest.raises(GatewayClientError, match='unreachable'):
            await agent.prompt([acp.text_block('yes')], owner.sid)
        await asyncio.wait_for(agent.prompt([acp.text_block('yes')], owner.sid), 5)
        assert owner.executed == ['yes']
        assert len(list_session_admissions(owner.authority.db, session_id=owner.sid, pending_only=False)) == 1
    finally:
        agent._event_task.cancel()


@pytest.mark.asyncio
async def test_slash_control_retry_after_lost_reply_is_one_mutation(owner):
    agent = await _acp(owner)
    try:
        owner.wire.lose = ('session.mutate', _model, 'after')
        with pytest.raises(GatewayClientError, match='outcome is unknown'):
            await agent.prompt([acp.text_block('/model model-a')], owner.sid)
        revision = owner.authority.db.get_session(owner.sid)['runtime_revision']
        await asyncio.wait_for(agent.prompt([acp.text_block('/model model-a')], owner.sid), 5)
        # The retry replays the committed receipt: no second mutation (a /compress would summarize twice).
        assert owner.authority.db.get_session(owner.sid)['runtime_revision'] == revision
        assert owner.authority.db.get_session(owner.sid)['model'] == 'model-a'
    finally:
        agent._event_task.cancel()

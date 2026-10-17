"""A cancel latched while ``prompt.submit`` is in flight belongs to that prompt only.

When the retried submit comes back ``unknown`` (or its terminal receipt cannot be read), the
prompt raises before it consumes the latch; the latch must not survive to cancel the session's
next prompt.
"""
import asyncio

import acp
import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError


class Wire:
    """Gateway client stub: the first submit blocks until released and answers ``first``."""

    def __init__(self, agent, first):
        self.agent, self.first = agent, first
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.submits, self.cancels = 0, []

    async def rpc(self, method, **params):
        if method == 'prompt.submit':
            self.submits += 1
            if self.submits == 1:
                self.entered.set()
                await self.release.wait()
                return self.first
            self.agent._terminals['a2'] = {'outcome': 'completed'}
            return {'admission_id': 'a2', 'status': 'queued'}
        if method == 'prompt.receipt':
            raise GatewayRPCError('not_found')
        if method == 'prompt.cancel':
            self.cancels.append(params['admission_id'])
            return {}
        raise AssertionError(method)


@pytest.mark.asyncio
@pytest.mark.parametrize('first', [
    {'admission_id': 'a1', 'status': 'unknown'},
    {'admission_id': 'a1', 'status': 'terminal', 'outcome': 'completed'},
], ids=['unknown-receipt', 'unreadable-terminal-receipt'])
async def test_cancel_during_failed_submit_does_not_cancel_next_prompt(first):
    agent = GatewayACPAgent()
    wire = Wire(agent, first)
    agent._gateway = wire
    agent._snapshots['s'] = {'pending': []}

    first_prompt = asyncio.create_task(agent.prompt([acp.text_block('first')], 's'))
    await asyncio.wait_for(wire.entered.wait(), 5)
    await agent.cancel('s')  # latched: the admission id is not known yet
    wire.release.set()
    with pytest.raises((GatewayClientError, GatewayRPCError)):
        await asyncio.wait_for(first_prompt, 5)

    response = await asyncio.wait_for(agent.prompt([acp.text_block('second')], 's'), 5)
    assert response.stop_reason == 'end_turn'
    assert wire.cancels == []


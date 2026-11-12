"""Attach command notices follow the reply even when speed discovery yields."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from hermes_agent.acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.asyncio
async def test_speed_discovery_does_not_publish_commands_before_the_attach_reply(monkeypatch):
    monkeypatch.setattr('hermes_agent.acp_adapter.gateway_session_meta.model_state', lambda info: None)
    monkeypatch.setattr('hermes_agent.acp_adapter.gateway_session_meta.provenance_meta', lambda *args: None)
    agent = GatewayACPAgent()
    reply_returned = False
    command_order = []

    async def session_update(*, session_id, update):
        if update.session_update == 'available_commands_update':
            command_order.append(reply_returned)

    async def rpc(method, **params):
        if method == 'config.get':
            await asyncio.gather(*tuple(agent._background))
            return {'value': 'normal', 'options': ['normal', 'fast']}
        return {}

    owner = AsyncMock()
    owner.rpc.side_effect = rpc
    agent._gateway = owner
    monkeypatch.setattr(agent, '_client', AsyncMock(return_value=owner))
    agent._conn = AsyncMock()
    agent._conn.session_update.side_effect = session_update
    fields = await agent._session_fields('session', {'model': 'gpt-6.1-sol'}, 'session')
    reply_returned = True
    await asyncio.gather(*tuple(agent._background))
    assert any(option.id == 'service_tier' for option in fields['config_options'])
    assert command_order == [True]


@pytest.mark.asyncio
async def test_mode_changed_during_speed_discovery_is_used_by_the_attach_reply(monkeypatch):
    monkeypatch.setattr('hermes_agent.acp_adapter.gateway_session_meta.model_state', lambda info: None)
    monkeypatch.setattr('hermes_agent.acp_adapter.gateway_session_meta.provenance_meta', lambda *args: None)
    agent = GatewayACPAgent()

    async def rpc(method, **params):
        await agent._project_mode('session', 'session', {'edit_approval_policy': 'session'})
        return {'value': 'normal', 'options': ['normal', 'fast']}

    owner = AsyncMock()
    owner.rpc.side_effect = rpc
    agent._gateway = owner
    fields = await agent._session_fields('session', {'model': 'gpt-6.1-sol'}, 'session')
    assert fields['modes'].current_mode_id == 'dont_ask'
    assert next(option for option in fields['config_options'] if option.category == 'mode').current_value == 'session'

"""Zed reads its pickers only from ``configOptions`` once an agent sends them, so another surface's mode
or effort change must reach it as ``config_option_update``, not ``current_mode_update`` alone."""
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.asyncio
async def test_another_surfaces_mode_and_effort_change_resend_the_config_options(monkeypatch):
    monkeypatch.setattr('acp_adapter.gateway_session_meta.model_state', lambda info: None)
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    info = {'edit_approval_policy': 'ask', 'model': 'fixture', 'reasoning_effort': 'medium'}
    await agent._project_mode('s', 's', info)  # the attach's snapshot: its reply shows it
    await agent._project_mode('s', 's', {**info, 'edit_approval_policy': 'session'})
    await agent._project_mode('s', 's', {'reasoning_effort': 'high'})  # a partial session.info
    shown = [{option.category: option.current_value for option in call.kwargs['update'].config_options}
             for call in agent._conn.session_update.await_args_list
             if call.kwargs['update'].session_update == 'config_option_update']
    assert shown == [{'mode': 'session', 'thought_level': 'medium'}, {'mode': 'session', 'thought_level': 'high'}]

"""Local reasoning travels as thought events, never as a second formatted final."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent
from gateway.config import Platform
from gateway.run_turn_persistence import GatewayTurnPersistenceMixin


@pytest.mark.asyncio
async def test_streamed_acp_final_is_not_repeated_with_a_reasoning_header(monkeypatch):
    import gateway.run as gateway_run
    from gateway import session_setting_commands

    monkeypatch.setattr(gateway_run, '_load_gateway_config', lambda: {'display': {'show_reasoning': True}})
    monkeypatch.setattr(session_setting_commands, 'local_policy', lambda *args: SimpleNamespace(source='acp'))
    editor = GatewayACPAgent()
    editor._conn = AsyncMock()
    editor._snapshots['s'] = {}

    async def project(kind, text):
        await editor._project({'session_id': 's', 'admission_id': 'a', 'type': kind,
                               'payload': {'text': text, 'outcome': 'completed'}})

    await project('reasoning.delta', 'weigh it')
    await project('message.delta', 'final answer')
    final = GatewayTurnPersistenceMixin()._hmwa_prepend_reasoning(
        {'last_reasoning': 'weigh it'}, 'final answer', SimpleNamespace(platform=Platform.LOCAL), False)
    await project('message.complete', final)
    updates = [call.kwargs['update'] for call in editor._conn.session_update.await_args_list]
    answers = [u.content.text for u in updates if u.session_update == 'agent_message_chunk']
    thoughts = [u.content.text for u in updates if u.session_update == 'agent_thought_chunk']
    assert ''.join(answers) == 'final answer'
    assert thoughts == ['weigh it']

"""A detached ACP question viewer cannot answer for another viewer."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['approval', 'clarify'])
async def test_editor_transport_exception_leaves_shared_question_unanswered(kind):
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    agent._conn.request_permission.side_effect = RuntimeError('editor transport detached')
    agent._gateway = AsyncMock()
    agent._permission('s', {'kind': kind, 'prompt_id': 'question', 'execution_generation': 2,
                            'question': 'Which target?', 'command': 'build', 'choices': ['once']})
    await asyncio.gather(*agent._permissions.values())
    agent._gateway.rpc.assert_not_awaited()
    assert agent._failure is None

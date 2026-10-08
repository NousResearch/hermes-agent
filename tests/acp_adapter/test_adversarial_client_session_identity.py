from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import acp
import pytest
from acp_adapter.gateway_server import GatewayACPAgent

@pytest.mark.asyncio
async def test_compression_preview_addresses_the_editor_retained_physical_alias():
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    gateway = AsyncMock()
    agent._client = AsyncMock(return_value=gateway)
    agent._aliases['editor-tip'] = 'logical-root'
    agent._snapshots['logical-root'] = {}
    agent._mutations = SimpleNamespace(apply=AsyncMock(return_value={'status': 'preview', 'lines': ['retained 10 rows']}), acknowledge=Mock())
    await agent.prompt([acp.text_block('/compress --preview')], 'editor-tip')
    assert agent._conn.session_update.await_args.kwargs['session_id'] == 'editor-tip'

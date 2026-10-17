"""Post-attach notices (command list, context indicator) never outlive ``aclose``."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_server import GatewayACPAgent


@pytest.mark.asyncio
async def test_aclose_stops_a_paused_post_attach_notice_before_it_reconnects(monkeypatch):
    import acp_adapter.gateway_session_meta as meta
    monkeypatch.setattr(meta, 'model_state', lambda info: None)
    monkeypatch.setattr(meta, 'provenance_meta', lambda *args: None)
    connects = []

    def connect_gateway():
        connects.append('after shutdown')
        raise ConnectionRefusedError

    monkeypatch.setattr('acp_adapter.gateway_server.connect_gateway', connect_gateway)
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    paused, release = asyncio.Event(), asyncio.Event()

    async def session_update(**kwargs):
        paused.set()
        await release.wait()  # an editor slow to read the notice

    agent._conn.session_update.side_effect = session_update
    await agent._session_fields('s', {}, 's')
    await asyncio.wait_for(paused.wait(), 5)
    tasks = list(agent._background)

    await agent.aclose()
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
    assert all(task.cancelled() for task in tasks) and not agent._background
    assert connects == []

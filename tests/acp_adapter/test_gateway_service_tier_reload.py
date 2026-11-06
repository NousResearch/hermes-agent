"""Requested speed changes still reach an editor after a frozen-policy reload."""
import pytest

from tests.acp_adapter.test_gateway_service_tier import _tier
from tests.acp_adapter.test_gateway_session_metadata import _updates
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.waiting import async_wait_until


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', [{
    'model': {'provider': 'openai', 'default': 'gpt-6.1-sol', 'base_url': 'https://api.openai.com/v1'},
}], indirect=True)
async def test_peer_normal_selection_after_fast_reload_is_not_hidden_by_the_snapshot(daemon, tmp_path, model_peer):
    async with editor(daemon, tmp_path) as acp, viewer(daemon) as ws:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']
        sid = created['sessionId']
        selected = await acp.rpc('session/set_config_option', sessionId=sid, configId='service_tier', value='fast')
        assert _tier(selected['result'])['currentValue'] == 'fast'
        loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert _tier(loaded['result'])['currentValue'] == 'fast'
        assert (await ws.rpc('session.info', session_id=sid))['service_tier'] == 'priority'
        start = len(acp.frames)
        await ws.rpc('config.set', session_id=sid, key='fast', value='normal')
        await async_wait_until(
            lambda: any(_tier(update)['currentValue'] == 'normal'
                        for update in _updates(acp, sid, 'config_option_update', start)),
            timeout=5, what='the peer Normal selection after a Fast reload')
    assert not model_peer.requests

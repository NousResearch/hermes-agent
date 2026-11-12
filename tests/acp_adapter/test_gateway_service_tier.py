"""Service-tier choices and selections travel through the real ACP/gateway connection."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from hermes_agent.acp_adapter.gateway_server import GatewayACPAgent
from hermes_cli.gateway_client import GatewayRPCError
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.acp_adapter.test_gateway_session_metadata import _updates
from tests.waiting import async_wait_until


def _tier(result):
    options = {option['id']: option for option in result['configOptions']}
    assert 'service_tier' in options, options
    return options['service_tier']


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', [{
    'model': {'provider': 'openai', 'default': 'gpt-6.1-sol', 'base_url': 'https://api.openai.com/v1'},
}], indirect=True)
async def test_tier_picker_changes_only_its_session_and_tracks_other_viewers(daemon, tmp_path, model_peer):
    home = daemon[0]
    original_config = (home / 'config.yaml').read_bytes()
    async with editor(daemon, tmp_path) as acp, viewer(daemon) as ws:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']
        sid = created['sessionId']
        sibling = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
        option = _tier(created)
        assert option['currentValue'] == 'normal'
        assert {'normal', 'fast', 'ultrafast', 'auto', 'cold'} <= {row['value'] for row in option['options']}
        for selected in ('ultrafast', 'auto', 'cold', 'normal'):
            changed = await acp.rpc('session/set_config_option', sessionId=sid,
                                    configId='service_tier', value=selected)
            assert _tier(changed['result'])['currentValue'] == selected, changed
            assert (await ws.rpc('config.get', session_id=sid, key='service_tier'))['value'] == selected
            assert (await ws.rpc('config.get', session_id=sibling, key='service_tier'))['value'] == 'normal'
        changed = await acp.rpc('session/set_config_option', sessionId=sid,
                                configId='service_tier', value='ultrafast')
        assert _tier(changed['result'])['currentValue'] == 'ultrafast'
        invalid = await acp.rpc('session/set_config_option', sessionId=sid,
                                configId='service_tier', value='warp')
        assert 'error' in invalid, invalid
        assert (await ws.rpc('config.get', session_id=sid, key='service_tier'))['value'] == 'ultrafast'
        loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert _tier(loaded['result'])['currentValue'] == 'ultrafast'
        start = len(acp.frames)
        await ws.rpc('config.set', session_id=sid, key='fast', value='fast')
        await async_wait_until(
            lambda: any(_tier(update)['currentValue'] == 'fast'
                        for update in _updates(acp, sid, 'config_option_update', start)),
            timeout=10, what='the other viewer’s service-tier selection')
        assert (await ws.rpc('config.get', session_id=sid, key='service_tier'))['value'] == 'fast'
    assert (home / 'config.yaml').read_bytes() == original_config
    assert not model_peer.requests


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_unsupported_route_has_no_tier_picker_and_refuses_a_direct_pick(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']
        assert 'service_tier' not in {option['id'] for option in created['configOptions']}
        picked = await acp.rpc('session/set_config_option', sessionId=created['sessionId'],
                               configId='service_tier', value='ultrafast')
        assert 'error' in picked, picked


@pytest.mark.asyncio
async def test_older_owner_keeps_existing_pickers_without_a_service_tier_read():
    from hermes_agent.acp_adapter.gateway_session_meta import session_config_options

    owner = AsyncMock()
    owner.rpc.side_effect = GatewayRPCError('invalid_params')
    info = {'model': 'gpt-6.1-sol', 'reasoning_effort': 'high'}
    options = await session_config_options(owner, 'session', None, info, 'default')
    assert {option.id for option in options} == {'edit_approval_policy', 'reasoning_effort'}
    owner.rpc.side_effect = GatewayRPCError('permission_denied')
    with pytest.raises(GatewayRPCError, match='permission_denied'):
        await session_config_options(owner, 'session', None, info, 'default')


@pytest.mark.asyncio
async def test_an_existing_unavailable_tier_is_not_silently_displayed_as_normal():
    from hermes_agent.acp_adapter.gateway_session_meta import session_config_options

    owner = AsyncMock()
    owner.rpc.return_value = {'value': 'ultrafast', 'options': ['normal', 'fast']}
    options = await session_config_options(owner, 'session', None, {'model': 'gpt-6.1-sol'}, 'default')
    tier = next(option for option in options if option.id == 'service_tier')
    assert tier.current_value == 'ultrafast'
    assert next(row.name for row in tier.options if row.value == 'ultrafast') == 'Ultrafast (unavailable on this route)'


@pytest.mark.asyncio
async def test_reconnect_projects_speed_using_the_connection_being_attached(monkeypatch):
    monkeypatch.setattr('hermes_agent.acp_adapter.gateway_session_meta.model_state', lambda info: None)
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    agent._snapshots['session'] = {}
    agent._configs['session'] = {'model': 'gpt-6.1-sol', 'mode': 'default', 'service_tier': ''}

    async def rpc(method, **params):
        if method == 'session.resume':
            return {'info': {'model': 'gpt-6.1-sol', 'service_tier': 'ultrafast'}}
        if method == 'config.get':
            return {'value': 'ultrafast', 'options': ['normal', 'fast', 'ultrafast']}
        return {}

    owner = AsyncMock()
    owner.rpc.side_effect = rpc
    async with agent._connect_lock, asyncio.timeout(2):
        await agent._reattach(owner)
    updates = [call.kwargs['update'] for call in agent._conn.session_update.await_args_list
               if call.kwargs['update'].session_update == 'config_option_update']
    assert any(option.id == 'service_tier' and option.current_value == 'ultrafast'
               for update in updates for option in update.config_options)

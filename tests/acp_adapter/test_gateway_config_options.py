"""Session config options (model + reasoning effort pickers) and the editor's thread list.

Zed (and the ACP spec) build the composer's model and thought-level pickers from ``configOptions``;
the legacy ``models`` field alone renders nothing there. An actual ``hermes acp`` stdio process against
an ordinary isolated gateway, as ``test_gateway_session_metadata.py``.
"""
import pytest

from tests.acp_adapter.test_gateway_model_pick_provider import _SECOND, _second_on_peer
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.waiting import async_wait_until


def _option(options, config_id):
    option = next((option for option in options or [] if option['id'] == config_id), None)
    assert option is not None, (config_id, options)
    return option


def _values(option):
    return [choice['value'] for choice in option['options']]


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_new_and_load_advertise_model_and_effort_pickers(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        result = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']
        sid = result['sessionId']
        model = _option(result.get('configOptions'), 'model')
        assert model['category'] == 'model' and model['type'] == 'select', result
        # The picker's current value is the same choice the legacy ``models`` field names.
        assert model['currentValue'] == result['models']['currentModelId'], result
        assert model['currentValue'] in _values(model), model
        effort = _option(result['configOptions'], 'reasoning_effort')
        assert effort['category'] == 'thought_level' and effort['type'] == 'select', result
        # ``default`` hands the level back to the profile (the owner's ``/reasoning reset``).
        assert {'default', 'none', 'low', 'medium', 'high'} <= set(_values(effort)), effort
        assert effort['currentValue'] in _values(effort), effort
        # The edit approval policy rides the same list, so the editor keeps its mode control.
        mode = _option(result['configOptions'], 'edit_approval_policy')
        assert mode['category'] == 'mode' and mode['currentValue'] == 'ask', result

        loaded = (await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[]))['result']
        assert _option(loaded.get('configOptions'), 'model')['currentValue'] == model['currentValue'], loaded
        assert _option(loaded['configOptions'], 'reasoning_effort'), loaded


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_effort_and_mode_options_mutate_the_session_and_echo_fresh_options(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']

        changed = await acp.rpc('session/set_config_option', sessionId=sid, configId='reasoning_effort', value='low')
        options = changed['result']['configOptions']
        assert _option(options, 'reasoning_effort')['currentValue'] == 'low', changed
        async with viewer(daemon) as ws:
            assert (await ws.rpc('session.info', session_id=sid))['reasoning_effort'] == 'low'

        changed = await acp.rpc('session/set_config_option', sessionId=sid,
                                configId='edit_approval_policy', value='session')
        assert _option(changed['result']['configOptions'], 'edit_approval_policy')['currentValue'] == 'session'
        async with viewer(daemon) as ws:
            assert (await ws.rpc('session.info', session_id=sid))['edit_approval_policy'] == 'session'

        refused = await acp.rpc('session/set_config_option', sessionId=sid, configId='reasoning_effort',
                                value='not-a-level')
        assert 'error' in refused, refused
        # ``default`` unpins the session's level: the profile's own level shows again, not ``low``.
        reset = await acp.rpc('session/set_config_option', sessionId=sid, configId='reasoning_effort', value='default')
        assert _option(reset['result']['configOptions'], 'reasoning_effort')['currentValue'] != 'low', reset
        # Another editor's own config id is accepted and means nothing here, as before.
        unknown = await acp.rpc('session/set_config_option', sessionId=sid, configId='no-such-option', value='x')
        assert _option(unknown['result']['configOptions'], 'reasoning_effort'), unknown


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', [_SECOND], indirect=True)
async def test_model_option_switches_the_session_model_and_provider(daemon, tmp_path, model_peer):
    _second_on_peer(daemon, model_peer)
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        result = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']
        sid = result['sessionId']
        target = 'custom:second:second-model'
        assert target in _values(_option(result['configOptions'], 'model')), result['configOptions']
        changed = await acp.rpc('session/set_config_option', sessionId=sid, configId='model', value=target)
        assert _option(changed['result']['configOptions'], 'model')['currentValue'] == target, changed
        async with viewer(daemon) as ws:
            info = await ws.rpc('session.info', session_id=sid)
            assert (info.get('provider'), info.get('model')) == ('custom:second', 'second-model'), info
        turn = await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': 'WS_SHARED pick'}])
        assert turn['result']['stopReason'] == 'end_turn', turn
        assert model_peer.requests[-1]['model'] == 'second-model', model_peer.requests[-1]


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_another_surface_effort_change_is_projected_as_config_option_update(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
        async with viewer(daemon) as ws:
            snapshot = await ws.rpc('session.resume', session_id=sid)
            await ws.rpc('session.mutate', session_id=sid, request_id='effort-elsewhere', operation='reasoning',
                         expected_revision=snapshot['revision'],
                         expected_generation=snapshot['execution_generation'], payload={'effort': 'high'})

        def projected():
            for frame in acp.frames:
                update = frame.get('params', {}).get('update', {}) if frame.get('method') == 'session/update' else {}
                if update.get('sessionUpdate') == 'config_option_update':
                    effort = [o for o in update.get('configOptions') or [] if o['id'] == 'reasoning_effort']
                    if effort and effort[0]['currentValue'] == 'high':
                        return True
            return False
        await async_wait_until(projected)


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_editor_threads_list_after_their_first_turn(daemon, tmp_path):
    """The owner stores an ACP thread as ``source='local'`` with ``created_source='acp'``; the
    editor's list must still find it (it was always empty after the gateway migration)."""
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
        turn = await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': 'WS_SHARED listed'}])
        assert turn['result']['stopReason'] == 'end_turn', turn
        listed = await acp.rpc('session/list', cwd=str(tmp_path))
        assert sid in [row['sessionId'] for row in listed['result']['sessions']], listed
        everywhere = await acp.rpc('session/list')
        assert sid in [row['sessionId'] for row in everywhere['result']['sessions']], everywhere
    # A fresh editor (Zed's Import Threads / history) reopens the listed thread with its transcript.
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert 'result' in loaded, loaded
        assert 'WS_SHARED listed' in str(acp.frames)


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_picked_effort_reaches_the_model_request(daemon, tmp_path, model_peer):
    """The picker is not cosmetic: the next turn's wire request carries the picked level."""
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
        picked = await acp.rpc('session/set_config_option', sessionId=sid, configId='reasoning_effort', value='high')
        assert 'result' in picked, picked
        turn = await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': 'WS_SHARED think'}])
        assert turn['result']['stopReason'] == 'end_turn', turn
        assert model_peer.requests[-1].get('reasoning_effort') == 'high', model_peer.requests[-1]

"""Editor session metadata and commands on the shipped gateway path (tracker ACP-1/2/3/6/7/11).

An actual ``hermes acp`` stdio process against an ordinary isolated gateway. Main's in-process
server (``acp_adapter/server.py``, ``acp_adapter/commands.py``) is the reference for each reply.
"""
import json

import pytest

from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.waiting import async_wait_until


def _updates(acp, sid, kind, start=0):
    return [frame['params']['update'] for frame in acp.frames[start:]
            if frame.get('method') == 'session/update' and frame['params'].get('sessionId') == sid
            and frame['params']['update'].get('sessionUpdate') == kind]


async def _reply(acp, sid, text):
    """The agent text a prompt produced before its response (a command's reply)."""
    start = len(acp.frames)
    response = await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': text}])
    assert response.get('result', {}).get('stopReason') == 'end_turn', (text, response)
    return ''.join(update['content']['text'] for update in _updates(acp, sid, 'agent_message_chunk', start))


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_new_session_carries_picker_modes_provenance_and_commands(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        result = created['result']
        sid = result['sessionId']
        assert 'local-wire-stub' in result['models']['currentModelId'], result
        assert result['modes']['currentModeId'] == 'default', result
        assert result['_meta']['hermes']['sessionProvenance']['acpSessionId'] == sid, result
        await async_wait_until(lambda: _updates(acp, sid, 'available_commands_update'))
        advertised = {c['name'] for c in _updates(acp, sid, 'available_commands_update')[0]['availableCommands']}
        assert {'help', 'model', 'tools', 'context', 'reset', 'compress', 'steer', 'queue', 'version',
                'usage', 'status'} <= advertised

        # Selecting the advertised mode / the owner's own edit policy never fails the editor.
        assert 'result' in await acp.rpc('session/set_mode', sessionId=sid, modeId='default')
        assert 'result' in await acp.rpc('session/set_config_option', sessionId=sid,
                                         configId='edit_approval_policy', value='ask')

        loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
        assert loaded['result']['modes']['currentModeId'] == 'default', loaded
        assert 'local-wire-stub' in loaded['result']['models']['currentModelId'], loaded


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_editor_commands_reply_and_turns_report_usage_and_title(daemon, tmp_path):
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
        assert 'Available commands' in await _reply(acp, sid, '/help')
        assert 'Current model: local-wire-stub' in await _reply(acp, sid, '/model')
        for read in ('/version', '/status', '/context', '/usage', '/tools'):
            assert (await _reply(acp, sid, read)).strip(), read
        assert 'No changes from compression: 0 messages' in await _reply(acp, sid, '/compress')

        start = len(acp.frames)
        turn = await acp.rpc('session/prompt', sessionId=sid, prompt=[{'type': 'text', 'text': 'WS_SHARED usage'}])
        assert turn['result']['stopReason'] == 'end_turn', turn
        # The loopback model reports no token counts; the owner's counters still ride the reply.
        assert set(turn['result']['usage']) >= {'inputTokens', 'outputTokens', 'totalTokens'}, turn
        contexts = _updates(acp, sid, 'usage_update', start)
        assert contexts and contexts[-1]['size'] == 256000, contexts

        async with viewer(daemon) as ws:
            snapshot = await ws.rpc('session.resume', session_id=sid)
            await ws.rpc('session.mutate', session_id=sid, request_id='rename-from-ws', operation='rename',
                         expected_revision=snapshot['revision'], payload={'title': 'Renamed elsewhere'})
        await async_wait_until(lambda: [u for u in _updates(acp, sid, 'session_info_update')
                                        if u.get('title') == 'Renamed elsewhere'])

        reset = await _reply(acp, sid, '/reset')
        assert 'Conversation history cleared' in reset
        async with viewer(daemon) as ws:
            assert 'WS_SHARED usage' not in json.dumps((await ws.rpc('session.resume', session_id=sid))['messages'])


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_thread_main_in_process_acp_wrote_lists_and_loads(daemon, tmp_path):
    from hermes_state import SessionDB
    home = daemon[0]
    # The row main's ``SessionManager.save_session`` wrote: source acp, no gateway route or policy.
    with SessionDB(db_path=home / 'state.db') as db:
        db.create_session(session_id='pre-upgrade-acp', source='acp', model='local-wire-stub',
                          model_config={'cwd': str(tmp_path)}, cwd=str(tmp_path))
        db.append_message('pre-upgrade-acp', 'user', 'LEGACY_EDITOR_QUESTION')
        db.append_message('pre-upgrade-acp', 'assistant', 'LEGACY_EDITOR_ANSWER')
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        listed = await acp.rpc('session/list', cwd=str(tmp_path))
        assert 'pre-upgrade-acp' in [row['sessionId'] for row in listed['result']['sessions']], listed
        loaded = await acp.rpc('session/load', sessionId='pre-upgrade-acp', cwd=str(tmp_path), mcpServers=[])
        assert 'result' in loaded, loaded
        assert 'LEGACY_EDITOR_ANSWER' in json.dumps(acp.frames)
        turn = await acp.rpc('session/prompt', sessionId='pre-upgrade-acp',
                             prompt=[{'type': 'text', 'text': 'WS_SHARED after upgrade'}])
        assert turn.get('result', {}).get('stopReason') == 'end_turn', turn

"""Bounded goal continuation through real ACP stdio, WS authority and local inference."""
import asyncio
import json
import threading

import pytest

from tests.acp_adapter.test_gateway_config_options_wire import _restarted, _slash
from tests.acp_adapter.test_gateway_sessions import editor, viewer
from tests.acp_adapter.test_control_runtime_context import RUNTIME_CONTEXT
from tests.waiting import async_wait_until


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', [{'goals': {'max_turns': 2},
    'auxiliary': {'goal_judge': {'provider': 'custom', 'model': 'local-goal-judge'}}}], indirect=True)
async def test_goal_follows_up_in_editor_and_survives_cold_load(daemon, tmp_path, model_peer):
    from tests.gateway.fixtures.acp_goal_peer import ModelPeer
    model_peer.RequestHandlerClass = ModelPeer
    model_peer.judgments = []
    model_peer.blocked, model_peer.release = threading.Event(), threading.Event()
    home = daemon[0]
    config_before = (home / 'config.yaml').read_bytes()
    body = 'Finish WS_SHARED BLOCK_FIFO\n  preserve indentation and --session\n<example>user XML</example>'
    async with viewer(daemon) as ws, editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        sid = created['result']['sessionId']
        await _slash(acp, sid, '/help')
        assert '/goal' in json.dumps(acp.frames)
        await async_wait_until(lambda: any(c['name'] == 'goal' for f in acp.frames
            for c in f.get('params', {}).get('update', {}).get('availableCommands', [])),
            what='goal advertised in ACP palette')
        for cmd in ('/goal', '/goal status'):
            assert 'result' in await _slash(acp, sid, cmd + '\n' + RUNTIME_CONTEXT)
        assert model_peer.requests == model_peer.judgments == []
        running = asyncio.create_task(_slash(acp, sid, '/goal ' + body + '\n' + RUNTIME_CONTEXT))
        try:
            assert await asyncio.to_thread(model_peer.blocked.wait, 15), 'goal kickoff never reached model'
            before = len(acp.frames)
            assert 'result' in await _slash(acp, sid, '/goal pause')
            assert 'Goal paused' in json.dumps(acp.frames[before:])
        finally:
            model_peer.release.set()
            reply = await running
        assert reply.get('result', {}).get('stopReason') == 'end_turn', reply
        assert model_peer.judgments == [], 'A paused goal must not judge or continue the finished turn'
        assert len(model_peer.requests) == 1
        assert 'result' in await _slash(acp, sid, '/goal resume')

        await async_wait_until(lambda: len(model_peer.judgments) == 2, timeout=25, what='bounded goal judging')
        snapshot = await ws.rpc('session.resume', session_id=sid)
        events = (await ws.rpc('session.events.since', session_id=sid,
                    replay_epoch=snapshot['replay_epoch'], last_sequence=0))['events']
        completes = [e for e in events if e['type'] == 'message.complete']
        assert len(completes) == 3, events
        assert len({e['admission_id'] for e in completes}) == 3
        await async_wait_until(lambda: sum('LOCAL_ACK_WS_SHARED' in
            f.get('params', {}).get('update', {}).get('content', {}).get('text', '')
            for f in acp.frames) == 3, what='kickoff and both resumed goal replies in editor')
        assert len(model_peer.requests) == 3
        users = [m['content'] for m in snapshot['messages'] if m['role'] == 'user']
        assert users[0] == body
        assert len(users) == 3 and all('standing goal' in text for text in users[1:])
        assert RUNTIME_CONTEXT not in json.dumps(snapshot['messages'])
        assert 'result' in await _slash(acp, sid, '/goal pause')
    async with _restarted(daemon, tmp_path) as cold:
        async with editor(cold, tmp_path) as acp:
            await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
            assert 'result' in await acp.rpc('session/resume', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
            assert 'result' in await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
            before = len(acp.frames)
            await _slash(acp, sid, '/goal status')
            status = json.dumps(acp.frames[before:])
            assert 'paused' in status and 'preserve indentation' in status, status
            before_requests = len(model_peer.requests)
            assert 'result' in await _slash(acp, sid, '/goal resume')
            await async_wait_until(lambda: len(model_peer.requests) > before_requests,
                                   what='resumed goal reaches model')
            await _slash(acp, sid, '/goal clear')
            before = len(acp.frames)
            await _slash(acp, sid, '/goal')
            assert 'No active goal' in json.dumps(acp.frames[before:])
    assert (home / 'config.yaml').read_bytes() == config_before

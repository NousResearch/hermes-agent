"""Actual daemon, ACP stdio, model HTTP and terminal execution for #137077."""
import asyncio
from contextlib import suppress
import json
import shlex

import pytest

from tests.acp_adapter.test_gateway_sessions import editor
from tests.gateway.fixtures.authority_controls_peer import ModelPeer
from tests.waiting import async_wait_until


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
@pytest.mark.parametrize('dangerous', [False, True])
@pytest.mark.parametrize('choice', ['allow_once', 'deny'])
async def test_gateway_terminal_waits_for_editor(daemon, tmp_path, model_peer, dangerous, choice):
    marker = tmp_path / 'executed.txt'
    command = 'printf approved > ' + shlex.quote(str(marker))
    if dangerous:
        target = tmp_path / 'empty-target'
        target.mkdir()
        command = 'rm -rf -- ' + shlex.quote(str(target)) + '; ' + command
    model_peer.command = command
    model_peer.RequestHandlerClass = ModelPeer
    async with editor(daemon, tmp_path) as client:
        await client.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await client.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        assert 'result' in created, created
        sid = created['result']['sessionId']
        task = asyncio.create_task(client.rpc('session/prompt', sessionId=sid,
            prompt=[{'type': 'text', 'text': 'Run the fixture command'}]))
        try:
            await async_wait_until(lambda: any(f.get('method') == 'session/request_permission' for f in client.frames),
                                   timeout=25, what='terminal permission before execution')
            requests = [f for f in client.frames if f.get('method') == 'session/request_permission']
            assert len(requests) == 1
            request = requests[0]
            assert not marker.exists()
            assert request['params']['sessionId'] == sid
            assert request['params']['toolCall']['kind'] == 'execute'
            assert request['params']['toolCall']['rawInput']['command'] == command
            client.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': request['id'],
                'result': {'outcome': {'outcome': 'selected', 'optionId': choice}}}).encode() + b'\n')
            await client.process.stdin.drain()
            result = await asyncio.wait_for(task, 30)
            assert result['result']['stopReason'] == 'end_turn', result
            assert marker.exists() == (choice == 'allow_once')
            assert len([f for f in client.frames if f.get('method') == 'session/request_permission']) == 1
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

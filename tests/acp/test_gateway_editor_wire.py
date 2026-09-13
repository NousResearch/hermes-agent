"""Real ACP editor permissions over an ordinary daemon's canonical ACP route."""
import asyncio
from contextlib import suppress
from http.server import BaseHTTPRequestHandler
import json

import pytest

from tests.acp.test_gateway_sessions import daemon, editor, viewer, model_peer  # noqa: F401


class EditPeer(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(request)
        tool = not any(m['role'] == 'tool' for m in request.get('messages', []))
        message = {'role': 'assistant', 'content': 'EDITOR_FINISHED'}
        if tool:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'owned-edit', 'type': 'function', 'function': {
                    'name': 'write_file', 'arguments': json.dumps({
                        'path': str(self.server.target), 'content': 'approved edit'})}}]}
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if request.get('stream') else 'application/json')
        self.end_headers()
        if request.get('stream'):
            if tool:
                message['tool_calls'][0]['index'] = 0
            for delta, finish in [(message, None), ({}, 'tool_calls' if tool else 'stop')]:
                self.wfile.write(('data: ' + json.dumps({'id': 'edit', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'fixture', 'choices': [{'index': 0, 'delta': delta,
                    'finish_reason': finish}]}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({'id': 'edit', 'object': 'chat.completion', 'created': 1,
                'model': 'fixture', 'choices': [{'index': 0, 'message': message,
                'finish_reason': 'tool_calls' if tool else 'stop'}]}).encode())
        self.wfile.flush()


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_acp_load_checks_authoritative_cwd(daemon, tmp_path):
    other = tmp_path / 'other'
    other.mkdir()
    async with viewer(daemon) as ws:
        created = await ws.rpc('session.create', source='acp', cwd=str(tmp_path), request_id='cwd')
        sid = created['session_id']
        assert created['info']['cwd'] == str(tmp_path)
        async with editor(daemon, tmp_path) as acp:
            await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
            refused = await acp.rpc('session/load', sessionId=sid, cwd=str(other), mcpServers=[])
            assert 'cwd_policy_conflict' in json.dumps(refused)
            loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
            assert 'result' in loaded


@pytest.mark.linux_only
@pytest.mark.asyncio
async def test_real_editor_diff_requires_native_consent_before_write(daemon, tmp_path, model_peer):
    target = tmp_path / 'owned.txt'
    target.write_text('original edit')
    model_peer.target = target
    model_peer.RequestHandlerClass = EditPeer
    async with viewer(daemon) as ws:
        created = await ws.rpc('session.create', source='acp', cwd=str(tmp_path), request_id='editor-diff')
        sid = created['session_id']
        async with editor(daemon, tmp_path) as acp:
            await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
            loaded = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
            assert 'result' in loaded, loaded
            task = asyncio.create_task(acp.rpc('session/prompt', sessionId=sid,
                prompt=[{'type': 'text', 'text': 'Edit the owned fixture'}]))
            try:
                async with asyncio.timeout(20):
                    while not any(f.get('method') == 'session/request_permission' for f in acp.frames):
                        await asyncio.sleep(.05)
                permission = next(f for f in acp.frames if f.get('method') == 'session/request_permission')
                call = permission['params']['toolCall']
                assert call['kind'] == 'edit', permission
                assert call['content'][0]['oldText'] == 'original edit'
                assert call['content'][0]['newText'] == 'approved edit'
                assert target.read_text() == 'original edit'
                acp.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': permission['id'], 'result': {
                    'outcome': {'outcome': 'selected', 'optionId': 'allow_once'}}}).encode() + b'\n')
                await acp.process.stdin.drain()
                assert (await task)['result']['stopReason'] == 'end_turn'
                assert target.read_text() == 'approved edit'
                print('ACP_EDITOR_RECEIPT=' + json.dumps({'session_id': sid, 'permission': permission,
                    'owned_write_after_consent': True}))
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

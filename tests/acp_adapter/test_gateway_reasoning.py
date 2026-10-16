"""Provider reasoning reaches an ACP editor's thought pane over the gateway route.

Main's in-process ACP server always streamed reasoning as ``agent_thought_chunk``. A gateway
session streams it only when its ``show_reasoning`` flag is on, and an editor has no
``config.set`` to turn it on, so an ACP launch follows its ``display.show_reasoning``.
"""
import json

import pytest

from tests.acp_adapter.test_gateway_sessions import editor
from tests.gateway.fixtures.shared_authority_peer import ModelPeer

REASONING = 'PONDER_PROBE'


class ReasoningPeer(ModelPeer):
    """One streamed reply: a reasoning delta, then the answer."""

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(request)
        deltas = [({'role': 'assistant', 'reasoning_content': REASONING}, None),
                  ({'content': 'ANSWER_DONE'}, None), ({}, 'stop')]
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if request.get('stream') else 'application/json')
        self.end_headers()
        if request.get('stream'):
            for delta, finish in deltas:
                self.wfile.write(('data: ' + json.dumps({'id': 'reasoning', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'fixture', 'choices': [{'index': 0, 'delta': delta,
                    'finish_reason': finish}]}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({'id': 'reasoning', 'object': 'chat.completion', 'created': 1,
                'model': 'fixture', 'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                    'role': 'assistant', 'content': 'ANSWER_DONE', 'reasoning_content': REASONING}}]}).encode())
        self.wfile.flush()


async def _thoughts(daemon, tmp_path, model_peer):
    model_peer.RequestHandlerClass = ReasoningPeer
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        assert 'result' in created, created
        reply = await acp.rpc('session/prompt', sessionId=created['result']['sessionId'],
                              prompt=[{'type': 'text', 'text': 'think out loud'}])
        assert reply.get('result', {}).get('stopReason') == 'end_turn', reply
    updates = [f['params']['update'] for f in acp.frames if f.get('method') == 'session/update']
    return ''.join(u['content'].get('text', '') for u in updates if u['sessionUpdate'] == 'agent_thought_chunk')


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_acp_launch_streams_reasoning_as_thought_chunks(daemon, tmp_path, model_peer):
    assert REASONING in await _thoughts(daemon, tmp_path, model_peer)


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', [{'display': {'show_reasoning': False}}], indirect=True)
async def test_acp_launch_honours_display_show_reasoning_off(daemon, tmp_path, model_peer):
    assert await _thoughts(daemon, tmp_path, model_peer) == ''

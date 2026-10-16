"""An ACP prompt returns delegated results before ``end_turn``.

Editors drop assistant updates a later wake publishes after ``session/prompt`` returns, so a
background ``delegate_task`` on an ACP launch must join its children inside the prompt.
"""
import json

import pytest

from tests.acp_adapter.test_gateway_sessions import editor
from tests.gateway.fixtures.shared_authority_peer import ModelPeer

PARENT_PROMPT = 'DELEGATE_PROBE'
CHILD_GOAL = 'Reply with exactly CHILD_RESULT_42'


class DelegatePeer(ModelPeer):
    """Parent turn: one background ``delegate_task``, then a final reply. Child turn: its result."""

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.server.requests.append(request)
        messages = request.get('messages', [])
        users = [str(m.get('content')) for m in messages if m['role'] == 'user']
        tool_results = [m for m in messages if m['role'] == 'tool']
        message = {'role': 'assistant', 'content': 'PARENT_DONE'}
        if not any(PARENT_PROMPT in text for text in users):
            message = {'role': 'assistant', 'content': 'CHILD_RESULT_42'}
        elif not tool_results:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'delegate', 'type': 'function', 'function': {
                    'name': 'delegate_task', 'arguments': json.dumps({'goal': CHILD_GOAL, 'background': True})}}]}
        else:
            self.server.parent_tool_results = [m['content'] for m in tool_results]
        tool = 'tool_calls' in message
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if request.get('stream') else 'application/json')
        self.end_headers()
        if request.get('stream'):
            if tool:
                message['tool_calls'][0]['index'] = 0
            for delta, finish in [(message, None), ({}, 'tool_calls' if tool else 'stop')]:
                self.wfile.write(('data: ' + json.dumps({'id': 'delegate', 'object': 'chat.completion.chunk',
                    'created': 1, 'model': 'fixture', 'choices': [{'index': 0, 'delta': delta,
                    'finish_reason': finish}]}) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')
        else:
            self.wfile.write(json.dumps({'id': 'delegate', 'object': 'chat.completion', 'created': 1,
                'model': 'fixture', 'choices': [{'index': 0, 'message': message,
                'finish_reason': 'tool_calls' if tool else 'stop'}]}).encode())
        self.wfile.flush()


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_acp_prompt_joins_background_delegate_before_end_turn(daemon, tmp_path, model_peer):
    model_peer.RequestHandlerClass = DelegatePeer
    model_peer.parent_tool_results = None
    async with editor(daemon, tmp_path) as acp:
        await acp.rpc('initialize', protocolVersion=1, clientCapabilities={})
        created = await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[])
        assert 'result' in created, created
        reply = await acp.rpc('session/prompt', sessionId=created['result']['sessionId'],
                              prompt=[{'type': 'text', 'text': PARENT_PROMPT}])
    assert reply.get('result', {}).get('stopReason') == 'end_turn', reply
    assert model_peer.parent_tool_results is not None, 'the parent never saw its delegate_task result'
    result = json.loads(model_peer.parent_tool_results[-1])
    # Joined inline, not a background handle whose completion only a later wake would deliver.
    assert result.get('mode') != 'background', result
    assert [(r['status'], r['summary']) for r in result['results']] == [('completed', 'CHILD_RESULT_42')], result

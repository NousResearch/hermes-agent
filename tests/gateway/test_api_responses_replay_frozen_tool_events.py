"""A keyed streaming Responses retry replays the incremental tool frames exactly as first sent:
terminal trimming of a long tool result must not reach the recorded events."""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


def _tool_output_texts(frames):
    return [frame['item']['output'][0]['text'] for frame in frames
            if frame.get('item', {}).get('type') == 'function_call_output']


@pytest.mark.asyncio
@pytest.mark.parametrize('size', [1500, 200])
async def test_exact_retry_replays_full_incremental_tool_output(api, owner, size):
    result_text = 'x' * size
    calls = []

    async def handle(event):
        from gateway.session_api_turn import publish_api_tool_event
        from gateway.session_results import execution_result
        calls.append(event.text)
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        publish_api_tool_event(owner, sid, generation, 'tool.start', 'call_1', 'read_file', {'path': 'f'})
        publish_api_tool_event(owner, sid, generation, 'tool.complete', 'call_1', 'read_file', {'path': 'f'},
                               result_text)
        execution_result.get()['result'] = {'final_response': 'done', 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'call_1', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': json.dumps({'path': 'f'})}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': result_text},
            {'role': 'assistant', 'content': 'done'}]}
        return 'done'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body = {'input': 'read it', 'stream': True}
    async with TestClient(TestServer(app)) as client:
        streams = []
        for key in ('frozen-tool', 'frozen-tool', 'fresh-key'):
            resp = await client.post('/v1/responses', json=body, headers={'Idempotency-Key': key})
            assert resp.status == 200
            streams.append([json.loads(line[6:]) for line in (await resp.text()).splitlines()
                            if line.startswith('data: ')])
    first, retry, fresh = streams
    assert _tool_output_texts(first) == [result_text, result_text]
    assert _tool_output_texts(retry) == _tool_output_texts(first)
    assert retry == first
    assert _tool_output_texts(fresh) == [result_text, result_text]
    assert retry[-1]['response'] == first[-1]['response']
    assert calls == ['read it', 'read it']

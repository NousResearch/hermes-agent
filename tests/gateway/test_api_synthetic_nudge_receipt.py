"""An internal user-role recovery nudge must not erase earlier output in the same API turn."""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('marked', [False, True])
async def test_tool_output_before_synthetic_user_nudge_survives_settlement(api, owner, stream, marked):
    from gateway.session_api_turn import publish_api_tool_event
    from gateway.session_results import execution_result
    prior = [{'role': 'user', 'content': 'read a'}, {'role': 'assistant', 'content': 'previous answer'}]
    transcript = []

    async def handle(event):
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        publish_api_tool_event(owner, sid, generation, 'tool.start', 'read-1', 'read_file', {'path': 'a'})
        publish_api_tool_event(owner, sid, generation, 'tool.complete', 'read-1', 'read_file', {'path': 'a'}, 'read')
        transcript.extend([*prior,
            {'role': 'user', 'content': event.text},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'read-1', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":"a"}'}}]},
            {'role': 'tool', 'tool_call_id': 'read-1', 'content': 'read'},
            {'role': 'user', 'content': 'Please finish your response.' if marked else event.text,
             **({'_empty_recovery_synthetic': True} if marked else {})},
            {'role': 'assistant', 'content': 'answer'},
        ])
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': transcript,
            **({} if marked else {'current_turn_user_idx': len(prior)})}
        return 'answer'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/responses', json={'input': 'read a', 'stream': stream,
                                                           'conversation_history': prior},
                                     headers={'Idempotency-Key': 'synthetic-nudge'})
        assert response.status == 200
        if stream:
            frames = [json.loads(line[6:]) for line in (await response.text()).splitlines()
                      if line.startswith('data: ')]
            result = frames[-1]['response']
        else:
            result = await response.json()
        stored = api._current_response_store().get(result['id'])
        assert stored['conversation_history'] == transcript
    assert [(item['type'], item.get('call_id')) for item in result['output']] == [
        ('function_call', 'read-1'), ('function_call_output', 'read-1'), ('message', None)]

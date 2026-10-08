"""Incremental output indexes retain their identity in the terminal Responses envelope."""
import json
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_live_tool_output_index_identifies_same_terminal_item(api, owner):
    async def handle(event):
        from gateway.session_api_turn import publish_api_tool_event
        from gateway.session_results import execution_result
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        publish_api_tool_event(owner, sid, generation, 'tool.start', 'call_1', 'read_file', {'path': 'a'})
        publish_api_tool_event(owner, sid, generation, 'tool.complete', 'call_1', 'read_file', {'path': 'a'}, 'read')
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': [
            {'role': 'assistant', 'content': '', 'reasoning': 'reasoned', 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"a"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'read'},
            {'role': 'assistant', 'content': 'answer'}]}
        return 'answer'

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/responses', json={'input': 'work', 'stream': True},
                                     headers={'Idempotency-Key': 'stable-output-index'})
        assert response.status == 200
        frames = [json.loads(line[6:]) for line in (await response.text()).splitlines()
                  if line.startswith('data: ')]
    terminal = frames[-1]['response']['output']
    for frame in frames:
        if frame['type'] in {'response.output_item.added', 'response.output_item.done'}:
            assert terminal[frame['output_index']]['id'] == frame['item']['id']


@pytest.mark.asyncio
async def test_late_stream_observer_receives_defined_output_indices(api, owner):
    entered, release = asyncio.Event(), asyncio.Event()
    async def handle(event):
        from gateway.session_api_turn import publish_api_tool_event
        from gateway.session_results import execution_result
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        publish_api_tool_event(owner, sid, generation, 'tool.start', 'call_1', 'read_file', {'path': 'a'})
        publish_api_tool_event(owner, sid, generation, 'tool.complete', 'call_1', 'read_file', {'path': 'a'}, 'read')
        entered.set()
        await release.wait()
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"a"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'read'},
            {'role': 'assistant', 'content': 'answer'}]}
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers, body = {'Idempotency-Key': 'late-stream-observer'}, {'input': 'work', 'stream': True}
    pending = []
    async with TestClient(TestServer(app)) as client:
        try:
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            await asyncio.wait_for(entered.wait(), 5)
            # Ensure the first stream actually published the tool items before the retry joins.
            async with asyncio.timeout(5):
                while api._current_response_store()._conn.execute('SELECT COUNT(*) FROM response_output_order').fetchone()[0] < 2:
                    await asyncio.sleep(.01)
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            async with asyncio.timeout(5):
                while sum(map(len, owner.api_observers.values())) < 2:
                    await asyncio.sleep(.01)
            release.set()
            responses = await asyncio.gather(*pending)
            second = [json.loads(line[6:]) for line in (await responses[1].text()).splitlines()
                      if line.startswith('data: ')]
            accumulated = []
            for frame in second:
                if frame['type'] == 'response.created':
                    accumulated = frame['response']['output'][:]
                elif frame['type'] == 'response.output_item.added':
                    assert frame['output_index'] == len(accumulated), 'late observer was given an output index with missing earlier items'
                    accumulated.append(frame['item'])
                elif 'output_index' in frame:
                    assert frame['output_index'] < len(accumulated)
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)

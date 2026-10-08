"""A completed observer cannot assign indexes ahead of a delayed live stream's items."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_nonstream_observer_does_not_create_gaps_in_delayed_live_stream(api, owner, monkeypatch):
    from gateway.platforms.api_server_openai_routes import _ResponsesStream
    blocked, release_stream, release_agent = asyncio.Event(), asyncio.Event(), asyncio.Event()
    emit = _ResponsesStream.emit_tool_started
    async def held(self, payload):
        blocked.set()
        await release_stream.wait()
        await emit(self, payload)
    monkeypatch.setattr(_ResponsesStream, 'emit_tool_started', held)

    async def handle(event):
        from gateway.session_api_turn import publish_api_tool_event
        from gateway.session_results import execution_result
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        publish_api_tool_event(owner, sid, generation, 'tool.start', 'call_1', 'read_file', {'path': 'a'})
        publish_api_tool_event(owner, sid, generation, 'tool.complete', 'call_1', 'read_file', {'path': 'a'}, 'read')
        await release_agent.wait()
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': [
            {'role': 'assistant', 'content': '', 'reasoning': 'reasoned', 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"a"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'read'},
            {'role': 'assistant', 'content': 'answer'}]}
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers, body = {'Idempotency-Key': 'backpressured-stream'}, {'input': 'work'}
    pending = []
    async with TestClient(TestServer(app)) as client:
        try:
            pending.append(asyncio.create_task(client.post('/v1/responses', json={**body, 'stream': True}, headers=headers)))
            await asyncio.wait_for(blocked.wait(), 5)
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            async with asyncio.timeout(5):
                while sum(map(len, owner.api_observers.values())) < 2:
                    await asyncio.sleep(.01)
            release_agent.set()
            observer = await asyncio.wait_for(pending[1], 5)
            assert observer.status == 200
            await observer.read()
            release_stream.set()
            original = await pending[0]
            frames = [json.loads(line[6:]) for line in (await original.text()).splitlines()
                      if line.startswith('data: ')]
            accumulated = []
            for frame in frames:
                if frame['type'] == 'response.created':
                    accumulated = frame['response']['output'][:]
                elif frame['type'] == 'response.output_item.added':
                    assert frame['output_index'] == len(accumulated), 'nonstream observer inserted output indexes ahead of the live stream'
                    accumulated.append(frame['item'])
                elif 'output_index' in frame:
                    assert frame['output_index'] < len(accumulated)
        finally:
            release_agent.set()
            release_stream.set()
            await asyncio.gather(*pending, return_exceptions=True)

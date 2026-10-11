"""Responses output ids and indexes stay identical across observers, late joiners and eviction."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_output_indices_survive_completion_order_and_eviction(api, owner):
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def handle(event):
        from gateway.session_api_turn import publish_api_tool_event
        from gateway.session_results import execution_result
        calls.append(event.text)
        entered.set()
        await release.wait()
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        # Parallel tools completing out of start order.
        order = [('tool.start', 'a'), ('tool.start', 'b'), ('tool.complete', 'b'), ('tool.complete', 'a')]
        for kind, call in order:
            publish_api_tool_event(owner, sid, generation, kind, call, 'read_file', {'path': call}, 'read ' + call)
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': [
            {'role': 'assistant', 'content': '', 'reasoning': 'available only in terminal receipt', 'tool_calls': [
                {'id': call, 'type': 'function', 'function': {'name': 'read_file', 'arguments': json.dumps({'path': call})}}
                for call in ('a', 'b')]},
            *[{'role': 'tool', 'tool_call_id': call, 'content': 'read ' + call} for call in ('a', 'b')],
            {'role': 'assistant', 'content': 'answer'}]}
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers, body = {'Idempotency-Key': 'ordered'}, {'input': 'work', 'stream': True}
    pending = []
    async with TestClient(TestServer(app)) as client:
        try:
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            await asyncio.wait_for(entered.wait(), 5)
            pending.append(asyncio.create_task(client.post('/v1/responses', json=body, headers=headers)))
            async with asyncio.timeout(5):
                while sum(map(len, owner.api_observers.values())) < 2:
                    await asyncio.sleep(.01)
            release.set()
            responses = await asyncio.gather(*pending)
            terminal = []
            for response in responses:
                frames = [json.loads(line[6:]) for line in (await response.text()).splitlines() if line.startswith('data: ')]
                final = frames[-1]['response']
                terminal.append(final)
                for frame in frames:
                    if 'output_index' in frame:
                        identity = frame.get('item_id') or frame.get('item', {}).get('id')
                        if identity:
                            assert final['output'][frame['output_index']]['id'] == identity
            assert all(value == terminal[0] for value in terminal)
            store = api._current_response_store()
            store._max_size = 1
            store.put('evict', {'response': {}})
            retry = await client.post('/v1/responses', json={**body, 'stream': False}, headers=headers)
            assert await retry.json() == terminal[0]
            assert calls == ['work']
        finally:
            release.set()
            await asyncio.gather(*pending, return_exceptions=True)


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

"""Sequential/parallel tool emission and every observer keep one durable output-index mapping."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('viewers', [1, 2])
async def test_output_indices_survive_completion_order_and_eviction(api, owner, parallel, viewers):
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
        order = [('tool.start', 'a'), ('tool.start', 'b'), ('tool.complete', 'b'), ('tool.complete', 'a')] if parallel else [
            ('tool.start', 'a'), ('tool.complete', 'a'), ('tool.start', 'b'), ('tool.complete', 'b')]
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
            if viewers == 2:
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

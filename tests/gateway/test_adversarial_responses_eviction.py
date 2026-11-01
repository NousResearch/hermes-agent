"""Durable exact Responses retries keep their output after response-body LRU eviction."""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_terminal_retry_after_cache_eviction_preserves_result(api, owner):
    calls = []
    api._model_name = 'accepted-model'
    async def handle(event):
        from gateway.session_results import execution_result
        from gateway.session_api_turn import publish_api_tool_event
        calls.append(event.text)
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
    body, headers = {'input': 'work', 'stream': True}, {'Idempotency-Key': 'evicted-terminal'}
    async with TestClient(TestServer(app)) as client:
        first = await client.post('/v1/responses', json=body, headers=headers)
        assert first.status == 200
        frames = [json.loads(line[6:]) for line in (await first.text()).splitlines() if line.startswith('data: ')]
        original = frames[-1]['response']
        final_ids = {item['id'] for item in original['output']}
        live_ids = {frame['item']['id'] for frame in frames if frame['type'] == 'response.output_item.added'}
        assert live_ids <= final_ids
        api._model_name = 'changed-after-acceptance'
        store = api._current_response_store()
        store._max_size = 1
        store.put('newer-unrelated-response', {'response': {}, 'conversation_history': []})
        assert store.get(original['id']) is None
        retry = await client.post('/v1/responses', json={**body, 'stream': False}, headers=headers)
        assert retry.status == 200
        repeated = await retry.json()
    assert calls == ['work']
    assert repeated == original


@pytest.mark.asyncio
async def test_failed_terminal_retry_after_eviction_keeps_envelope(api, owner):
    calls = []
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': '', 'failed': True, 'error': 'provider unavailable', 'messages': []}
        return ''
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers = {'Idempotency-Key': 'failed-eviction'}
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/responses', json={'input': 'work'}, headers=headers)
        original = await response.json()
        store = api._current_response_store()
        store._max_size = 1
        store.put('unrelated', {'response': {}})
        retry = await client.post('/v1/responses', json={'input': 'work', 'stream': True}, headers=headers)
        frames = [json.loads(line[6:]) for line in (await retry.text()).splitlines() if line.startswith('data: ')]
    assert calls == ['work']
    assert frames[-1]['response'] == original


@pytest.mark.asyncio
async def test_retired_terminal_retry_after_eviction_keeps_compact_outcome(api, owner):
    calls = []
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"deleted-sentinel"}'}}]},
            {'role': 'assistant', 'content': 'answer'}]}
        return 'answer'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    headers = {'Idempotency-Key': 'retired-eviction'}
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/responses', json={'input': 'work'}, headers=headers)
        original = await response.json()
        assert owner.db.delete_session(response.headers['X-Hermes-Session-Id'])
        store = api._current_response_store()
        store._max_size = 1
        store.put('unrelated', {'response': {}})
        repeated = await client.post('/v1/responses', json={'input': 'work', 'stream': True}, headers=headers)
        assert repeated.status == 200
        frames = [json.loads(line[6:]) for line in (await repeated.text()).splitlines() if line.startswith('data: ')]
        value = frames[-1]['response']
        assert all('output_index' not in frame for frame in frames)
    assert calls == ['work']
    assert value['id'] == original['id'] and value['status'] == original['status']
    assert 'deleted-sentinel' not in str(value)
    assert value['output'][-1]['content'][0]['text'] == 'answer'


@pytest.mark.asyncio
async def test_terminal_replay_does_not_roll_named_conversation_back(api, owner):
    calls = []
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': event.text, 'messages': []}
        return event.text
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    first_headers = {'Idempotency-Key': 'first-conversation-turn'}
    body = {'input': 'first', 'conversation': 'shared'}
    async with TestClient(TestServer(app)) as client:
        first = await client.post('/v1/responses', json=body, headers=first_headers)
        original = await first.json()
        later = await client.post('/v1/responses', json={'input': 'later', 'conversation': 'shared'},
                                  headers={'Idempotency-Key': 'later-conversation-turn'})
        latest = await later.json()
        store = api._current_response_store()
        store._max_size = 4
        for index in range(3):
            store.get(latest['id'])
            store.put('cache-pressure-' + str(index), {'response': {}})
        assert store.get(original['id']) is None
        assert store.get_conversation('shared') == latest['id']
        store._max_size = 100
        repeated = await client.post('/v1/responses', json=body, headers=first_headers)
        assert repeated.status == 200
        await repeated.read()
        assert store.get_conversation('shared') == latest['id']
    assert calls == ['first', 'later']

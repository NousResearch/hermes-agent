"""Public speech observes the canonical admission without changing its private receipt.

Only the model/turn producer is synthetic; HTTP, authority, persistence and SSE are real.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server_public_speech as speech
from tests.gateway.test_api_server import _create_app
from tests.gateway.test_responses_public_speech import _events, _messages, _read_until


PUBLIC = {'input': 'question', 'stream': True, 'public_commentary': True}
HEADERS = {'Idempotency-Key': 'speech-turn', 'Authorization': 'Bearer fixture-key'}
PRIVATE = 'PRIVATE_PROTOCOL_CANARY'


def _producer(owner, calls, *, release=None, started=None, result=None):
    async def handle(event):
        from gateway.session_api_turn import publish_api_event, publish_api_tool_event, wire_api_observers
        from gateway.session_results import execution_result
        calls.append(event.text)
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        agent = SimpleNamespace(status_callback=None, interim_assistant_callback=None, tool_progress_callback=None)
        wire_api_observers(agent, (owner, sid, generation), True)
        publish_api_tool_event(owner, sid, generation, 'tool.start', PRIVATE, PRIVATE, {'arg': PRIVATE})
        publish_api_event(owner, sid, 'message.delta', {'text': 'unclassified prefix'})
        agent.reasoning_callback(PRIVATE)
        agent.status_callback('lifecycle', PRIVATE)
        agent.interim_assistant_callback('Safe progress.', already_streamed=True)
        publish_api_tool_event(owner, sid, generation, 'tool.complete', PRIVATE, PRIVATE, {}, PRIVATE)
        if started is not None:
            started.set()
        if release is not None:
            await release.wait()
        execution_result.get()['result'] = result or {
            'final_response': 'Authoritative answer.', 'completed': True,
            'messages': [
                {'role': 'assistant', 'content': 'Safe progress.', 'tool_calls': [
                    {'id': PRIVATE, 'type': 'function', 'function': {'name': PRIVATE, 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': PRIVATE, 'content': PRIVATE},
                {'role': 'assistant', 'content': 'Authoritative answer.'}]}
        return execution_result.get()['result']['final_response']
    owner.runner._handle_message = handle


@pytest.mark.asyncio
async def test_public_stream_observes_authority_callbacks_and_replays_exact_public_frames(api, owner, monkeypatch):
    calls, started, release = [], asyncio.Event(), asyncio.Event()
    api._api_key = 'fixture-key'
    _producer(owner, calls, release=release, started=started)
    bypass = AsyncMock(return_value=({'failed': True, 'completed': False}, {}))
    monkeypatch.setattr(api, '_run_agent', bypass)
    async with TestClient(TestServer(_create_app(api))) as client:
        denied = await client.post('/v1/responses', json=PUBLIC)
        assert denied.status == 401
        response = await client.post('/v1/responses', json=PUBLIC, headers=HEADERS)
        try:
            early = await asyncio.wait_for(_read_until(response, 'response.output_item.done'), 3)
            assert started.is_set()
            created = _events(early)[0][1]['response']
            from gateway.platforms.api_server_response_identity import durable_response_id
            scope = api._run_idempotency_scope(SimpleNamespace(headers=HEADERS))
            assert created['id'] == durable_response_id(f'idem:{scope}:speech-turn')
            snapshot = await (await client.get('/v1/responses/' + created['id'], headers=HEADERS)).json()
            assert snapshot['status'] == 'in_progress' and snapshot['output'] == []
            assert PRIVATE not in early.decode() and 'unclassified prefix' not in early.decode()
        finally:
            release.set()
        wire = early + await response.read()
        frames = _events(wire)
        terminal = frames[-1][1]['response']
        assert frames[-1][0] == 'response.completed'
        assert _messages(terminal) == [('commentary', 'Safe progress.'), (None, 'Authoritative answer.')]
        assert PRIVATE not in wire.decode()
        retry = await client.post('/v1/responses', json=PUBLIC, headers=HEADERS)
        assert await retry.read() == wire
        stored = await (await client.get('/v1/responses/' + terminal['id'], headers=HEADERS)).json()
        assert stored == terminal
        history = api._current_response_store().get(terminal['id'])['conversation_history']
        assert PRIVATE in repr(history)
        with owner.db._read_ctx() as conn:
            admission_id = conn.execute('SELECT admission_id FROM session_admissions').fetchone()[0]
        from gateway.session_results import admission_result
        assert PRIVATE in repr(admission_result(owner.db, admission_id))
    assert calls == ['question'] and bypass.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('first_public', [False, True], ids=['generic-first', 'public-first'])
async def test_projection_mode_change_conflicts_before_replay_or_admission(api, owner, monkeypatch, first_public):
    calls = []
    _producer(owner, calls)
    monkeypatch.setattr(api, '_run_agent', AsyncMock(return_value=({'final_response': 'fallback'}, {})))
    body = {**PUBLIC, 'public_commentary': first_public}
    async with TestClient(TestServer(_create_app(api))) as client:
        first = await client.post('/v1/responses', json=body, headers=HEADERS)
        assert first.status == 200
        await first.read()
        changed = await client.post('/v1/responses', json={**body, 'public_commentary': not first_public}, headers=HEADERS)
        await changed.read()
        assert changed.status == 409
    with owner.db._read_ctx() as conn:
        assert conn.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 1


@pytest.mark.asyncio
async def test_terminal_retry_after_public_disconnect_keeps_one_turn_and_safe_identity(api, owner, monkeypatch):
    calls = []
    _producer(owner, calls)
    monkeypatch.setattr(api, '_run_agent', AsyncMock(return_value=({'final_response': 'fallback'}, {})))
    original = web.StreamResponse.write
    broken = False

    async def disconnect(response, data):
        nonlocal broken
        if b'event: response.completed\n' in data and not broken:
            broken = True
            raise ConnectionResetError('synthetic disconnect')
        return await original(response, data)

    monkeypatch.setattr(web.StreamResponse, 'write', disconnect)
    async with TestClient(TestServer(_create_app(api))) as client:
        first = await client.post('/v1/responses', json=PUBLIC, headers=HEADERS)
        first_frames = _events(await first.read())
        assert all(name != 'response.completed' for name, _ in first_frames)
        retry = await client.post('/v1/responses', json=PUBLIC, headers=HEADERS)
        frames = _events(await retry.read())
        terminal = frames[-1][1]['response']
        assert frames[-1][0] == 'response.completed'
        assert first_frames[0][1]['response']['id'] == terminal['id']
        assert terminal['hermes']['session_id'] == retry.headers['X-Hermes-Session-Id']
        assert PRIVATE not in json.dumps(frames)
        assert _messages(terminal)[-1] == (None, 'Authoritative answer.')
        continuation = await client.post('/v1/responses', json={**PUBLIC, 'input': 'next',
            'previous_response_id': terminal['id']}, headers={'Idempotency-Key': 'next-speech'})
        assert continuation.status == 200
        await continuation.read()
        assert continuation.headers['X-Hermes-Session-Id'] == terminal['hermes']['session_id']
    assert calls == ['question', 'next']


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['failed', 'incomplete'])
async def test_public_canonical_unsuccessful_receipt_replays_without_diagnostics(api, owner, monkeypatch, status):
    calls = []
    result = {'final_response': PRIVATE, 'completed': False, 'failed': status == 'failed',
              'messages': [], 'error': PRIVATE if status == 'failed' else None}
    _producer(owner, calls, result=result)
    monkeypatch.setattr(api, '_run_agent', AsyncMock(return_value=(result, {})))
    async with TestClient(TestServer(_create_app(api))) as client:
        first = await client.post('/v1/responses', json=PUBLIC, headers=HEADERS)
        wire = await first.read()
        terminal = _events(wire)[-1][1]['response']
        assert terminal['status'] == status and terminal['hermes']['partial'] is True
        assert PRIVATE not in wire.decode()
        retry = await client.post('/v1/responses', json=PUBLIC, headers=HEADERS)
        assert await retry.read() == wire
    assert calls == ['question']

"""Reviewer-owned SPEC: only the external turn and IO faults are synthetic.

HTTP/SSE/GET, canonical admission, result retention, history builder, stores and
continuation parsing are real. No candidate test adapters are imported.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

PRIVATE = 'SPEC_PRIVATE_TOOL_CONTEXT_2940'
PUBLIC = {'input': 'question', 'stream': True, 'public_commentary': True}
KEY = {'Idempotency-Key': 'independent-spec'}


def frames(wire):
    out = []
    for chunk in wire.decode().split('\n\n'):
        event = next((s[7:] for s in chunk.splitlines() if s.startswith('event: ')), None)
        data = next((s[6:] for s in chunk.splitlines() if s.startswith('data: ')), None)
        if event and data:
            out.append((event, json.loads(data)))
    return out


def app_for(api):
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_get('/v1/responses/{response_id}', api._handle_get_response)
    return app


def producer(owner, seen, status, *, release=None, commentary=False):
    async def handle(event):
        from gateway.session_api_turn import api_execution, wire_api_observers
        from gateway.session_results import execution_result
        seen.append({'input': event.text, 'history': api_execution.get()['history']})
        if commentary:
            sid = event.source.chat_id
            generation = owner.sessions[sid].event_stream.execution['execution_generation']
            agent = SimpleNamespace(status_callback=None, interim_assistant_callback=None,
                                    tool_progress_callback=None)
            wire_api_observers(agent, (owner, sid, generation), True)
            agent.interim_assistant_callback('Safe progress.')
        if release:
            await release.wait()
        result = {'final_response': 'Safe answer.' if status == 'completed' else 'PRIVATE_DIAGNOSTIC',
                  'completed': status == 'completed', 'messages': [
                      {'role': 'assistant', 'content': '', 'tool_calls': [
                          {'id': PRIVATE, 'type': 'function', 'function': {
                              'name': 'external_tool', 'arguments': '{"key":"' + PRIVATE + '"}'}}]},
                      {'role': 'tool', 'tool_call_id': PRIVATE, 'content': PRIVATE}]}
        outcomes = {
            'completed': {'messages': [*result['messages'], {'role': 'assistant', 'content': 'Safe answer.'}]},
            'failed': {'failed': True, 'error': 'PRIVATE_DIAGNOSTIC'},
            'incomplete': {'partial': True},
            'interrupted': {'interrupted': True},
        }
        result.update(outcomes[status])
        execution_result.get()['result'] = result
        return result['final_response']
    owner.runner._handle_message = handle


@pytest.mark.asyncio
@pytest.mark.parametrize('public,status', [(False, 'failed'), (True, 'completed'),
    (True, 'failed'), (True, 'incomplete'), (True, 'interrupted')],
    ids=['generic-failed', 'public-completed', 'public-failed', 'public-incomplete', 'public-interrupted'])
async def test_private_history_preservation(api, owner, public, status):
    from gateway.session_results import admission_result
    seen = []
    producer(owner, seen, status)
    body = {**PUBLIC, 'public_commentary': public}
    async with TestClient(TestServer(app_for(api))) as client:
        response = await client.post('/v1/responses', json=body, headers=KEY)
        assert response.status == 200
        wire = await response.read()
        terminal = frames(wire)[-1][1]['response']
        stored = api._current_response_store().get(terminal['id'])
        retrieved = await (await client.get('/v1/responses/' + terminal['id'])).json()
        admission_id = owner.db._read_all('SELECT admission_id FROM session_admissions')[0]['admission_id']
        receipt = admission_result(owner.db, admission_id)
        following = await client.post('/v1/responses', json={**body, 'input': 'next',
            'previous_response_id': terminal['id']}, headers={'Idempotency-Key': 'independent-next'})
        assert following.status == 200
        await following.read()
    if public:
        assert PRIVATE not in wire.decode()
        assert PRIVATE not in json.dumps(retrieved)
        if status != 'completed':
            assert 'PRIVATE_DIAGNOSTIC' not in wire.decode()
        assert retrieved == terminal
    assert PRIVATE in repr(receipt), 'positive witness: canonical receipt retains executed tool context'
    assert PRIVATE in repr(stored['conversation_history']), 'private stored history lost executed tool context'
    assert PRIVATE in repr(seen[1]['history']), 'real next continuation lost executed tool context'


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', [False, True], ids=['healthy-store', 'replay-cache-fault'])
async def test_joined_publisher_snapshot_on_late_store_failure(api, owner, monkeypatch, fault):
    seen, release = [], asyncio.Event()
    producer(owner, seen, 'completed', release=release, commentary=True)
    original_write = web.StreamResponse.write
    store, original_put = api._current_response_store(), api._current_response_store().put
    joined_at_terminal, allow_joined = asyncio.Event(), asyncio.Event()
    responses, failed_replay = [], []

    async def hold(response, data):
        if b'event: response.created\n' in data:
            responses.append(response)
        if b'event: response.completed\n' in data and len(responses) == 2 and response is responses[1]:
            joined_at_terminal.set()
            await allow_joined.wait()
        return await original_write(response, data)

    def fail_replay(key, data):
        if fault and key.startswith('idem:'):
            failed_replay.append(key)
            raise RuntimeError('synthetic derivative replay store fault')
        return original_put(key, data)

    monkeypatch.setattr(web.StreamResponse, 'write', hold)
    monkeypatch.setattr(store, 'put', fail_replay)
    async with TestClient(TestServer(app_for(api))) as client:
        first = await client.post('/v1/responses', json=PUBLIC, headers=KEY)
        early = bytearray()
        async with asyncio.timeout(5):
            while not any(e == 'response.output_item.done' for e, _ in frames(bytes(early))):
                early.extend(await first.content.readline())
        joined = await client.post('/v1/responses', json=PUBLIC, headers=KEY)
        joined_created = await joined.content.readuntil(b'\n\n')
        release.set()
        try:
            await asyncio.wait_for(joined_at_terminal.wait(), 5)
            first_wire = bytes(early) + await first.read()
            original = frames(first_wire)[-1][1]['response']
            before = await (await client.get('/v1/responses/' + original['id'])).json()
            assert before == original
        finally:
            allow_joined.set()
        joined_wire = joined_created + await joined.read()
        after = await (await client.get('/v1/responses/' + original['id'])).json()
    terminal_types = {'response.completed', 'response.failed', 'response.incomplete'}
    assert [kind for kind, _ in frames(first_wire) if kind in terminal_types] == ['response.completed']
    joined_env = frames(joined_wire)[-1][1]['response']
    assert [kind for kind, _ in frames(joined_wire) if kind in terminal_types] == ['response.completed']
    assert joined_env['output'][-1]['id'] == original['output'][-1]['id']
    assert [item.get('phase') for item in original['output']] == ['commentary', None]
    assert [item.get('phase') for item in joined_env['output']] == [None]
    assert len(seen) == 1
    assert bool(failed_replay) is fault
    assert after == original, 'late observer overwrote publisher GET snapshot after replay-cache fault'


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['final-delta', 'terminal', 'preflight-store'])
async def test_result_tool_history_survives_writer_recovery(api, owner, monkeypatch, fault):
    seen = []
    producer(owner, seen, 'completed')
    original_write = web.StreamResponse.write
    store = api._current_response_store()
    original_put = store.put
    injected = []

    async def disconnect(response, data):
        marker = b'event: response.output_text.delta\n' if fault == 'final-delta' else b'event: response.completed\n'
        if fault != 'preflight-store' and marker in data and not injected:
            injected.append(fault)
            raise ConnectionResetError('synthetic transport fault after result')
        return await original_write(response, data)

    def reject_preflight(key, data):
        if (fault == 'preflight-store' and not injected
                and data.get('response', {}).get('status') == 'in_progress'
                and seen):
            injected.append(fault)
            raise RuntimeError('synthetic preflight store fault after result')
        return original_put(key, data)

    monkeypatch.setattr(web.StreamResponse, 'write', disconnect)
    monkeypatch.setattr(store, 'put', reject_preflight)
    async with TestClient(TestServer(app_for(api))) as client:
        first = await client.post('/v1/responses', json=PUBLIC, headers=KEY)
        wire = await first.read()
        response_id = frames(wire)[0][1]['response']['id']
        saved = store.get(response_id)
        retrieved = await (await client.get('/v1/responses/' + response_id)).json()
        following = await client.post('/v1/responses', json={**PUBLIC, 'input': 'next',
            'previous_response_id': response_id}, headers={'Idempotency-Key': 'recovery-next'})
        assert following.status == 200
        await following.read()
    assert injected == [fault]
    assert saved['response']['status'] == ('failed' if fault == 'preflight-store' else 'incomplete')
    assert saved['session_id'] == first.headers['X-Hermes-Session-Id']
    assert PRIVATE not in wire.decode() and PRIVATE not in json.dumps(retrieved)
    assert PRIVATE in repr(saved['conversation_history']), 'writer recovery lost available result tool context'
    assert PRIVATE in repr(seen[1]['history']), 'continuation after writer recovery lost tool context'

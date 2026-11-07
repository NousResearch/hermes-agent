"""Flagged failure finals are diagnostics, not Responses answer deltas (Jordan #63894).

These are real HTTP/SSE and retrieval requests over the canonical admission path;
only the agent turn is synthetic. Private continuation history is not public output.
"""
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

_SAFE = 'Useful partial answer.'
_DIAGNOSTIC = 'Synthetic provider diagnostic, not an answer.'
_ERROR = 'Provider unavailable: Authorization: Bearer synthetic-offline-credential-0123456789'
_USAGE = {'input_tokens': 13, 'output_tokens': 5, 'total_tokens': 18}
_TOOL_CALL = {'id': 'call_probe', 'type': 'function',
              'function': {'name': 'read_file', 'arguments': '{"path": "fixture.txt"}'}}


def _frames(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]


def _answer(envelope):
    return ''.join(part['text'] for item in envelope['output'] if item['type'] == 'message'
                   for part in item['content'] if part['type'] == 'output_text')


def _app(api, owner, *, flags, mode, final):
    seen = []

    async def handle(event):
        from gateway.session_api_turn import api_execution, publish_api_event, publish_api_tool_event
        from gateway.session_results import execution_result
        history = api_execution.get()['history'] or []
        seen.append(history)
        if event.text == 'continue':
            execution_result.get().update(result={'final_response': 'Continued.', 'messages': []}, usage=_USAGE)
            return 'Continued.'
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        if mode != 'silent':
            publish_api_event(owner, sid, 'message.delta', {'text': _SAFE})
        # The tool also flushes the preceding batched delta, before the failure settles.
        publish_api_tool_event(owner, sid, generation, 'tool.start', 'call_probe',
                               'read_file', {'path': 'fixture.txt'})
        publish_api_tool_event(owner, sid, generation, 'tool.complete', 'call_probe',
                               'read_file', {'path': 'fixture.txt'}, 'fixture contents')
        result = {**flags, 'final_response': final, 'messages': [
            *history, {'role': 'user', 'content': event.text},
            {'role': 'assistant', 'content': '', 'tool_calls': [_TOOL_CALL]},
            {'role': 'tool', 'tool_call_id': 'call_probe', 'content': 'fixture contents'},
            {'role': 'assistant', 'content': final}]}
        if mode in ('append', 'rewrite'):
            result.update(response_transformed=True, pre_transform_response=_SAFE)
        execution_result.get().update(result=result, usage=_USAGE)
        return final

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    app.router.add_get('/v1/responses/{response_id}', api._handle_get_response)
    return app, seen


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['flag', 'error'])
@pytest.mark.parametrize('mode', ['silent', 'delta', 'append', 'rewrite'])
@pytest.mark.parametrize('projection', ['unkeyed', 'keyed', 'recovered'])
async def test_failed_final_stays_out_of_answer_text(api, owner, failure, mode, projection):
    flags = {'failed': True} if failure == 'flag' else {'completed': False, 'error': _ERROR}
    final = _SAFE + _DIAGNOSTIC if mode == 'append' else _DIAGNOSTIC
    app, seen = _app(api, owner, flags=flags, mode=mode, final=final)
    body = {'input': 'probe', 'stream': True}
    headers = {} if projection == 'unkeyed' else {'Idempotency-Key': 'failure-probe'}
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/responses', json=body, headers=headers)
        assert response.status == 200
        frames = _frames(await response.text())
        terminal = frames[-1]['response']
        # Check the wire before looking at the terminal: no new diagnostic delta,
        # including a transformed append, may be promoted from the failed final.
        deltas = ''.join(f['delta'] for f in frames if f['type'] == 'response.output_text.delta')
        expected = '' if mode == 'silent' else _SAFE
        assert bool(deltas == expected), 'failure final changed the delivered answer deltas'
        done = [f['text'] for f in frames if f['type'] == 'response.output_text.done']
        assert bool(done == ([expected] if expected else [])), 'failure final replaced the text done event'
        assert frames[-1]['type'] == 'response.failed'
        assert terminal['status'] == 'failed'
        assert bool(_answer(terminal) == expected), 'failure final entered the terminal answer'
        assert _DIAGNOSTIC not in json.dumps(frames), 'diagnostic entered a public SSE frame'
        assert 'synthetic-offline-credential' not in json.dumps(frames), 'error was not redacted'
        assert terminal['usage'] == _USAGE
        assert [item['type'] for item in terminal['output']].count('function_call') == 1
        assert [item['type'] for item in terminal['output']].count('function_call_output') == 1
        expected_error = ('Provider unavailable: Authorization: Bearer ***' if failure == 'error'
                          else 'The admitted turn failed.')
        error_kind = {'code': 'agent_error'} if headers else {'type': 'server_error'}
        assert terminal['error'] == {**error_kind, 'message': expected_error}
        call = next(item for item in terminal['output'] if item['type'] == 'function_call')
        tool = next(item for item in terminal['output'] if item['type'] == 'function_call_output')
        assert call['call_id'] == tool['call_id'] == 'call_probe'
        assert call['name'] == 'read_file'
        assert json.loads(call['arguments']) == {'path': 'fixture.txt'}
        assert tool['output'] in ('fixture contents', [{'type': 'input_text', 'text': 'fixture contents'}])
        fetched = await client.get('/v1/responses/' + terminal['id'])
        assert fetched.status == 200
        assert await fetched.json() == terminal
        private = api._current_response_store().get(terminal['id'])['conversation_history']
        assert any(m.get('tool_calls') == [_TOOL_CALL] for m in private)
        assert any(m.get('role') == 'tool' and m['content'] == 'fixture contents' for m in private)
        assert any(m.get('content') == final for m in private), 'private transcript lost the failed turn'
        if headers:
            replay = await client.post('/v1/responses', json=body, headers=headers)
            assert _frames(await replay.text()) == frames
        if projection == 'recovered':
            # Evict derivative bodies, not the accepted request identity or private receipt.
            store = api._current_response_store()
            store._max_size = 1
            store.put('eviction-probe', {'response': {}})
            recovered = await client.post('/v1/responses', json=body, headers=headers)
            recovered_frames = _frames(await recovered.text())
            recovered_terminal = recovered_frames[-1]['response']
            assert recovered_terminal['status'] == 'failed'
            assert recovered_terminal['usage'] == _USAGE
            assert _DIAGNOSTIC not in json.dumps(recovered_frames), 'receipt replay promoted a diagnostic'
            assert not any(f['type'] == 'response.output_text.delta' for f in recovered_frames)
            assert bool(_answer(recovered_terminal) == ''), 'receipt replay invented an answer'
            assert (await (await client.get('/v1/responses/' + terminal['id'])).json()) == recovered_terminal
        continued = await client.post('/v1/responses', json={
            'input': 'continue', 'previous_response_id': terminal['id'], 'stream': True})
        assert continued.status == 200
        assert _frames(await continued.text())[-1]['type'] == 'response.completed'
    assert len(seen) == 2, 'an idempotent retry re-executed the agent'
    assert any(m.get('tool_calls') == [_TOOL_CALL] for m in seen[1])
    assert any(m.get('role') == 'tool' and m['content'] == 'fixture contents' for m in seen[1])


@pytest.mark.asyncio
@pytest.mark.parametrize('status,flags', [
    ('completed', {'completed': True}), ('incomplete', {'completed': False}),
    ('cancelled', {'completed': False, 'interrupted': True}),
    ('cancelled', {'completed': False, 'interrupted': True, 'error': 'Interrupted turn.'})],
    ids=['success', 'incomplete', 'cancelled', 'cancelled-with-error'])
@pytest.mark.parametrize('mode', ['silent', 'append', 'rewrite'])
@pytest.mark.parametrize('keyed', [False, True], ids=['unkeyed', 'keyed'])
async def test_nonfailure_finals_keep_existing_outcomes(api, owner, status, flags, mode, keyed):
    final = _SAFE + ' Verified.' if mode == 'append' else 'Authoritative answer.'
    app, seen = _app(api, owner, flags=flags, mode=mode, final=final)
    headers = {'Idempotency-Key': 'control-probe'} if keyed else {}
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/responses', json={'input': 'probe', 'stream': True}, headers=headers)
        assert response.status == 200
        frames = _frames(await response.text())
        terminal = frames[-1]['response']
        assert frames[-1]['type'] == 'response.' + status
        assert terminal['status'] == status
        assert 'error' not in terminal
        assert _answer(terminal) == final
        deltas = ''.join(f['delta'] for f in frames if f['type'] == 'response.output_text.delta')
        assert deltas == (_SAFE if mode == 'rewrite' else final)
        assert terminal['usage'] == _USAGE
        assert await (await client.get('/v1/responses/' + terminal['id'])).json() == terminal
    assert len(seen) == 1

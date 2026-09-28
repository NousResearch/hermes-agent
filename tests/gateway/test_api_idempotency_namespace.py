"""Canonical OpenAI-compat idempotency is scoped to the authenticated key namespace."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_cutover_contract import api  # noqa: F401
from tests.gateway.test_api_source_binding import owner  # noqa: F401

OLD, NEW = 'probe-old-key-long-enough', 'probe-new-key-long-enough'


def _execute(owner, calls):
    async def handle(event):
        from gateway.session_results import execution_result
        calls.append(event.text)
        text = f'execution-{len(calls)}'
        execution_result.get()['result'] = {'final_response': text, 'messages': []}
        return text
    owner.runner._handle_message = handle


async def _rotated(api, client, path, body, key):
    outputs = []
    for api_key in (OLD, NEW):
        api._api_key = api_key
        resp = await client.post(path, json=body, headers={'Authorization': 'Bearer ' + api_key, 'Idempotency-Key': key})
        assert resp.status == 200, await resp.text()
        outputs.append(await resp.json())
    return outputs


@pytest.mark.asyncio
async def test_rotated_api_key_is_a_different_principal_for_chat_and_responses(api, owner):
    calls = []
    _execute(owner, calls)
    app = web.Application()
    app.router.add_post('/v1/chat/completions', api._handle_chat_completions)
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        chat = await _rotated(api, client, '/v1/chat/completions',
                              {'messages': [{'role': 'user', 'content': 'same request'}]}, 'rotate-chat')
        assert [c['choices'][0]['message']['content'] for c in chat] == ['execution-1', 'execution-2']
        responses = await _rotated(api, client, '/v1/responses', {'input': 'fresh rotation'}, 'rotate-responses')
        assert [r['output'][-1]['content'][0]['text'] for r in responses] == ['execution-3', 'execution-4']
    assert len(calls) == 4

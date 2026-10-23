"""A chained Responses turn replays the earlier turn's user row as the model saw it, including
its ``api_content`` (memory prefetch, plugin context), not a synthetic row rebuilt from the raw
request text."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
@pytest.mark.parametrize('stream', [False, True], ids=['json', 'sse'])
async def test_previous_response_chain_keeps_turn_user_api_content(api, owner, stream):
    seen = []

    async def handle(event):
        from gateway.session_api_turn import api_execution
        from gateway.session_results import execution_result
        history = api_execution.get()['history']
        seen.append(history)
        user = {'role': 'user', 'content': event.text}
        if event.text == 'first':
            user['api_content'] = 'first\n\n<user_memory>project is ALPHA</user_memory>'
        execution_result.get()['result'] = {'final_response': 'a', 'messages': [
            *history, user, {'role': 'assistant', 'content': 'a'}]}
        return 'a'
    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    async with TestClient(TestServer(app)) as client:
        first = await client.post('/v1/responses', json={'input': 'first'})
        rid = (await first.json())['id']
        second = await client.post('/v1/responses', json={
            'input': 'second', 'previous_response_id': rid, 'stream': stream})
        assert second.status == 200, await second.text()
        await second.text()
    users = [m for m in seen[1] if m['role'] == 'user']
    assert users[0]['content'] == 'first'
    assert users[0].get('api_content', '').endswith('ALPHA</user_memory>'), seen[1]
    assert [m['role'] for m in seen[1]] == ['user', 'assistant']

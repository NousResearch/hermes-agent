"""A chat completion admission carries the caller's run owner scope durably."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_chat_completion_admission_carries_durable_run_owner(api, owner):
    from gateway.session_api_turn import owns_api_run
    async def handle(event):
        from gateway.session_results import execution_result
        execution_result.get()['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'
    owner.runner._handle_message = handle
    api._api_key = 'run-owner-key-long-enough'
    app = web.Application()
    app.router.add_post('/v1/chat/completions', api._handle_chat_completions)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post('/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'hi'}]},
                                 headers={'Authorization': 'Bearer ' + api._api_key, 'Idempotency-Key': 'k1'})
        assert resp.status == 200, await resp.text()
    with owner.db._read_ctx() as conn:
        rid = conn.execute("SELECT request_id FROM session_admissions WHERE principal_id='api'").fetchone()[0]
    scope = rid.split(':')[1]
    assert owns_api_run(api, rid, scope)
    assert not owns_api_run(api, rid, 'b' * 64)

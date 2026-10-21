"""Only the publishing Responses stream writes the durable ``idem:`` replay record."""
import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.waiting import async_wait_until


def _frames(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: ')]


@pytest.mark.asyncio
async def test_in_flight_retry_does_not_win_the_replay_record_without_deltas(api, owner, monkeypatch):
    release = asyncio.Event()
    calls = []

    async def handle(event):
        from gateway.session_api_turn import publish_api_event
        from gateway.session_results import execution_result
        calls.append(event.text)
        await release.wait()
        publish_api_event(owner, event.source.chat_id, 'message.delta', {'text': 'answer'})
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': [
            {'role': 'assistant', 'content': 'answer'}]}
        return 'answer'
    owner.runner._handle_message = handle
    from gateway.platforms import api_server_openai_routes as routes
    emit, retried = routes._ResponsesStream.emit_completed, asyncio.Event()

    async def lagging_publisher(stream):
        # The original client lags: the joined retry reaches its terminal write first.
        if stream.terminal_replay:
            try:
                return await emit(stream)
            finally:
                retried.set()
        await asyncio.wait_for(retried.wait(), 10)
        return await emit(stream)
    monkeypatch.setattr(routes._ResponsesStream, 'emit_completed', lagging_publisher)
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body, headers = {'input': 'work', 'stream': True}, {'Idempotency-Key': 'lagging-publisher'}
    async with TestClient(TestServer(app)) as client:
        original = asyncio.create_task(client.post('/v1/responses', json=body, headers=headers))
        await async_wait_until(lambda: calls, timeout=10, what='the original turn to start')
        retry = asyncio.create_task(client.post('/v1/responses', json=body, headers=headers))
        # The retry joins the in-flight admission as a second observer beside the original.
        await async_wait_until(lambda: any(len(o) >= 2 for o in getattr(owner, 'api_observers', {}).values()),
                               timeout=10, what='the retry to join the in-flight admission as an observer')
        release.set()
        first, joined = await original, await retry
        assert first.status == joined.status == 200
        await first.text(), await joined.text()
        later = await client.post('/v1/responses', json=body, headers=headers)
        replayed = _frames(await later.text())
    assert calls == ['work']
    assert any(f['type'] == 'response.output_text.delta' for f in replayed), [f['type'] for f in replayed]

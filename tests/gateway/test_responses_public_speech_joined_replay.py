"""A joined canonical public observer does not invent missing commentary identities."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server import _create_app
from tests.gateway.test_responses_public_speech import _events, _messages, _read_until
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_joined_public_retry_defers_to_terminal_without_relabeling_late_commentary(api, owner, monkeypatch):
    calls, release = [], asyncio.Event()

    async def handle(event):
        from gateway.session_api_turn import wire_api_observers
        from gateway.session_results import execution_result
        calls.append(event.text)
        sid = event.source.chat_id
        generation = owner.sessions[sid].event_stream.execution['execution_generation']
        agent = SimpleNamespace(status_callback=None, interim_assistant_callback=None, tool_progress_callback=None)
        wire_api_observers(agent, (owner, sid, generation), True)
        agent.interim_assistant_callback('First progress.')
        await release.wait()
        agent.interim_assistant_callback('Second progress.')
        execution_result.get()['result'] = {'final_response': 'Final answer.', 'messages': [
            {'role': 'assistant', 'content': 'Final answer.'}]}
        return 'Final answer.'

    owner.runner._handle_message = handle
    monkeypatch.setattr(api, '_run_agent', AsyncMock(return_value=({'failed': True}, {})))
    body = {'input': 'work', 'stream': True, 'public_commentary': True}
    headers = {'Idempotency-Key': 'joined-public'}
    async with TestClient(TestServer(_create_app(api))) as client:
        original = await client.post('/v1/responses', json=body, headers=headers)
        early = await asyncio.wait_for(_read_until(original, 'response.output_item.done'), 5)
        joined = await client.post('/v1/responses', json=body, headers=headers)
        try:
            await async_wait_until(lambda: any(len(observers) == 2 for observers in owner.api_observers.values()),
                                   timeout=5, what='both observers registered')
        finally:
            release.set()
        first_wire = early + await original.read()
        joined_frames = _events(await joined.read())
        original_frames = _events(first_wire)
        first_env = original_frames[-1][1]['response']
        joined_env = joined_frames[-1][1]['response']
        # A late observer missed First; assigning its id to Second would corrupt
        # stable public item identity. It observes only the authoritative final.
        assert _messages(joined_env) == [(None, 'Final answer.')]
        assert joined_env['output'][-1]['id'] == first_env['output'][-1]['id']
        assert _messages(first_env) == [('commentary', 'First progress.'),
                                       ('commentary', 'Second progress.'), (None, 'Final answer.')]
        replay = await client.post('/v1/responses', json=body, headers=headers)
        assert await replay.read() == first_wire
        snapshot = api._current_response_store().get(first_env['id'])['response']
        assert snapshot == first_env
    assert calls == ['work']

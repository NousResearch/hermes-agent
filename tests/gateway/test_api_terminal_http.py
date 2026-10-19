"""Transport projections never report a non-successful result as completion."""
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize('flag,status', [('failed', 'failed'), ('interrupted', 'cancelled')])
@pytest.mark.parametrize('stream', [False, True])
async def test_responses_terminal_result_is_not_success(flag, status, stream):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._run_agent = AsyncMock(return_value=({'final_response': 'diagnostic', flag: True,
                                                 'completed': False, 'messages': []}, {}))
    app = web.Application()
    app.router.add_post('/v1/responses', adapter._handle_responses)
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post('/v1/responses', json={'input': 'hello', 'stream': stream})
            assert response.status == 200
            if stream:
                text = await response.text()
                assert f'event: response.{status}\n' in text
                assert 'event: response.completed\n' not in text
            else:
                assert (await response.json())['status'] == status
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()

def test_chat_interruption_is_not_stop():
    from gateway.platforms.api_server_openai_routes import _result_flags, _finish_reason
    assert _finish_reason(*_result_flags({'completed': False, 'interrupted': True})) != 'stop'

@pytest.mark.asyncio
@pytest.mark.parametrize('flag,status', [('failed', 'failed'), ('interrupted', 'cancelled')])
async def test_session_chat_terminal_result_is_not_success(flag, status):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._prepare_session_chat = AsyncMock(return_value=({'gateway_session_key': None,
        'session_id': 'owned', 'run_kwargs': {}, 'runtime_request': {}, 'user_message': 'hello',
        'lock_active': False, 'body': {}}, None))
    adapter._conversation_history_for_session = AsyncMock(return_value=[])
    adapter._run_agent = AsyncMock(return_value=({'final_response': 'diagnostic', flag: True,
                                                 'completed': False, 'messages': []}, {}))
    app = web.Application()
    app.router.add_post('/chat', adapter._handle_session_chat)
    app.router.add_post('/stream', adapter._handle_session_chat_stream)
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.post('/chat', json={})
            assert (await response.json())['status'] == status
            response = await client.post('/stream', json={})
            text = await response.text()
            assert f'event: run.{status}\n' in text
            assert 'event: run.completed\n' not in text
    finally:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()

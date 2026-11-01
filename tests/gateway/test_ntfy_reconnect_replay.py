"""Reconnect replay and monotonic receipt cursors for the ntfy streaming adapter."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

NtfyAdapter = load_plugin_adapter('ntfy').NtfyAdapter


def _message(identifier, text):
    return {'event': 'message', 'id': identifier, 'topic': 'alerts', 'message': text}


def test_stream_reconnect_replays_gap_without_redelivering_overlap():
    async def exercise():
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={'topic': 'alerts'}))
        adapter._running = True
        adapter.handle_message = AsyncMock()
        requests = []

        def server(request):
            requests.append(request)
            events = [_message('before-gap', 'before')]
            if request.url.params.get('since') == 'before-gap':
                events.append(_message('during-gap', 'missed while disconnected'))
            return httpx.Response(200, content='\n'.join(json.dumps(event) for event in events))

        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            adapter._http_client = client
            await adapter._consume_stream('https://ntfy.example/alerts/json', {})
            await adapter._consume_stream('https://ntfy.example/alerts/json', {})

        assert [call.args[0].text for call in adapter.handle_message.await_args_list] == [
            'before', 'missed while disconnected']
        assert dict(requests[0].url.params) == {'poll': 'false'}
        assert dict(requests[1].url.params) == {'poll': 'false', 'since': 'before-gap'}

    asyncio.run(exercise())


@pytest.mark.parametrize('trailing', [_message('old', 'replayed overlap'), {'event': 'message', 'message': 'no server id'}])
def test_duplicate_or_synthetic_id_does_not_replace_last_server_cursor(trailing):
    async def exercise():
        adapter = NtfyAdapter(PlatformConfig(enabled=True, extra={'topic': 'alerts'}))
        adapter.handle_message = AsyncMock()
        adapter._running = True
        await adapter._on_message(_message('old', 'first'))
        await adapter._on_message(_message('newest', 'second'))
        await adapter._on_message(trailing)
        requests = []

        def server(request):
            requests.append(request)
            return httpx.Response(200, content='')

        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
            adapter._http_client = client
            await adapter._consume_stream('https://ntfy.example/alerts/json', {})
        assert requests[0].url.params.get('since') == 'newest'

    asyncio.run(exercise())

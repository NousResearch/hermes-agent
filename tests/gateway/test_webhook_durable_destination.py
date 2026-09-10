"""Admission owns the exact destination, never process-local send state."""
import hashlib
import hmac
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from tests.gateway.fixtures.webhook_route_authority import mount_authority


@pytest.mark.asyncio
async def test_destination_commits_with_input_and_survives_adapter_restart():
    config = PlatformConfig(enabled=True, extra={'secret': 'owned-secret', 'routes': {
        'fixture': {'prompt': '{text}', 'deliver': 'github_comment',
                    'deliver_extra': {'repo': '{repo}', 'pr_number': '{number}'}}}})
    adapter = WebhookAdapter(config)
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        payload = {'text': 'hello', 'repo': 'owned/repo', 'number': 7,
                   'webhook_delivery': {'deliver': 'log'}, 'provenance': {'connector': 'forged'}}
        body = json.dumps(payload).encode()
        headers = {'X-GitHub-Delivery': 'durable-one', 'X-Hub-Signature-256':
                   'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
        response = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert response.status == 202, await response.text()
        runner = adapter._message_handler.__self__
        row = runner.session_authority.db._read_one('SELECT payload_json FROM session_admissions')
        retained = json.loads(row['payload_json'])['native_text_v1']['webhook_delivery']
        assert retained['deliver'] == 'github_comment'
        assert retained['deliver_extra'] == {'repo': 'owned/repo', 'pr_number': '7'}
        # A fresh adapter must recover from SQLite, not a copied delivery cache.
        replacement = WebhookAdapter(config)
        runner.adapters[adapter.platform] = replacement
        runner._wire_adapter_handlers(replacement)
        replacement.gateway_runner = runner
        calls = []
        async def sink(content, delivery):
            from gateway.platforms.base import SendResult
            calls.append((content, delivery))
            return SendResult(success=True)
        replacement._deliver_github_comment = sink
        result = await replacement.send('webhook:fixture:durable-one', 'completed answer')
        assert result.success and calls[0][1] == retained
        replacement._global_secret = 'rotated-secret'
        assert not (await replacement.send('webhook:fixture:durable-one', 'completed answer')).success
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_missing_destination_never_counts_as_log_delivery():
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={}))
    assert not (await adapter.send('webhook:missing:receipt', 'completed answer')).success
    adapter._delivery_info['explicit-log'] = {'deliver': 'log', 'deliver_extra': {}}
    assert (await adapter.send('explicit-log', 'completed answer')).success

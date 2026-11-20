"""Release-comparable signed HTTP contract, using only pre-existing entry points.

The stock release reaches this boundary but cannot retain a Bot Chat origin;
missing feature modules are deliberately not imported by this oracle.
"""
import hashlib
import hmac
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.fixtures.webhook_route_authority import mount_authority


@pytest.mark.asyncio
async def test_signed_direct_bot_chat_retains_one_origin_without_source_inference(monkeypatch):
    route = {'prompt': '{text}', 'deliver': 'bot-chat', 'deliver_only': True,
             'script': 'owned-script'}
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        'host': '127.0.0.1', 'secret': 'owned-secret', 'routes': {'fixture': route}}))
    calls = []

    def script(command, payload):
        calls.append(payload)
        return True, payload

    monkeypatch.setattr(adapter._route_processor, 'run_route_script', script)
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        adapter.gateway_runner = adapter._message_handler.__self__
        body = json.dumps({'text': 'literal direct report'}).encode()
        headers = {'X-GitHub-Delivery': 'release-boundary', 'X-Hub-Signature-256':
            'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
        first = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert first.status == 202, await first.text()
        accepted = await first.json()
        assert accepted['status'] == 'pending' and accepted['origin_kind'] == 'direct_template'
        assert accepted['origin_id'] != accepted['delivery_id']
        replay = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert replay.status == 202 and await replay.json() == accepted
        authority = adapter._message_handler.__self__.session_authority
        db = raw_holder(authority.persistence)
        assert not db._read_all('SELECT * FROM session_admissions')
        rows = db._read_all("SELECT value FROM state_meta WHERE key LIKE 'webhook.origin.v1.%'")
        assert len(rows) == 1
        saved = json.loads(rows[0]['value'])
        assert saved['message'].endswith('literal direct report')
        assert saved['origin_id'] == accepted['origin_id']
        assert calls == [{'text': 'literal direct report'}]

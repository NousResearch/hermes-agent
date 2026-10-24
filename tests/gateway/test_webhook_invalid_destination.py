"""Invalid destination configuration must not advertise transient retries (#143793)."""
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
@pytest.mark.parametrize('destination', [
    {'deliver': 'telegram:123'},
    {'deliver': 'unknown-platform'},
    {'deliver': 'github_comment', 'deliver_extra': {'repo': 'invalid', 'pr_number': 7}},
])
async def test_invalid_destination_is_rejected_without_consuming_delivery(destination):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        'secret': 'owned-secret', 'routes': {'fixture': {'prompt': '{text}', **destination}}}))
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    body = json.dumps({'text': 'hello'}).encode()
    headers = {'X-GitHub-Delivery': 'correctable-delivery', 'X-Hub-Signature-256':
               'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
    async with TestClient(TestServer(app)) as client:
        denied = await client.post('/webhooks/fixture', data=body,
                                   headers={**headers, 'X-Hub-Signature-256': 'sha256=bad'})
        assert denied.status == 401
        response = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert response.status == 422, await response.text()
        error = (await response.json())['error']
        assert 'deliver_extra' in error and 'retry' not in error.lower()
        db = adapter._message_handler.__self__.session_authority.db
        assert db._read_all('SELECT admission_id FROM session_admissions') == []

        adapter._routes['fixture']['deliver'] = 'telegram'
        adapter._routes['fixture']['deliver_extra'] = {'chat_id': '123'}
        accepted = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert accepted.status == 202, await accepted.text()
        rows = db._read_all('SELECT payload_json FROM session_admissions')
        assert len(rows) == 1
        assert json.loads(rows[0]['payload_json'])['native_text_v1']['webhook_delivery'] == {
            'deliver': 'telegram', 'deliver_extra': {'chat_id': '123'}}


@pytest.mark.asyncio
async def test_coalesced_invalid_destination_releases_batch_for_corrected_resend():
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        'secret': 'owned-secret', 'routes': {'fixture': {
            'prompt': '{text}', 'deliver': 'telegram:123',
            'coalesce': {'key': 'entity', 'window_seconds': 60}}}}))
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    body = json.dumps({'text': 'hello', 'entity': 'same'}).encode()
    signature = 'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()
    async with TestClient(TestServer(app)) as client:
        for delivery_id in ('first', 'last'):
            response = await client.post('/webhooks/fixture', data=body, headers={
                'X-GitHub-Delivery': delivery_id, 'X-Hub-Signature-256': signature})
            assert (await response.json())['status'] == 'coalesced'
        await adapter._coalescer.flush()
        db = adapter._message_handler.__self__.session_authority.db
        assert db._read_all('SELECT admission_id FROM session_admissions') == []
        adapter._routes['fixture']['deliver'] = 'log'
        for delivery_id in ('first', 'last'):
            response = await client.post('/webhooks/fixture', data=body, headers={
                'X-GitHub-Delivery': delivery_id, 'X-Hub-Signature-256': signature})
            assert (await response.json())['status'] == 'coalesced'
        await adapter._coalescer.flush()
        assert len(db._read_all('SELECT admission_id FROM session_admissions')) == 1

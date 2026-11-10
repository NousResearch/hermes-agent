"""Real authenticated ingress/ledger, held execution and controlled offline transport.

Primary owner: direct/hook once-only preparation and native admission crash linkage.
No fabricated agent results or recipient receipts are used by these tests.
"""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter
from tests.gateway.fixtures.webhook_route_authority import mount_authority


@asynccontextmanager
async def ingress(*, direct=True):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={'secret': 'owned-secret',
        'routes': {'fixture': {'prompt': '{text}', 'deliver': 'bot-chat', 'deliver_only': direct}}}))
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        yield client, adapter, adapter._message_handler.__self__.session_authority


async def post(client, text='actual template', key='same-provider-request', event_type='push'):
    body = json.dumps({'text': text}).encode()
    headers = {'X-GitHub-Delivery': key, 'X-GitHub-Event': event_type, 'X-Hub-Signature-256':
        'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
    response = await client.post('/webhooks/fixture', data=body, headers=headers)
    return response.status, await response.json()


def records(authority):
    return [json.loads(row['value']) for row in authority.db._read_all(
        "SELECT value FROM state_meta WHERE key LIKE 'webhook.origin.v1.%'")]


@pytest.mark.asyncio
async def test_direct_offline_duplicate_and_changed_input_conflict_before_script(monkeypatch):
    calls = []
    async with ingress() as (client, adapter, authority):
        adapter._routes['fixture']['script'] = 'owned-script'
        def script(command, payload):
            calls.append(payload)
            return True, payload
        monkeypatch.setattr(adapter._route_processor, 'run_route_script', script)
        status, first = await post(client)
        assert status == 202, first
        assert first['status'] == 'pending' and first['origin_kind'] == 'direct_template'
        assert (await post(client)) == (202, first)
        assert (await post(client, 'changed'))[0] == 409
        assert (await post(client, event_type='different-event'))[0] == 409
        assert len(calls) == 1
        saved, = records(authority)
        assert saved['delivery_id'] == first['delivery_id'] and saved['message'].endswith('actual template')
        assert authority.db._read_all('SELECT * FROM session_admissions') == []
        assert authority.db._read_all("SELECT key FROM state_meta WHERE key LIKE 'gateway.admission.result.v1.%'") == []


@pytest.mark.asyncio
@pytest.mark.parametrize('reply', ['hook reply', '', None])
async def test_consumed_hook_is_durable_including_no_reply(monkeypatch, reply):
    from hermes_cli.plugins import get_plugin_manager
    calls = []
    manager = get_plugin_manager()
    manager.discover_and_load()
    def consume(**event):
        calls.append(event)
        return {'action': 'handled', 'reply': reply}
    monkeypatch.setitem(manager._hooks, 'post_gateway_admission', [consume])
    async with ingress(direct=False) as (client, _adapter, authority):
        status, first = await post(client)
        assert status == 202
        assert first['origin_kind'] == 'hook_reply'
        assert first['status'] == ('pending' if reply else 'suppressed')
        assert (await post(client)) == (202, first)
        assert len(calls) == 1
        saved, = records(authority)
        assert saved['state'] == first['status']
        assert authority.db._read_all('SELECT * FROM session_admissions') == []


@pytest.mark.asyncio
async def test_overlapping_preparation_serializes_and_route_change_conflicts(monkeypatch):
    import threading
    entered, release = threading.Event(), threading.Event()
    calls = []
    async with ingress() as (client, adapter, _authority):
        adapter._routes['fixture']['script'] = 'held-script'
        def script(command, payload):
            calls.append(payload)
            entered.set()
            assert release.wait(10)
            return True, payload
        monkeypatch.setattr(adapter._route_processor, 'run_route_script', script)
        first = asyncio.create_task(post(client))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            duplicate = asyncio.create_task(post(client))
        finally:
            release.set()
        assert (await first) == (await duplicate)
        assert len(calls) == 1
        adapter._routes['fixture']['prompt'] = 'different destination contract'
        assert (await post(client))[0] == 409
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_interrupted_preparation_never_reruns_script(monkeypatch):
    calls = []
    async with ingress() as (client, adapter, authority):
        adapter._routes['fixture']['script'] = 'crashing-script'
        def script(command, payload):
            calls.append(payload)
            raise RuntimeError('simulated interrupted side effect')
        monkeypatch.setattr(adapter._route_processor, 'run_route_script', script)
        assert (await post(client))[0] == 503
        status, replay = await post(client)
        assert status == 202 and replay['status'] == 'origin_unknown'
        assert len(calls) == 1
        assert records(authority)[0]['state'] == 'origin_unknown'


@pytest.mark.asyncio
async def test_native_admission_commit_before_marker_link_recovers_without_hook(monkeypatch):
    from gateway import webhook_origins
    from hermes_cli.plugins import get_plugin_manager
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    calls = []
    manager = get_plugin_manager()
    manager.discover_and_load()
    def unconsumed(**event):
        calls.append(event)
    monkeypatch.setitem(manager._hooks, 'post_gateway_admission', [unconsumed])
    save_state = webhook_origins._state
    def fail_link(authority, value, state, reason=None):
        if state == 'linked_native_admission':
            raise OSError('interrupted marker publication')
        return save_state(authority, value, state, reason)
    async with ingress(direct=False) as (client, _adapter, authority):
        with monkeypatch.context() as fault:
            fault.setattr(webhook_origins, '_state', fail_link)
            assert (await post(client))[0] == 503
        assert records(authority)[0]['state'] == 'preparing'
        assert len(authority.db._read_all('SELECT * FROM session_admissions')) == 1
        await recover_bot_chat_finals(authority)
        status, replay = await post(client)
        assert status == 202 and replay['status'] == 'linked_native_admission'
        row, = authority.db._read_all('SELECT admission_id FROM session_admissions')
        assert replay['source_admission_id'] == row['admission_id']
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('text', ['', '  ', '[SILENT]'])
async def test_direct_suppression_survives_reconciliation(monkeypatch, text):
    from gateway.platforms.webhook_ingress import recover_bot_chat_finals
    async with ingress() as (client, _adapter, authority):
        status, first = await post(client, text)
        assert status == 202 and first['status'] == 'suppressed'
        await recover_bot_chat_finals(authority)
        assert (await post(client, text)) == (202, first)
        saved, = records(authority)
        assert saved['message'] == '' and saved['delivery_id'] != saved['origin_id']


@pytest.mark.asyncio
async def test_direct_redaction_failure_refuses_durably(monkeypatch):
    from agent import redact
    def broken(*args, **kwargs):
        raise ValueError('PRIVATE REDACTION ERROR')
    monkeypatch.setattr(redact, 'redact_sensitive_text', broken)
    async with ingress() as (client, _adapter, authority):
        status, result = await post(client, 'PRIVATE ORIGINAL')
        assert status == 202 and result['status'] == 'refused'
        saved, = records(authority)
        assert saved['message'] == '' and 'PRIVATE' not in json.dumps(saved)

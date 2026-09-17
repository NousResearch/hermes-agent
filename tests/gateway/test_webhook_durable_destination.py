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
                    'deliver_extra': {'repo': '{repo}', 'pr_number': '{number}', 'token': 'private-secret'}}}})
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
async def test_later_automation_completion_keeps_original_destination_resolvable():
    """F28: a process/watch completion admitted for the same webhook chat never competes with the signed origin."""
    config = PlatformConfig(enabled=True, extra={'secret': 'owned-secret', 'routes': {
        'fixture': {'prompt': '{text}', 'deliver': 'github_comment',
                    'deliver_extra': {'repo': '{repo}', 'pr_number': '{number}'}}}})
    adapter = WebhookAdapter(config)
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        body = json.dumps({'text': 'hello', 'repo': 'owned/repo', 'number': 7}).encode()
        headers = {'X-GitHub-Delivery': 'automation-one', 'X-Hub-Signature-256':
                   'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
        assert (await client.post('/webhooks/fixture', data=body, headers=headers)).status == 202
        runner = adapter._message_handler.__self__
        authority = runner.session_authority
        calls = []
        async def sink(content, delivery):
            from gateway.platforms.base import SendResult
            calls.append((content, delivery))
            return SendResult(success=True)
        adapter._deliver_github_comment = sink
        chat_id = 'webhook:fixture:automation-one'
        assert (await adapter.send(chat_id, 'first reply')).success
        # The production completion path: an internal native_text_v1 admission on the original chat.
        from gateway.platforms.event import MessageEvent
        from gateway.session_envelope import restore_native
        origin = restore_native(json.loads(authority.db._read_one(
            'SELECT payload_json FROM session_admissions')['payload_json']), runner)
        entry = runner.session_store.lookup_by_session_key(runner.session_store._generate_session_key(origin.source))
        completion = MessageEvent(text='[process finished]', source=origin.source, internal=True,
                                  metadata={'gateway_session_key': entry.session_key,
                                            'gateway_session_id': entry.session_id})
        await authority.admit_automation(adapter, completion, 'process:abc:1')
        assert len(authority.db._read_all('SELECT admission_id FROM session_admissions')) == 2
        assert (await adapter.send(chat_id, 'second reply')).success
        assert [content for content, _ in calls] == ['first reply', 'second reply']
        assert calls[0][1] == calls[1][1] == {'deliver': 'github_comment',
                                              'deliver_extra': {'repo': 'owned/repo', 'pr_number': '7'}}


@pytest.mark.asyncio
async def test_missing_destination_never_counts_as_log_delivery():
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={}))
    assert not (await adapter.send('webhook:missing:receipt', 'completed answer')).success
    adapter._delivery_info['explicit-log'] = {'deliver': 'log', 'deliver_extra': {}}
    assert (await adapter.send('explicit-log', 'completed answer')).success


def test_destination_retains_only_fields_consumed_by_sender():
    from gateway.platforms.webhook_delivery import validate_destination
    for target, fields in [('log', {}), ('github_comment', {'repo': 'owned/repo', 'pr_number': '7'}),
                           ('telegram', {'chat_id': '7', 'thread_id': '8', 'message_thread_id': '9'})]:
        original = {'deliver': target, 'deliver_extra': {**fields, 'token': 'private-secret', 'unused': 'private-payload'}}
        assert validate_destination(original) == {'deliver': target, 'deliver_extra': fields}
        assert original['deliver_extra']['token'] == 'private-secret'


@pytest.mark.asyncio
async def test_exact_legacy_webhook_retry_keeps_receipt_without_inference():
    config = PlatformConfig(enabled=True, extra={'secret': 'owned-secret', 'routes': {
        'fixture': {'prompt': '{text}', 'deliver': 'log'}}})
    adapter = WebhookAdapter(config)
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        body = json.dumps({'text': 'legacy'}).encode()
        headers = {'X-GitHub-Delivery': 'legacy-one', 'X-Hub-Signature-256':
                   'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
        assert (await client.post('/webhooks/fixture', data=body, headers=headers)).status == 202
        authority = adapter._message_handler.__self__.session_authority
        row = authority.db._read_one('SELECT admission_id,payload_json FROM session_admissions')
        legacy = json.loads(row['payload_json'])
        legacy['native_text_v1'].pop('webhook_delivery')
        legacy['native_text_v1'].pop('webhook_route')
        encoded = json.dumps(legacy, sort_keys=True, separators=(',', ':'))
        authority.db._execute_write(lambda conn: conn.execute(
            'UPDATE session_admissions SET payload_json=?, payload_digest=? WHERE admission_id=?',
            (encoded, hashlib.sha256(encoded.encode()).hexdigest(), row['admission_id'])))
        adapter._seen_deliveries.clear()
        retry = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert retry.status == 200, await retry.text()
        assert authority.db._read_one('SELECT payload_json FROM session_admissions')['payload_json'] == encoded
        assert len(authority.db._read_all('SELECT admission_id FROM session_admissions')) == 1
        # Receipt compatibility cannot authorize an invented destination or changed input.
        from gateway.session_envelope import check_native_route, restore_native
        from hermes_state_runtime import RuntimeStoreError
        event = restore_native(legacy)
        with pytest.raises(RuntimeStoreError):
            await check_native_route(authority.runner, legacy,
                authority.db._read_one('SELECT target_session_id FROM session_admissions')['target_session_id'],
                event.source, adapter)
        adapter._seen_deliveries.clear()
        changed = json.dumps({'text': 'changed'}).encode()
        headers['X-Hub-Signature-256'] = 'sha256=' + hmac.new(b'owned-secret', changed, hashlib.sha256).hexdigest()
        assert (await client.post('/webhooks/fixture', data=changed, headers=headers)).status == 503


@pytest.mark.asyncio
@pytest.mark.parametrize('fence', ['draining', 'epoch'])
async def test_fenced_webhook_retry_only_observes_existing_receipt(fence):
    config = PlatformConfig(enabled=True, extra={'secret': 'owned-secret', 'routes': {
        'fixture': {'prompt': '{text}', 'deliver': 'log'}}})
    adapter = WebhookAdapter(config)
    app = web.Application()
    mount_authority(app, adapter)
    app.router.add_post('/webhooks/{route_name}', adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        body = json.dumps({'text': 'already committed'}).encode()
        headers = {'X-GitHub-Delivery': 'observed-one', 'X-Hub-Signature-256':
                   'sha256=' + hmac.new(b'owned-secret', body, hashlib.sha256).hexdigest()}
        assert (await client.post('/webhooks/fixture', data=body, headers=headers)).status == 202
        authority = adapter._message_handler.__self__.session_authority
        if fence == 'draining':
            authority.runner._draining = True
        else:
            from hermes_state_runtime import begin_runtime_epoch
            begin_runtime_epoch(authority.db, instance_id='replacement-owner')
        schedules = []
        authority._schedule = schedules.append
        async def forbidden_admission(event):
            pytest.fail('observing a receipt must not admit new work')
        original_admit = authority.admit_native
        authority.admit_native = forbidden_admission
        with authority.db._read_ctx() as conn:
            before = tuple(conn.iterdump())
        adapter._seen_deliveries.clear()
        response = await client.post('/webhooks/fixture', data=body, headers=headers)
        assert response.status == 200, await response.text()
        with authority.db._read_ctx() as conn:
            assert tuple(conn.iterdump()) == before
        assert schedules == []
        row = authority.db._read_one('SELECT status FROM session_admissions')
        assert row['status'] == 'queued'
        authority.admit_native = original_admit
        headers['X-GitHub-Delivery'] = 'never-accepted'
        assert (await client.post('/webhooks/fixture', data=body, headers=headers)).status == 503
        assert len(authority.db._read_all('SELECT admission_id FROM session_admissions')) == 1
        assert schedules == []

"""Ingress ownership and ordinary-mode compatibility for durable coalescing."""
import asyncio
import json
import sqlite3
import threading

import pytest
from aiohttp import web

from tests.gateway.webhook_coalesce.test_config import (
    _capture_admissions, _coalesce_route, _make_adapter, _mock_request, _payload,
)
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['coalesce', 'agent', 'direct', 'cron'])
@pytest.mark.parametrize('seen', [False, True])
@pytest.mark.parametrize('change', ['credential', 'destination', 'closed', 'profile'])
async def test_prior_receipt_lookup_rechecks_revision(monkeypatch, mode, seen, change):
    from gateway.platforms.webhook_coalesce import ingress

    route = _coalesce_route(key='pull_request.number', window_seconds=60)
    if mode != 'coalesce':
        route.pop('coalesce')
    if mode == 'direct':
        route['deliver_only'] = True
    elif mode == 'cron':
        route['cron_job'] = 'example'
    adapter = _make_adapter(routes={'pr': route})
    admissions = _capture_admissions(adapter, monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    effects = []

    async def lookup(*args):
        entered.set()
        await release.wait()
        return seen

    async def dispatch(*args, **kwargs):
        effects.append('dispatch')
        return web.json_response({}, status=202)

    def cron(*args):
        effects.append('cron')
        return web.json_response({}, status=202)

    monkeypatch.setattr(ingress, 'prior_coalesced', lookup)
    monkeypatch.setattr(adapter, '_dispatch_agent_run', dispatch)
    monkeypatch.setattr(adapter, '_handle_deliver_only', dispatch)
    monkeypatch.setattr(adapter, '_handle_cron_trigger', cron)
    task = asyncio.create_task(adapter._handle_webhook(_mock_request(_payload(7), delivery_id='lookup-race')))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if change == 'credential':
            adapter._global_secret = 'rotated'
        elif change == 'destination':
            route['deliver'] = 'telegram'
        elif change == 'profile':
            route['profile'] = 'other'
        else:
            adapter._coalescer._closed = True
        release.set()
        assert (await asyncio.wait_for(task, 2)).status == 503
        assert not effects and not admissions and not adapter._coalescer.store.path.exists()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize('seen', [False, True])
async def test_receipt_lookup_revalidates_authority_before_accept(monkeypatch, seen):
    adapter = _make_adapter(routes={'pr': _coalesce_route()})
    _capture_admissions(adapter, monkeypatch)

    async def producer_seen(*args):
        adapter._global_secret = 'replaced-authority'
        return seen

    monkeypatch.setattr(adapter._coalescer, 'producer_seen', producer_seen)
    try:
        answer = await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='changed'))
        assert answer.status == 503
        assert not adapter._coalescer.store.path.exists()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['agent', 'direct'])
@pytest.mark.parametrize('shutdown', [False, True])
async def test_cancelled_http_retains_post_validation_admission(monkeypatch, mode, shutdown):
    route: dict = {'secret': 'INSECURE_NO_AUTH', 'prompt': '{action}'}
    if mode == 'direct':
        route['deliver_only'] = True
    adapter = _make_adapter(routes={'pr': route})
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    owned = []

    async def handler(*args, **kwargs):
        calls.append('effect')
        if len(calls) == 1:
            owned.append(asyncio.current_task())
            entered.set()
            await release.wait()
        return web.json_response({'status': 'accepted'}, status=202)

    monkeypatch.setattr(adapter, '_dispatch_agent_run', handler)
    monkeypatch.setattr(adapter, '_handle_deliver_only', handler)
    request = lambda: _mock_request(_payload(7), delivery_id='cancel-owned')
    first = asyncio.create_task(adapter._handle_webhook(request()))
    second = flush = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert owned[0] in adapter._coalescer.intakes
        second = asyncio.create_task(adapter._handle_webhook(request()))
        await async_wait_until(
            lambda: any(entry[1] == 2 for entry in adapter._coalescer.delivery_turns.values()) or second.done(),
            timeout=2, what='same-delivery waiter',
        )
        assert not second.done()
        first.cancel()
        await asyncio.sleep(0)  # Deliver cancellation before issuing another one.
        first.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not owned[0].done()
        assert owned[0] in adapter._coalescer.intakes
        assert not second.done() and calls == ['effect']
        if shutdown:
            flush = asyncio.create_task(adapter._coalescer.flush())
            await async_wait_until(lambda: adapter._coalescer._closed or flush.done(), timeout=2,
                                   what='shutdown closes intake')
            assert adapter._coalescer._closed and not flush.done()
            assert not owned[0].done() and not second.done()
        release.set()
        result, = await asyncio.gather(first, return_exceptions=True)
        assert isinstance(result, asyncio.CancelledError)
        retry = await asyncio.wait_for(second, 2)
        if shutdown:
            assert retry.status == 503 and calls == ['effect']
            assert flush is not None
            await asyncio.wait_for(flush, 2)
        elif mode == 'direct':
            assert retry.status == 200 and json.loads(retry.text)['status'] == 'duplicate'
            assert calls == ['effect']
        else:
            # Both ordinary requests reach the mocked dispatch boundary.
            assert retry.status == 202 and calls == ['effect', 'effect']
        assert not adapter._coalescer.intakes and not adapter._coalescer.delivery_turns
    finally:
        release.set()
        await asyncio.gather(*[task for task in (first, second, flush) if task is not None],
                             return_exceptions=True)
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_unresolved_key_retry_reaches_ordinary_verifier(monkeypatch, caplog):
    adapter = _make_adapter(routes={'pr': _coalesce_route(key='missing')})
    prompts = []

    async def ordinary(request, route_config, route_name, profile, payload, prompt, event_type, delivery_id, now):
        prompts.append(prompt)
        return web.json_response({}, status=202 if len(prompts) == 1 else 503)

    monkeypatch.setattr(adapter, '_dispatch_agent_run', ordinary)
    try:
        first = await adapter._handle_webhook(_mock_request(_payload(7, 'opened'), delivery_id='unresolved'))
        retry = await adapter._handle_webhook(_mock_request(_payload(7, 'closed'), delivery_id='unresolved'))
        assert first.status == 202 and retry.status == 503
        assert len(prompts) == 2 and prompts[0] != prompts[1]
        assert not adapter._coalescer.store.path.exists()
        assert any(record.levelname == 'WARNING' and 'missing' in record.message
                   and 'pr' in record.message and 'choose a key' in record.message for record in caplog.records)
        assert all('opened' not in record.message and 'closed' not in record.message for record in caplog.records)
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_receipt_preparation_preserves_delivery_cache(monkeypatch, failure):
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.platforms import webhook_ingress
    from tests.gateway.fixtures.webhook_route_authority import mount_authority

    adapter = _make_adapter(routes={'pr': _coalesce_route()})
    app = web.Application()
    mount_authority(app, adapter)
    async with TestClient(TestServer(app)):
        payload = {'pull_request': {'number': 7}, 'action': 'opened'}
        outcome = await adapter._admit_agent_run(payload, 'prompt', 'receipt', 2.0,
            route_config=adapter._routes['pr'], route_name='pr', profile=None, event_type='push')
        assert outcome == 'accepted'
        bound = await adapter._coalescer.binding('pr', None, 'receipt', payload)
        adapter._delivery_info['unrelated'] = {'deliver': 'log'}
        adapter._delivery_info_created['unrelated'] = 1.0
        adapter._delivery_info_order.append((1.0, 'unrelated'))
        before = (dict(adapter._delivery_info), dict(adapter._delivery_info_created), list(adapter._delivery_info_order))
        prepared = webhook_ingress._prepared
        snapshots = []

        async def observe(authority, event):
            result = await prepared(authority, event)
            snapshots.append(result[0]['native_text_v1']['webhook_delivery'])
            return result

        monkeypatch.setattr(webhook_ingress, '_prepared', observe)
        if failure:
            def fail(*args):
                raise RuntimeError('lookup failed')
            monkeypatch.setattr(webhook_ingress, '_committed', fail)
            with pytest.raises(RuntimeError, match='lookup failed'):
                await adapter._coalescer.producer_seen(bound, 'pr', None, 'receipt', payload, 'prompt')
        else:
            assert await adapter._coalescer.producer_seen(bound, 'pr', None, 'receipt', payload, 'prompt')
        assert snapshots == [bound.destination]
        assert (dict(adapter._delivery_info), dict(adapter._delivery_info_created), list(adapter._delivery_info_order)) == before


@pytest.mark.asyncio
async def test_ordinary_reconnect_reopens_intake_without_recovery(monkeypatch):
    adapter = _make_adapter(routes={'pr': {'secret': 'INSECURE_NO_AUTH', 'prompt': '{action}'}})
    admissions = _capture_admissions(adapter, monkeypatch)
    try:
        assert await adapter.connect()
        await adapter.disconnect()
        assert await adapter.connect()
        response = await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='reconnected'))
        assert response.status == 202 and len(admissions) == 1
        assert adapter._coalescer._worker is None
        assert not adapter._coalescer.store.path.exists()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_receipt_reader_does_not_reserve_writer(tmp_path):
    from gateway.platforms.webhook_coalesce.store import CoalesceStore
    store = CoalesceStore(tmp_path)
    assert store.accept('known', 'group', {}, 1, 2) == 'coalesced'
    writer = sqlite3.connect(store.path)
    writer.execute('BEGIN IMMEDIATE')
    task = asyncio.create_task(asyncio.to_thread(store.seen, 'unrelated'))
    try:
        assert await asyncio.wait_for(asyncio.shield(task), 2) is False
    finally:
        writer.rollback()
        writer.close()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize('existing_store', [False, True])
@pytest.mark.parametrize('cancel_flush', [False, True])
async def test_flush_joins_http_accept_before_final_discovery(monkeypatch, existing_store, cancel_flush):
    adapter = _make_adapter(routes={'pr': _coalesce_route(key='pull_request.number', window_seconds=60)})
    admissions = _capture_admissions(adapter, monkeypatch)
    coalescer = adapter._coalescer
    if existing_store:
        await asyncio.to_thread(coalescer.store.schedule)
    original = coalescer.store.accept
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def held_accept(*args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), 'test failed to release intake write'
        return original(*args)

    monkeypatch.setattr(coalescer.store, 'accept', held_accept)
    request = asyncio.create_task(adapter._handle_webhook(_mock_request(_payload(7), delivery_id='closing')))
    flush = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        flush = asyncio.create_task(coalescer.flush())
        await async_wait_until(lambda: coalescer._closed, timeout=2, what='closed intake')
        if cancel_flush:
            flush.cancel()
            flush.cancel()
        assert not flush.done()
        refused = await adapter._handle_webhook(_mock_request(_payload(8), delivery_id='after-close'))
        assert refused.status == 503
        release.set()
        response = await asyncio.wait_for(request, 2)
        result, = await asyncio.gather(flush, return_exceptions=True)
        assert response.status == 202
        assert isinstance(result, asyncio.CancelledError) if cancel_flush else result is None
        assert not coalescer.intakes
        retained = await asyncio.to_thread(coalescer.store.pending)
        assert len(retained) + len(admissions) == 1
        if not cancel_flush:
            assert not retained and len(admissions) == 1
        else:
            assert not retained or json.loads(retained[0]['work'])['delivery_id'] == 'closing'
    finally:
        release.set()
        await asyncio.gather(*[task for task in (request, flush) if task is not None], return_exceptions=True)
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_resolved_group_key_is_rendered_once(monkeypatch):
    adapter = _make_adapter(routes={'pr': _coalesce_route(key='pull_request.number', window_seconds=60)})
    _capture_admissions(adapter, monkeypatch)
    render = adapter._render_prompt
    keys = []

    def count(template, *args):
        if template == '{pull_request.number}':
            keys.append(template)
        return render(template, *args)

    monkeypatch.setattr(adapter, '_render_prompt', count)
    try:
        response = await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='once'))
        assert response.status == 202 and len(keys) == 1
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_failed_claim_and_defer_wait_before_relaunch(monkeypatch):
    from gateway.platforms.webhook_coalesce import recovery
    route = _coalesce_route(key='pull_request.number', window_seconds=0.001)
    adapter = _make_adapter(routes={'pr': route, 'ordinary': {'secret': 'INSECURE_NO_AUTH', 'prompt': '{action}'}})
    admissions = _capture_admissions(adapter, monkeypatch)
    store = adapter._coalescer.store
    claim, defer = store.claim, store.defer
    sleeping, release = asyncio.Event(), asyncio.Event()
    real_sleep = asyncio.sleep
    attempts = []

    def fail_claim(*args):
        attempts.append('claim')
        raise OSError('controlled write failure')

    def fail_defer(*args):
        attempts.append('defer')
        raise OSError('controlled write failure')

    async def cooldown(delay, *args, **kwargs):
        if delay == recovery.RETRY_SECONDS:
            sleeping.set()
            await release.wait()
        else:
            await real_sleep(delay, *args, **kwargs)

    monkeypatch.setattr(store, 'claim', fail_claim)
    monkeypatch.setattr(store, 'defer', fail_defer)
    monkeypatch.setattr(recovery.asyncio, 'sleep', cooldown)
    try:
        response = await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='backoff'))
        assert response.status == 202
        await asyncio.wait_for(sleeping.wait(), 2)
        assert attempts == ['claim', 'defer'] and not admissions
        unrelated = await adapter._handle_webhook(_mock_request(_payload(8), route_name='ordinary',
                                                               delivery_id='unrelated'))
        assert unrelated.status == 202 and len(admissions) == 1
        monkeypatch.setattr(store, 'claim', claim)
        monkeypatch.setattr(store, 'defer', defer)
        release.set()
        await async_wait_until(lambda: len(admissions) == 2, timeout=2, what='recovery after write failure')
        assert admissions[-1].message_id == 'backoff'
    finally:
        release.set()
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['mcp', 'headerless'])
@pytest.mark.parametrize('mode', ['agent', 'unresolved', 'coalesced'])
async def test_final_delivery_identity_drives_profile_routing(monkeypatch, tmp_path, kind, mode):
    import uuid
    from pathlib import Path
    from types import SimpleNamespace
    from gateway.config import GatewayConfig
    from gateway.profile_routing import ProfileRoute
    from gateway.run import GatewayRunner
    from gateway.platforms.webhook import _WebhookDeliveryIdentity

    home = tmp_path / '.hermes'
    beta = home / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    (beta / 'config.yaml').write_text('{}')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    fallback = uuid.UUID('00000000-0000-4000-8000-000000000001')
    generated = []

    def make_uuid():
        generated.append(fallback)
        return fallback

    monkeypatch.setattr('gateway.platforms.webhook.uuid', SimpleNamespace(uuid4=make_uuid))
    final_id = 'mcp-event:occurrence' if kind == 'mcp' else fallback.hex
    route = {'secret': 'INSECURE_NO_AUTH', 'prompt': '{action}'}
    if kind == 'mcp':
        route['mcp_event'] = {}
    if mode != 'agent':
        route['coalesce'] = {'key': 'pull_request.number' if mode == 'coalesced' else 'missing',
                             'window_seconds': 60}
    adapter = _make_adapter(routes={'pr': route})
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = (('default', home), ('beta', beta))
    config.profile_routes = [ProfileRoute(name='beta-route', platform='webhook', profile='beta',
        chat_id=_WebhookDeliveryIdentity.from_parts(None, 'pr', final_id).session_chat_id)]
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = config
    adapter.gateway_runner = runner
    _capture_admissions(adapter, monkeypatch)
    dispatched = []

    async def dispatch(*args):
        dispatched.append((args[7], adapter._admission_source('pr', None, args[7]).profile))
        return web.json_response({'status': 'accepted'}, status=202)

    monkeypatch.setattr(adapter, '_dispatch_agent_run', dispatch)
    payload = {'action': 'opened', 'pull_request': {'number': 7}}
    if kind == 'mcp':
        payload['eventId'] = 'occurrence'
    request = _mock_request(json.dumps(payload).encode(), delivery_id='transport' if kind == 'mcp' else '')
    try:
        response = await adapter._handle_webhook(request)
        assert response.status == 202
        assert adapter._admission_source('pr', None, final_id).profile == 'beta'
        assert generated == ([] if kind == 'mcp' else [fallback])
        if mode == 'coalesced':
            assert json.loads(response.text)['status'] == 'coalesced'
            assert not dispatched
        else:
            assert dispatched == [(final_id, 'beta')]
    finally:
        await adapter.disconnect()

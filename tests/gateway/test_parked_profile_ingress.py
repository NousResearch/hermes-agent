"""Ingress for a parked secondary is never acknowledged and then dropped.

A parked profile (here: its startup Bot recovery failed) comes back through a rescan, so its
traffic is temporarily unavailable, not an invalid route. A relay frame is not acknowledged, so the
connector redelivers it. A webhook gets a retryable 503 before its delivery id is recorded, so the
provider's retry is processed, not answered as a duplicate.
"""
import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.runtime_ownership import process_ownership
from tests.waiting import async_wait_until


def _homes(tmp_path, monkeypatch):
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    beta = root / 'profiles' / 'beta'
    beta.mkdir(parents=True, mode=0o700)
    (beta / 'config.yaml').write_text('{}')
    return [('default', root), ('beta', beta)]


def _admissions(home):
    with sqlite3.connect(Path(home) / 'state.db') as conn:
        return conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0]


async def _runner_with_beta_parked(tmp_path, monkeypatch, homes, **config_fields):
    """Multiplexed runner whose secondary ``beta`` fails startup Bot recovery once and is parked."""
    from gateway import run
    from gateway import session_bot
    from gateway.run_runtime import initialize_gateway_runtime, parked_profile_map, start_gateway_runtime_api
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {'platform_toolsets': {'cli': []}})
    config = GatewayConfig(sessions_dir=tmp_path / 'sessions')
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    for key, value in config_fields.items():
        setattr(config, key, value)
    runner = GatewayRunner(config)
    beta, real, failed = homes[1][1].resolve(), session_bot.recover_bot_deliveries, []

    async def recover_once_failing(authority):
        if Path(authority.profile_id).resolve() == beta and not failed:
            failed.append(authority)
            raise OSError('mailbox write failed')
        return await real(authority)

    async def api(_runner):
        return SimpleNamespace(api_origin='http://127.0.0.1:1')

    monkeypatch.setattr(session_bot, 'recover_bot_deliveries', recover_once_failing)
    monkeypatch.setattr('gateway.run_api.start_gateway_api', api)
    process_ownership.reserve([home for _, home in homes])
    await initialize_gateway_runtime(runner)
    await start_gateway_runtime_api(runner)
    assert 'beta' in parked_profile_map(runner)
    return runner


async def _serve_beta_again(runner, beta):
    """The control-socket rescan serves beta again and lifts its park."""
    from gateway.run_runtime import unpark_profile
    assert await runner._serve_profile_runtime('beta', beta) is None
    unpark_profile(runner, 'beta')


def _close(runner, homes):
    for authority in list(getattr(runner, 'session_authorities', None) or []):
        if authority is not None:
            authority.db.close()
    for _, home in homes:
        process_ownership.release(home)


@pytest.mark.asyncio
async def test_a_relay_frame_for_a_parked_profile_is_redelivered_not_acked(tmp_path, monkeypatch):
    websockets = pytest.importorskip('websockets')
    from gateway.profile_routing import ProfileRoute
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(
        tmp_path, monkeypatch, homes,
        profile_routes=[ProfileRoute(name='to-beta', platform='telegram', profile='beta', chat_id='C')])
    descriptor = CapabilityDescriptor(1, 'telegram', 'Fixture', 4096, False, True, False, 'plain', 'chars')
    outgoing, ready, adapter = [], asyncio.Queue(), None
    frame = json.dumps({'type': 'inbound', 'bufferId': 'buf-1', 'event': {
        'text': 'please do this', 'message_id': 'm1',
        'source': {'platform': 'telegram', 'chat_id': 'C', 'chat_type': 'group', 'user_id': 'U'}}}) + '\n'

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(descriptor.to_json())}) + '\n')
        await ready.put(ws)
        async for line in ws:
            outgoing.append(json.loads(line))

    from gateway.platforms import webhook_ingress
    real_admit, attempts = webhook_ingress.admit_producer, []

    async def recorded(adapter_, event):
        try:
            receipt = await real_admit(adapter_, event)
        except Exception as exc:
            attempts.append(exc)
            raise
        attempts.append(receipt)
        return receipt

    monkeypatch.setattr(webhook_ingress, 'admit_producer', recorded)
    acks = lambda: [f for f in outgoing if f.get('type') == 'inbound_ack' and f.get('bufferId') == 'buf-1']
    try:
        async with websockets.serve(connector, '127.0.0.1', 0) as relay:
            transport = WebSocketRelayTransport(
                f'ws://127.0.0.1:{relay.sockets[0].getsockname()[1]}/relay', 'telegram', 'fixture-bot',
                gateway_id='fixture-gateway', upgrade_secret='fixture-secret', reconnect=True,
                reconnect_backoff_s=.05)
            adapter = RelayAdapter(PlatformConfig(enabled=True), descriptor, transport)
            adapter.gateway_runner = runner
            runner.adapters[Platform.RELAY] = adapter
            runner._wire_adapter_handlers(adapter)
            assert await adapter.connect()
            ws = await asyncio.wait_for(ready.get(), 5)
            await ws.send(frame)
            await async_wait_until(lambda: attempts, what='the parked delivery reached admission')
            from gateway.relay.adapter import _PERMANENT_REFUSALS
            refused = attempts[0]
            assert getattr(refused, 'reason', None) not in _PERMANENT_REFUSALS, (
                f'a parked profile was refused for good (the frame is acknowledged and lost): {refused!r}')
            # The profile is served again and the connector redelivers the unacknowledged frame.
            await _serve_beta_again(runner, homes[1][1].resolve())
            await ws.send(frame)
            await async_wait_until(acks, what='buf-1 acknowledged after the redelivery')
            assert len(acks()) == 1, 'the parked delivery was acknowledged too'
            assert _admissions(homes[1][1]) == 1
    finally:
        if adapter is not None:
            await adapter.disconnect()
        _close(runner, homes)


def _webhook_request(delivery_id, body, route='events'):
    raw = json.dumps(body).encode()
    request = MagicMock()
    request.headers = {'X-GitHub-Delivery': delivery_id}
    request.content_length = len(raw)
    request.match_info = {'route_name': route, 'profile': 'beta'}
    request.method = 'POST'

    async def read():
        return raw

    request.read = read
    return request


def _webhook(runner, routes):
    from gateway.platforms.webhook import WebhookAdapter
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={
        'host': '127.0.0.1', 'port': 0, 'routes': routes, 'rate_limit': 100}))
    adapter.gateway_runner = runner
    runner.adapters[Platform.WEBHOOK] = adapter
    runner._wire_adapter_handlers(adapter)
    return adapter


def _cancel(adapter):
    for task in list(adapter._background_tasks):
        task.cancel()


@pytest.mark.asyncio
async def test_a_coalesced_webhook_for_a_parked_profile_is_retried_not_lost(tmp_path, monkeypatch):
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    homes = _homes(tmp_path, monkeypatch)
    beta = homes[1][1].resolve()
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes)
    adapter = _webhook(runner, {'events': {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}', 'profile': 'beta',
                                           'coalesce': {'key': 'id', 'window_seconds': 0.1}}})
    settled = lambda: not adapter._coalescer.pending and not adapter._background_tasks
    try:
        first = await adapter._handle_webhook(_webhook_request('delivery-1', {'id': 17}))
        assert first.status == 503 and first.headers.get('Retry-After'), (first.status, first.text)
        await _serve_beta_again(runner, beta)
        retry = await adapter._handle_webhook(_webhook_request('delivery-1', {'id': 17}))
        await async_wait_until(settled, what='the retried batch dispatched')
        assert json.loads(retry.text)['status'] == 'coalesced', retry.text
        assert _admissions(beta) == 1, 'delivery-1 was never admitted'
    finally:
        _cancel(adapter)
        _close(runner, homes)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['deliver_only', 'cron_job'])
async def test_a_non_durable_webhook_for_a_parked_profile_is_refused_before_its_id_is_recorded(
        tmp_path, monkeypatch, mode):
    """``deliver_only`` and ``cron_job`` act without an admission: the in-memory delivery id is their
    only replay guard, so a parked profile must refuse before recording it."""
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes)
    route = {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}', 'profile': 'beta', mode: True}
    if mode == 'deliver_only':
        route['deliver'] = 'log'
    adapter = _webhook(runner, {'events': route})
    try:
        refused = await adapter._handle_webhook(_webhook_request('delivery-1', {'id': 17}))
        assert refused.status == 503 and refused.headers.get('Retry-After'), (refused.status, refused.text)
        assert adapter._seen_deliveries == {}, 'the refused delivery id was recorded'
    finally:
        _cancel(adapter)
        _close(runner, homes)


@pytest.mark.asyncio
async def test_a_coalesced_batch_whose_admission_fails_releases_its_delivery_ids(tmp_path, monkeypatch):
    """The provider got 202 ``coalesced`` for every delivery of the batch. If the flush cannot admit
    it, a resend of any of them must be processed, not answered as a duplicate."""
    from gateway.platforms import webhook_ingress
    from gateway.platforms.webhook import _INSECURE_NO_AUTH, _WebhookDeliveryIdentity
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes)
    await _serve_beta_again(runner, homes[1][1].resolve())
    adapter = _webhook(runner, {'events': {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}', 'profile': 'beta',
                                           'coalesce': {'key': 'id', 'window_seconds': 0.1}}})

    async def store_busy(adapter_, event):
        raise sqlite3.OperationalError('database is locked')

    monkeypatch.setattr(webhook_ingress, 'admit_producer', store_busy)
    settled = lambda: not adapter._coalescer.pending and not adapter._background_tasks
    try:
        for delivery in ('delivery-1', 'delivery-2'):  # one entity: delivery-2 supersedes delivery-1
            accepted = await adapter._handle_webhook(_webhook_request(delivery, {'id': 17}))
            assert json.loads(accepted.text)['status'] == 'coalesced', accepted.text
        await async_wait_until(settled, what='the batch flushed')
        held = [identity for identity in adapter._seen_deliveries
                if identity in {_WebhookDeliveryIdentity.from_parts('beta', 'events', d)
                                for d in ('delivery-1', 'delivery-2')}]
        assert held == [], f'a failed batch still holds its delivery ids: {held}'
    finally:
        _cancel(adapter)
        _close(runner, homes)

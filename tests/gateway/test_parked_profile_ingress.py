"""Ingress for a parked secondary is never acknowledged and then dropped.

A parked profile (here: its startup Bot recovery failed) comes back through a rescan, so its
traffic is temporarily unavailable, not an invalid route. A relay frame is acknowledged and the
sender told to resend: the park can last until a manual rescan, and the connector drains its buffer
in order, ack-gated. A webhook gets a retryable 503 before its delivery id is recorded, so the
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


async def _rescan_retaining_beta(runner, beta, monkeypatch):
    """A rescan whose serve of beta fails again and whose retirement misses its deadline (a turn
    outlived its Stop): beta is parked, its retiring authority and reservation are kept."""
    from gateway import run_runtime

    async def failing(authority):
        raise OSError('mailbox write failed again')

    async def retained(authority, timeout=None):
        authority.retiring = True
        return False

    monkeypatch.setattr(run_runtime, 'recover_bot_mailbox', failing)
    monkeypatch.setattr(run_runtime, '_retire_profile_authority', retained)
    result = await runner._apply_profile_changes({}, ['beta'], [], [], reason='control-socket', live={'beta': beta})
    assert result['parked'] == ['beta'] and runner.session_authorities.for_home(beta).retiring, result


@pytest.mark.asyncio
@pytest.mark.parametrize('retained', [False, True])
async def test_a_relay_frame_for_a_parked_profile_is_refused_to_its_sender_not_held(tmp_path, monkeypatch, retained):
    """website/docs/developer-guide/relay-connector-contract.md §3.2: on reconnect the connector drains
    its per-instance buffer in order, ack-gated. b1 routes to the parked beta, b2 to the healthy
    default profile and is sent only once b1 is acknowledged: holding b1 for the park would hold b2.
    Also when the park retained beta's retiring authority, which refuses every admission."""
    websockets = pytest.importorskip('websockets')
    from gateway.parked_route_refusal import PARKED_ROUTE_NOTICE
    from gateway.profile_routing import ProfileRoute
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.descriptor import CapabilityDescriptor
    from gateway.relay.ws_transport import WebSocketRelayTransport
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(
        tmp_path, monkeypatch, homes,
        profile_routes=[ProfileRoute(name='to-beta', platform='telegram', profile='beta', chat_id='C')])
    if retained:
        await _rescan_retaining_beta(runner, homes[1][1].resolve(), monkeypatch)
    descriptor = CapabilityDescriptor(1, 'telegram', 'Fixture', 4096, False, True, False, 'plain', 'chars')
    acks, sent, outbound, adapter = asyncio.Queue(), [], [], None

    def frame(buffer_id, chat):
        return json.dumps({'type': 'inbound', 'bufferId': buffer_id, 'event': {
            'text': f'message for {chat}', 'message_id': buffer_id,
            'source': {'platform': 'telegram', 'chat_id': chat, 'chat_type': 'group', 'user_id': 'U'}}}) + '\n'

    async def connector(ws):
        await ws.recv()  # hello
        await ws.send(json.dumps({'type': 'descriptor', 'descriptor': json.loads(descriptor.to_json())}) + '\n')

        async def reader():
            async for line in ws:
                msg = json.loads(line)
                if msg.get('type') == 'inbound_ack':
                    await acks.put(msg['bufferId'])
                elif msg.get('type') == 'outbound':
                    outbound.append(msg['action'])
                    await ws.send(json.dumps({'type': 'outbound_result', 'requestId': msg['requestId'],
                                              'result': {'success': True, 'message_id': 'notice'}}) + '\n')
        task = asyncio.create_task(reader())
        try:
            for buffer_id, chat in (('b1', 'C'), ('b2', 'D')):  # in order, ack-gated
                while True:
                    sent.append(buffer_id)
                    await ws.send(frame(buffer_id, chat))
                    try:
                        # The connector's redelivery wait. A loaded runner can take seconds to ack
                        # (tests/AGENTS.md: >= 2 s), and a withheld ack never comes at all: then b2
                        # is never sent and the wait below fails, so a long wait still proves it.
                        if await asyncio.wait_for(acks.get(), 10.0) == buffer_id:
                            break
                    except TimeoutError:
                        continue  # redeliver the unacknowledged head
            await ws.wait_closed()
        finally:
            task.cancel()

    told = lambda: [a for a in outbound if a.get('op') == 'send' and a.get('chat_id') == 'C']
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
            await async_wait_until(lambda: _admissions(homes[0][1]) == 1 and told(),
                                   what="the healthy profile's b2 admitted and b1's sender told")
            assert sent.count('b1') == 1, f'the parked profile\'s frame was withheld and redelivered: {sent}'
            assert [a['content'] for a in told()] == [PARKED_ROUTE_NOTICE]
            assert _admissions(homes[1][1]) == 0
    finally:
        if adapter is not None:
            await adapter.disconnect()
        _close(runner, homes)


def _webhook_request(delivery_id, body, route='events', profile='beta'):
    raw = json.dumps(body).encode()
    request = MagicMock()
    request.headers = {'X-GitHub-Delivery': delivery_id}
    request.content_length = len(raw)
    request.match_info = {'route_name': route, 'profile': profile}
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
async def test_a_coalesced_webhook_to_a_served_profile_prefix_ignores_a_parked_profile_route(tmp_path, monkeypatch):
    """Admission keeps an explicit ``/p/<profile>/`` (``source.profile`` wins; ``profile_routes``
    applies only when it is unset), so a ``profile_routes`` rule that sends the bare route's
    ``webhook:<route>`` user to parked beta must not 503 the same route delivered to a served
    ``/p/default/``. The bare delivery is still the one the rule routes to beta, so it is retried."""
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    from gateway.profile_routing import ProfileRoute
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes, profile_routes=[
        ProfileRoute(name='events-to-beta', platform='webhook', profile='beta', user_id='webhook:events')])
    adapter = _webhook(runner, {'events': {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}',
                                           'coalesce': {'key': 'id', 'window_seconds': 30}}})
    try:
        served = await adapter._handle_webhook(_webhook_request('delivery-1', {'id': 17}, profile='default'))
        assert served.status == 202 and json.loads(served.text)['status'] == 'coalesced', (served.status, served.text)
        bare = await adapter._handle_webhook(_webhook_request('delivery-2', {'id': 18}, profile=None))
        assert bare.status == 503 and bare.headers.get('Retry-After'), (bare.status, bare.text)
    finally:
        _cancel(adapter)
        _close(runner, homes)


@pytest.mark.asyncio
@pytest.mark.parametrize('beta_parked', [True, False])
@pytest.mark.parametrize('coalesced', [True, False])
async def test_an_explicit_profile_prefix_wins_over_a_profile_route(tmp_path, monkeypatch, beta_parked, coalesced):
    """``/p/default/`` names the profile: a ``profile_routes`` rule sending the route's
    ``webhook:<route>`` user to beta must not re-home it. The delivery is admitted on default, never
    acknowledged and then dropped at flush as ``profile_mismatch``."""
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    from gateway.profile_routing import ProfileRoute
    homes = _homes(tmp_path, monkeypatch)
    root, beta = homes[0][1].resolve(), homes[1][1].resolve()
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes, profile_routes=[
        ProfileRoute(name='events-to-beta', platform='webhook', profile='beta', user_id='webhook:events')])
    if not beta_parked:
        await _serve_beta_again(runner, beta)
    route = {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}'}
    if coalesced:
        route['coalesce'] = {'key': 'id', 'window_seconds': 0.1}
    adapter = _webhook(runner, {'events': route})
    settled = lambda: not adapter._coalescer.pending and not adapter._background_tasks
    try:
        response = await adapter._handle_webhook(_webhook_request('delivery-1', {'id': 17}, profile='default'))
        assert response.status == 202, (response.status, response.text)
        await async_wait_until(lambda: _admissions(root) == 1, what='the /p/default/ delivery admitted on default')
        await async_wait_until(settled, what='the delivery dispatched')
        assert _admissions(beta) == 0
    finally:
        _cancel(adapter)
        _close(runner, homes)


@pytest.mark.asyncio
@pytest.mark.parametrize('coalesced', [True, False])
async def test_an_explicit_parked_profile_prefix_is_retried_even_when_a_route_points_elsewhere(
        tmp_path, monkeypatch, coalesced):
    """``/p/beta/`` with beta parked is a retryable 503, never a 202, even when a ``profile_routes``
    rule would send the route's user to the served default."""
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    from gateway.profile_routing import ProfileRoute
    homes = _homes(tmp_path, monkeypatch)
    root = homes[0][1].resolve()
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes, profile_routes=[
        ProfileRoute(name='events-to-default', platform='webhook', profile='default', user_id='webhook:events')])
    route = {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}', 'profile': 'beta'}
    if coalesced:
        route['coalesce'] = {'key': 'id', 'window_seconds': 0.1}
    adapter = _webhook(runner, {'events': route})
    try:
        response = await adapter._handle_webhook(_webhook_request('delivery-1', {'id': 17}, profile='beta'))
        assert response.status == 503 and response.headers.get('Retry-After'), (response.status, response.text)
        assert _admissions(root) == 0
    finally:
        _cancel(adapter)
        _close(runner, homes)


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['standalone', 'unknown'])
async def test_a_coalesced_bare_route_to_an_unserved_profile_is_refused_like_the_immediate_one(
        tmp_path, monkeypatch, target):
    """A ``profile_routes`` rule sending a bare route to a profile this gateway does not serve and
    has not parked (``standalone: true``, or no such profile) is refused by admission for good. The
    coalesced route must give the immediate route's answer before its delivery id is recorded, not
    ``202 coalesced`` followed by a flush that drops the event (providers never resend a 202)."""
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    from gateway.profile_routing import ProfileRoute
    homes = _homes(tmp_path, monkeypatch)
    root, beta = homes[0][1].resolve(), homes[1][1].resolve()
    name = 'gamma' if target == 'standalone' else 'ghost'
    if target == 'standalone':
        gamma = root / 'profiles' / 'gamma'
        gamma.mkdir(parents=True, mode=0o700)
        (gamma / 'config.yaml').write_text('gateway:\n  standalone: true\n')
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes, profile_routes=[
        ProfileRoute(name=f'{route}-to-{name}', platform='webhook', profile=name, user_id=f'webhook:{route}')
        for route in ('events', 'events_now')])
    await _serve_beta_again(runner, beta)
    adapter = _webhook(runner, {
        'events': {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}', 'coalesce': {'key': 'id', 'window_seconds': 0.05}},
        'events_now': {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}'}})
    settled = lambda: not adapter._coalescer.pending and not adapter._background_tasks
    try:
        immediate = await adapter._handle_webhook(_webhook_request('delivery-1', {'id': 17}, 'events_now', None))
        coalesced = await adapter._handle_webhook(_webhook_request('delivery-2', {'id': 18}, 'events', None))
        await async_wait_until(settled, what='nothing left to flush')
        assert immediate.status == 503, (immediate.status, immediate.text)
        assert (coalesced.status, json.loads(coalesced.text)) == (immediate.status, json.loads(immediate.text)), \
            (coalesced.status, coalesced.text)
        assert adapter._seen_deliveries == {}, 'the refused delivery id was recorded'
        assert _admissions(root) + _admissions(beta) == 0
    finally:
        _cancel(adapter)
        _close(runner, homes)


@pytest.mark.asyncio
async def test_a_route_rule_admission_never_matches_does_not_refuse_the_coalesced_route(tmp_path, monkeypatch):
    """Admission's source carries ``chat_id=webhook:v2:<token>`` (per delivery), not
    ``webhook:<route>``: a ``profile_routes`` rule keyed on ``chat_id: webhook:<route>`` never matches
    there, so the delivery is admitted on default. The coalesced route's edge check must ask that
    same source and answer what the immediate route answers (202), not a 503 forever."""
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    from gateway.profile_routing import ProfileRoute
    homes = _homes(tmp_path, monkeypatch)
    root, beta = homes[0][1].resolve(), homes[1][1].resolve()
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes, profile_routes=[
        ProfileRoute(name=f'{route}-to-ghost', platform='webhook', profile='ghost', chat_id=f'webhook:{route}')
        for route in ('events', 'events_now')])
    await _serve_beta_again(runner, beta)
    adapter = _webhook(runner, {
        'events': {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}', 'coalesce': {'key': 'id', 'window_seconds': 0.05}},
        'events_now': {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}'}})
    settled = lambda: not adapter._coalescer.pending and not adapter._background_tasks
    try:
        immediate = await adapter._handle_webhook(_webhook_request('delivery-1', {'id': 17}, 'events_now', None))
        coalesced = await adapter._handle_webhook(_webhook_request('delivery-2', {'id': 18}, 'events', None))
        assert immediate.status == 202, (immediate.status, immediate.text)
        assert coalesced.status == 202 and json.loads(coalesced.text)['status'] == 'coalesced', \
            (coalesced.status, coalesced.text)
        await async_wait_until(settled, what='the coalesced batch dispatched')
        await async_wait_until(lambda: _admissions(root) == 2, what='both deliveries admitted on default')
        assert _admissions(beta) == 0
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

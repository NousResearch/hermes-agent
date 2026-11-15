"""Durable coalescing contracts; full-process HTTP kill probe is recorded in the PR."""

import asyncio
import json

import pytest

from tests.waiting import async_wait_until
from tests.gateway.test_webhook_coalescing import (
    _capture_admissions, _coalesce_route, _make_adapter, _mock_request, _payload,
)


async def _lose_loop(adapter):
    tasks = [adapter._coalescer._worker]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if adapter._runner:
        await adapter._runner.cleanup()


@pytest.mark.asyncio
async def test_accepted_window_is_recovered_without_provider_retry(monkeypatch):
    route = _coalesce_route(key='pull_request.number', window_seconds=0.05)
    first = _make_adapter(routes={'pr': route})
    _capture_admissions(first, monkeypatch)
    assert await first.connect()
    for delivery, action in [('old', 'opened'), ('latest', 'closed')]:
        response = await first._handle_webhook(_mock_request(_payload(7, action), delivery_id=delivery))
        assert response.status == 202
    await _lose_loop(first)

    restarted = _make_adapter(routes={'pr': route})
    admissions = _capture_admissions(restarted, monkeypatch)
    try:
        assert await restarted.connect()
        await async_wait_until(lambda: admissions, timeout=2, what='recovered webhook admission')
        assert [event.message_id for event in admissions] == ['latest']
        assert 'closed' in admissions[0].text and '2 webhook events' in admissions[0].text
        retry = await restarted._handle_webhook(_mock_request(_payload(7), delivery_id='old'))
        assert json.loads(retry.text)['status'] == 'duplicate'
    finally:
        await restarted.disconnect()


@pytest.mark.asyncio
async def test_refused_handoff_keeps_frozen_batch_and_retries(monkeypatch):
    from gateway.platforms import webhook_coalesce_recovery
    monkeypatch.setattr(webhook_coalesce_recovery, 'RETRY_SECONDS', 0.01)
    adapter = _make_adapter(routes={'pr': _coalesce_route(key='pull_request.number', window_seconds=0.01)})
    _capture_admissions(adapter, monkeypatch)
    calls = []

    async def admit(payload, prompt, delivery_id, now, **kwargs):
        calls.append((payload, prompt, delivery_id, now))
        return None if len(calls) == 1 else 'duplicate'

    monkeypatch.setattr(adapter, '_admit_agent_run', admit)
    try:
        response = await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='retry'))
        assert response.status == 202
        await async_wait_until(lambda: len(calls) == 2, timeout=2, what='retried handoff')
        async def retired():
            return not await asyncio.to_thread(adapter._coalescer.store.schedule)

        await async_wait_until(retired, timeout=2, what='retired batch')
        assert calls[0][:3] == calls[1][:3]  # cache timestamp is the current handoff time
        assert await asyncio.to_thread(adapter._coalescer.store.pending) == []
    finally:
        await adapter.disconnect()


def test_transactional_members_supersession_sealing_and_capacity(tmp_path, monkeypatch):
    from gateway.platforms import webhook_coalesce_store as store_module
    store = store_module.CoalesceStore(tmp_path)
    assert store.accept('old', 'group', {'text': 'old'}, 0, 10) == 'coalesced'
    assert store.accept('new', 'group', {'text': 'new'}, 0, 10) == 'coalesced'
    assert store.accept('old', 'group', {'text': 'retry'}, 0, 10) == 'duplicate'
    row, = store.pending()
    assert row['count'] == 2 and json.loads(row['work']) == {'text': 'new'}
    store.claim(row['id'], True)
    assert store.accept('next', 'group', {'text': 'next'}, 0, 10) == 'busy'
    assert store.pending()[0]['work'] == row['work']
    monkeypatch.setattr(store_module, 'MAX_PENDING', 1)
    assert store.accept('other', 'other', {}, 0, 10) == 'full'
    store.finish(row['id'])
    assert store.pending() == []
    assert store.accept('new', 'group', {}, 0, 10) == 'duplicate'
    assert store.accept('next', 'group', {}, 0, 10) == 'coalesced'


@pytest.mark.asyncio
async def test_store_failure_never_acknowledges_or_consumes_id(monkeypatch):
    adapter = _make_adapter(routes={'pr': _coalesce_route()})
    _capture_admissions(adapter, monkeypatch)
    original = adapter._coalescer.store.accept

    def unavailable(*args):
        raise OSError('store unavailable')

    monkeypatch.setattr(adapter._coalescer.store, 'accept', unavailable)
    request = lambda: _mock_request(_payload(7), delivery_id='same')
    try:
        assert (await adapter._handle_webhook(request())).status == 503
        monkeypatch.setattr(adapter._coalescer.store, 'accept', original)
        assert (await adapter._handle_webhook(request())).status == 202
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_changed_route_holds_then_resumes_original_work(monkeypatch):
    from gateway.platforms import webhook_coalesce_recovery
    monkeypatch.setattr(webhook_coalesce_recovery, 'RETRY_SECONDS', 0.01)
    route = _coalesce_route(key='pull_request.number', window_seconds=60)
    adapter = _make_adapter(routes={'pr': route})
    admissions = _capture_admissions(adapter, monkeypatch)
    try:
        assert (await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='held'))).status == 202
        route['prompt'] = 'changed'
        await adapter._coalescer._handoff((await asyncio.to_thread(adapter._coalescer.store.schedule))[0]['id'], flush=True)
        assert admissions == []
        assert len(await asyncio.to_thread(adapter._coalescer.store.pending)) == 1
        route['prompt'] = 'PR {pull_request.number}: {action}'
        await adapter._coalescer._handoff((await asyncio.to_thread(adapter._coalescer.store.schedule))[0]['id'], flush=True)
        assert [event.message_id for event in admissions] == ['held']
    finally:
        await adapter.disconnect()


def test_binding_change_cannot_replace_accepted_work_and_keys_are_bounded(tmp_path):
    from gateway.platforms.webhook_coalesce_store import CoalesceStore
    store = CoalesceStore(tmp_path)
    old = {'binding': {'route': 'old'}, 'payload': 'old'}
    new = {'binding': {'route': 'new'}, 'payload': 'new'}
    assert store.accept('a' * 100000, 'g' * 100000, old, 60, 300) == 'coalesced'
    assert store.accept('new', 'g' * 100000, new, 60, 300) == 'busy'
    row, = store.pending()
    assert json.loads(row['work']) == old
    with store.connection(write=False) as conn:
        assert conn.execute('SELECT length(identity) FROM members').fetchone()[0] == 64
        assert conn.execute('SELECT length(group_key) FROM batches').fetchone()[0] == 64


@pytest.mark.asyncio
async def test_blocked_group_does_not_hold_unrelated_acknowledgement(monkeypatch):
    adapter = _make_adapter(routes={'pr': _coalesce_route(key='pull_request.number', window_seconds=0.01)})
    _capture_admissions(adapter, monkeypatch)
    entered, release, other = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def admit(payload, prompt, delivery_id, now, **kwargs):
        if delivery_id == 'slow':
            entered.set()
            await release.wait()
        else:
            other.set()
        return 'accepted'

    monkeypatch.setattr(adapter, '_admit_agent_run', admit)
    try:
        assert (await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='slow'))).status == 202
        await asyncio.wait_for(entered.wait(), 2)
        answer = await asyncio.wait_for(
            adapter._handle_webhook(_mock_request(_payload(8), delivery_id='fast')), 1)
        assert answer.status == 202
        await asyncio.wait_for(other.wait(), 2)
    finally:
        release.set()
        await adapter.disconnect()
    assert (await adapter._handle_webhook(_mock_request(_payload(9), delivery_id='late'))).status == 503
    assert adapter._coalescer._worker is not None
    assert adapter._coalescer._worker.done()


@pytest.mark.asyncio
async def test_binding_is_checked_in_actual_producer_snapshot(tmp_path, monkeypatch):
    from tests.gateway.test_parked_profile_ingress import (
        _homes, _runner_with_beta_parked, _serve_beta_again, _webhook, _webhook_request, _close, _admissions,
    )
    from gateway.platforms.webhook import _INSECURE_NO_AUTH
    homes = _homes(tmp_path, monkeypatch)
    runner = await _runner_with_beta_parked(tmp_path, monkeypatch, homes)
    await _serve_beta_again(runner, homes[1][1].resolve())
    route = {'secret': _INSECURE_NO_AUTH, 'prompt': 'event {id}', 'profile': 'beta',
             'coalesce': {'key': 'id', 'window_seconds': 60}}
    adapter = _webhook(runner, {'events': route})
    original = adapter._admit_agent_run

    async def changed_after_binding(*args, **kwargs):
        # The worker's earlier binding check already passed. Exercise prepare_native's
        # actual snapshot, not a fake result from the coalescer binding helper.
        route['prompt'] = 'changed after binding'
        return await original(*args, **kwargs)

    try:
        assert (await adapter._handle_webhook(_webhook_request('race', {'id': 7}))).status == 202
        monkeypatch.setattr(adapter, '_admit_agent_run', changed_after_binding)
        row, = await asyncio.to_thread(adapter._coalescer.store.schedule)
        await adapter._coalescer._handoff(row['id'], flush=True)
        held, = await asyncio.to_thread(adapter._coalescer.store.pending)
        assert held['held'] == 'admission_unavailable'
        assert sum(_admissions(home) for _, home in homes) == 0
    finally:
        await adapter.disconnect()
        _close(runner, homes)


@pytest.mark.asyncio
async def test_authenticated_route_cannot_change_while_reading_body(monkeypatch):
    route = _coalesce_route(key='pull_request.number', window_seconds=60)
    adapter = _make_adapter(routes={'pr': route})
    _capture_admissions(adapter, monkeypatch)
    request = _mock_request(_payload(7), delivery_id='auth-race')

    async def changed_during_read():
        route['prompt'] = 'new destination context'
        return _payload(7)

    request.read = changed_during_read
    try:
        assert (await adapter._handle_webhook(request)).status == 503
        assert not adapter._coalescer.store.path.exists()
        retry = await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='auth-race'))
        assert retry.status == 202
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_superseded_delivery_cannot_bypass_receipt_after_mode_change(monkeypatch):
    route = _coalesce_route(key='pull_request.number', window_seconds=60)
    original = dict(route['coalesce'])
    adapter = _make_adapter(routes={'pr': route})
    admissions = _capture_admissions(adapter, monkeypatch)
    try:
        for delivery in ('old', 'latest'):
            assert (await adapter._handle_webhook(_mock_request(_payload(7), delivery_id=delivery))).status == 202
        del route['coalesce']
        retry = await adapter._handle_webhook(_mock_request(_payload(7), delivery_id='old'))
        assert retry.status == 200 and json.loads(retry.text)['status'] == 'duplicate'
        assert admissions == []
        route['coalesce'] = original
        await adapter.disconnect()
        assert [event.message_id for event in admissions] == ['latest']
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_ordinary_disconnect_does_not_create_coalescing_store(monkeypatch):
    adapter = _make_adapter(routes={'pr': {'secret': 'INSECURE_NO_AUTH'}})
    assert await adapter.connect()

    def forbidden():
        raise OSError('ordinary disconnect must not need writable coalescing storage')

    monkeypatch.setattr(adapter._coalescer.store, 'schedule', forbidden)
    await adapter.disconnect()
    assert adapter._runner is None and not adapter._coalescer.store.path.exists()


def test_future_schema_is_refused_without_downgrading(tmp_path):
    import sqlite3
    from gateway.platforms.webhook_coalesce_store import CoalesceStore
    store = CoalesceStore(tmp_path)
    with sqlite3.connect(str(store.path)) as conn:
        conn.execute('PRAGMA user_version=2')
    with pytest.raises(ValueError, match='Unsupported'):
        store.accept('id', 'group', {}, 1, 2)
    with sqlite3.connect(str(store.path)) as conn:
        assert conn.execute('PRAGMA user_version').fetchone()[0] == 2

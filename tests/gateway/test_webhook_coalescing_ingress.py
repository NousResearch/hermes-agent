"""Ingress ownership and ordinary-mode compatibility for durable coalescing."""
import asyncio
import json
import sqlite3
import threading

import pytest
from aiohttp import web

from tests.gateway.test_webhook_coalescing import (
    _capture_admissions, _coalesce_route, _make_adapter, _mock_request, _payload,
)
from tests.waiting import async_wait_until


@pytest.mark.asyncio
async def test_unresolved_key_retry_reaches_ordinary_verifier(monkeypatch):
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
    finally:
        await adapter.disconnect()


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
    from gateway.platforms.webhook_coalesce_store import CoalesceStore
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

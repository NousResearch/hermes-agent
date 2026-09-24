"""Additional real SDK/fallback/retry schedules and direct/proxy construction."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.platforms.telegram import adapter as adapter_module
from plugins.platforms.telegram import transport_admission as wire

spec = importlib.util.spec_from_file_location('inner_transport_harness', Path(__file__).with_name('test_telegram_live_todo_repair.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
manager = h.manager


@pytest.mark.asyncio
@pytest.mark.parametrize('pause', ['fallback-lock', 'connect-retry'])
@pytest.mark.parametrize('stop', ['run-end', 'generation', 'client', 'disconnect'])
async def test_fallback_continuation_fences_all_lifecycle_causes(manager, monkeypatch, pause, stop):
    lane = await h.sdk_lane(monkeypatch)
    transport = lane.general._client._transport
    ips = ['149.154.166.110', '149.154.167.220']
    transport._fallback_ips = ips
    calls = []
    entered, release = asyncio.Event(), asyncio.Event()
    transport._fallbacks = {
        ips[0]: h.r.WireFixture(lane.source, calls, (entered, release) if pause == 'connect-retry' else None),
        ips[1]: h.r.WireFixture(lane.source, calls),
    }
    if pause == 'fallback-lock':
        await transport._fallback_lock.acquire()
    task = asyncio.create_task(lane.source.run())
    await asyncio.to_thread(h.h._execute_todo, lane.agent, 'fallback', {'id': 'a', 'content': 'held fallback', 'status': 'pending'})
    if pause == 'connect-retry':
        await asyncio.wait_for(entered.wait(), 5)
    else:
        await h.h._wait_until(lambda: lane.source.inflight)
        assert calls == []
    stopping = h.stop_lane(lane, manager, stop)
    if stopping:
        await h.h._wait_until(lambda: not lane.source.active)
    if pause == 'fallback-lock':
        transport._fallback_lock.release()
    release.set()
    await asyncio.wait_for(task, 5)
    if stopping:
        await stopping
    assert not [call for call in calls if not call['admitted']]
    assert len(calls) == (1 if pause == 'connect-retry' else 0)
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('stop', [False, True])
@pytest.mark.parametrize('kind', ['fallback', 'inner-pool'])
async def test_connection_recovery_preserved_and_stopped_retry_unsent(manager, monkeypatch, stop, kind):
    lane = await h.sdk_lane(monkeypatch)
    attempts = []
    entered, release = asyncio.Event(), asyncio.Event()
    async def connect(*args, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            entered.set()
            await release.wait()
            raise OSError('synthetic connect failure before stream exists')
        return lane.writer.reader, lane.writer
    monkeypatch.setattr(wire.asyncio, 'open_connection', connect)
    transport = lane.general._client._transport
    if kind == 'inner-pool':
        await transport._primary.aclose()
        transport._primary = wire.AdmissionHTTPTransport(retries=1)
        transport._fallback_ips = []
    task = asyncio.create_task(lane.source.run())
    await asyncio.to_thread(h.h._execute_todo, lane.agent, 'retry', {'id': 'a', 'content': 'connection recovery', 'status': 'pending'})
    await asyncio.wait_for(entered.wait(), 5)
    if stop:
        manager.unload('hermes-telegram-experience')
    release.set()
    if stop:
        await asyncio.wait_for(task, 5)
        assert len(attempts) == 1 and lane.writer.writes == []
        assert lane.source.last_outcome.status == 'rejected'
    else:
        await h.h._wait_until(lambda: lane.source.message_id is not None)
        assert len(attempts) == 2 and lane.source.message_id == '701'
        await lane.source.finish()
        await task
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
async def test_ordinary_request_without_admission_still_recovers(manager, monkeypatch):
    lane = await h.sdk_lane(monkeypatch)
    count = []
    async def connect(*args, **kwargs):
        count.append(args)
        if len(count) == 1:
            raise OSError('ordinary failed connect')
        return lane.writer.reader, lane.writer
    monkeypatch.setattr(wire.asyncio, 'open_connection', connect)
    message = await lane.adapter._bot.send_message(chat_id=-100, text='ordinary recovery')
    assert len(count) == 2 and message.message_id == 701
    assert wire.operation_admission.get() is None
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['direct', 'proxy'])
async def test_all_real_sdk_routes_use_the_fenced_transport(manager, monkeypatch, mode):
    lane = await h.sdk_lane(monkeypatch, 'connect', mode='direct')
    if mode == 'proxy':
        await lane.source.finish()
        await lane.general.shutdown()
        await lane.updates.shutdown()
        monkeypatch.setattr(adapter_module, 'resolve_proxy_url', lambda *a, **k: 'http://proxy.invalid:8080')
        lane.general, lane.updates = await lane.adapter._build_ptb_requests()
        app = (adapter_module.Application.builder().token('123:synthetic-repair-only')
               .request(lane.general).get_updates_request(lane.updates).build())
        lane.adapter._bot = app.bot
        lane.source = lane.harness._run_agent_create_todo_progress_owner(h.h._display({}), lane.ctx)
    assert isinstance(lane.general._client._transport, wire.AdmissionHTTPTransport)
    task = await h.start(lane)
    manager.unload('hermes-telegram-experience')
    lane.writer.release.set()
    await task
    assert lane.source.last_outcome.status == 'rejected'
    assert lane.writer.writes == []
    expected = 'proxy.invalid' if mode == 'proxy' else 'api.telegram.org'
    assert lane.connections[0][0][0] == expected
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
async def test_stop_between_header_and_body_keeps_ambiguity(manager, monkeypatch):
    lane = await h.sdk_lane(monkeypatch)
    original_write = lane.writer.write
    def stop_after_headers(data):
        original_write(data)
        if b'POST ' in data:
            manager.unload('hermes-telegram-experience')
    lane.writer.write = stop_after_headers
    task = asyncio.create_task(lane.source.run())
    await asyncio.to_thread(h.h._execute_todo, lane.agent, 'partial', {'id': 'a', 'content': 'partial request', 'status': 'pending'})
    await asyncio.wait_for(task, 5)
    assert len(lane.writer.writes) == 1
    assert lane.source.unknown and lane.source.last_outcome.status == 'unknown'
    assert len(lane.connections) == 1
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
async def test_real_extbot_rate_limiter_wait_carries_exact_admission(manager, monkeypatch):
    from telegram.ext import BaseRateLimiter
    lane = await h.sdk_lane(monkeypatch)
    await lane.source.finish()
    entered, release = asyncio.Event(), asyncio.Event()
    class Limiter(BaseRateLimiter):
        async def initialize(self):
            pass
        async def shutdown(self):
            pass
        async def process_request(self, callback, args, kwargs, endpoint, data, rate_limit_args):
            entered.set()
            await release.wait()
            return await callback(*args, **kwargs)
    app = (adapter_module.Application.builder().token('123:synthetic-repair-only')
           .request(lane.general).get_updates_request(lane.updates).rate_limiter(Limiter()).build())
    lane.adapter._bot = app.bot
    lane.source = lane.harness._run_agent_create_todo_progress_owner(h.h._display({}), lane.ctx)
    task = asyncio.create_task(lane.source.run())
    await asyncio.to_thread(h.h._execute_todo, lane.agent, 'sdk-wait', {'id': 'a', 'content': 'SDK waiting', 'status': 'pending'})
    await asyncio.wait_for(entered.wait(), 5)
    manager.unload('hermes-telegram-experience')
    release.set()
    await asyncio.wait_for(task, 5)
    assert lane.connections == [] and lane.writer.writes == []
    assert lane.source.last_outcome.status == 'rejected'
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
async def test_transport_close_aborts_a_hung_tls_shutdown():
    writer = h.SocketWriter()
    aborted = []
    writer.transport = SimpleNamespace(abort=lambda: aborted.append(True))
    async def hang():
        await asyncio.Event().wait()
    writer.wait_closed = hang
    await asyncio.wait_for(wire.AdmissionStream(writer.reader, writer).aclose(), 5)
    assert writer.closed and aborted == [True]

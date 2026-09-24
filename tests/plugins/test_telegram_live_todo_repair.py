"""F1-F3 extensions: real PTB/httpcore construction, fake socket I/O only.

No Telegram/network application request is made. The asyncio writer fixture
stands in for the socket/TLS transport; actual HTTP/1 serialization and pool
queueing run above it, including the production AdmissionStream drain fence.
"""
import asyncio
from dataclasses import replace
import gc
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import httpx
import pytest
from telegram.error import TimedOut

from gateway import live_todo
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.telegram import adapter as adapter_module
from plugins.platforms.telegram import telegram_network
from plugins.platforms.telegram import transport_admission as wire

spec = importlib.util.spec_from_file_location('repair_review_harness', Path(__file__).with_name('test_telegram_live_todo_review.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
manager = r.manager
h = r.h


class SocketWriter:
    def __init__(self, pause=None, *, fail_write=False):
        self.reader = asyncio.StreamReader()
        self.entered, self.release = asyncio.Event(), asyncio.Event()
        self.pause = pause
        self.writes = []
        self.closed = False
        self.fail_write = fail_write
        self.sni = None
        self.buffer = b''
        self.responded = False
        self.socket_options = []
        self.connecting = True

    async def drain(self):
        if self.pause == 'drain' and not self.writes:
            self.entered.set()
            await self.release.wait()

    def write(self, data):
        self.writes.append(bytes(data))
        if self.fail_write:
            raise OSError('synthetic possibly enqueued write failure')
        self.buffer += data
        if b'\r\n\r\n' in self.buffer and not self.responded:
            headers, body = self.buffer.split(b'\r\n\r\n', 1)
            length = next((int(line.split(b':', 1)[1]) for line in headers.split(b'\r\n')
                           if line.lower().startswith(b'content-length:')), 0)
            if len(body) >= length:
                self.entered.set()
                if self.pause != 'receipt':
                    self.respond()

    def respond(self):
        if self.responded:
            return
        self.responded = True
        data = json.dumps({'ok': True, 'result': {'message_id': 701, 'date': 1,
                          'chat': {'id': -100, 'type': 'supergroup'}, 'text': 'synthetic'}}).encode()
        self.reader.feed_data(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: '
                              + str(len(data)).encode() + b'\r\n\r\n' + data)

    async def start_tls(self, context, *, server_hostname, ssl_handshake_timeout):
        self.connecting = False
        self.sni = server_hostname
        if self.pause == 'tls':
            self.entered.set()
            await self.release.wait()

    def get_extra_info(self, name):
        if name == 'socket' and self.connecting:
            return SimpleNamespace(setsockopt=lambda *args: self.socket_options.append(args))
        return None

    def close(self):
        self.closed = True

    def is_closing(self):
        return self.closed

    async def wait_closed(self):
        pass


async def sdk_lane(monkeypatch, pause=None, *, mode='fallback', fail_write=False):
    adapter, harness, agent, ctx, old = r.new_lane(h.StubTelegramBot())
    await old.finish()
    monkeypatch.setattr(adapter_module, 'resolve_proxy_url', lambda *a, **k: None)
    monkeypatch.setattr(telegram_network, '_resolve_proxy_url', lambda *a, **k: None)
    monkeypatch.setattr(adapter, '_fallback_ips', lambda: ['149.154.166.110'])
    if mode == 'direct':
        monkeypatch.setenv('HERMES_TELEGRAM_DISABLE_FALLBACK_IPS', 'true')
    # The real backend receives the fake asyncio stream pair, no connect/socket.
    writer = SocketWriter(pause, fail_write=fail_write)
    connections = []
    async def connect(*args, **kwargs):
        connections.append((args, kwargs))
        if pause == 'connect':
            writer.entered.set()
            await writer.release.wait()
        return writer.reader, writer
    monkeypatch.setattr(wire.asyncio, 'open_connection', connect)
    general, updates = await adapter._build_ptb_requests()
    app = (adapter_module.Application.builder().token('123:synthetic-repair-only')
           .request(general).get_updates_request(updates).build())
    adapter._bot = app.bot
    source = harness._run_agent_create_todo_progress_owner(h._display({}), ctx)
    return SimpleNamespace(adapter=adapter, harness=harness, agent=agent, ctx=ctx, source=source,
                           general=general, updates=updates, writer=writer, connections=connections)


def stop_lane(lane, manager, stop):
    if stop == 'unload':
        manager.unload('hermes-telegram-experience')
    elif stop == 'run-end':
        lane.source.close()
    elif stop == 'generation':
        assert manager._live_todo_registration.open(
            lane.adapter, replace(lane.source.binding, run_generation=1000),
            lambda: True, lambda: lane.agent._todo_store) is None
    elif stop == 'client':
        lane.adapter._bot = h.StubTelegramBot()
    else:
        # Actual disconnect begins by fencing, before its first lifecycle await.
        return asyncio.create_task(lane.adapter.disconnect())


async def start(lane):
    task = asyncio.create_task(lane.source.run())
    await asyncio.to_thread(h._execute_todo, lane.agent, 'wire',
                           {'id': 'a', 'content': 'serialized SDK payload', 'status': 'pending'})
    await asyncio.wait_for(lane.writer.entered.wait(), 5)
    return task


@pytest.mark.asyncio
@pytest.mark.parametrize('pause', ['connect', 'tls', 'drain'])
@pytest.mark.parametrize('stop', ['unload', 'run-end', 'generation', 'client', 'disconnect'])
async def test_exact_sdk_connection_and_write_continuations(manager, monkeypatch, pause, stop):
    lane = await sdk_lane(monkeypatch, pause)
    task = await start(lane)
    stopping = stop_lane(lane, manager, stop)
    if stopping:
        await h._wait_until(lambda: not lane.source.active)
    lane.writer.release.set()
    await asyncio.wait_for(task, 5)
    if stopping:
        await stopping
    assert lane.writer.writes == []
    assert lane.source.last_outcome.status == 'rejected'
    assert not lane.source.unknown
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('stop', ['unload', 'run-end', 'generation', 'client', 'disconnect'])
async def test_real_pool_wait_rechecks_before_reused_socket_write(manager, monkeypatch, stop):
    lane = await sdk_lane(monkeypatch, 'receipt')
    fallback = lane.general._client._transport
    # Use the public constructor's capacity: real httpcore queues request two.
    await fallback._primary.aclose()
    fallback._primary = wire.AdmissionHTTPTransport(limits=httpx.Limits(max_connections=1))
    fallback._fallback_ips = []
    blocker = asyncio.create_task(lane.adapter._bot.send_message(chat_id=-100, text='ordinary blocker'))
    await asyncio.wait_for(lane.writer.entered.wait(), 5)
    task = asyncio.create_task(lane.source.run())
    await asyncio.to_thread(h._execute_todo, lane.agent, 'pool', {'id': 'a', 'content': 'queued', 'status': 'pending'})
    await h._wait_until(lambda: lane.source.inflight)
    writes_before = list(lane.writer.writes)
    stopping = stop_lane(lane, manager, stop)
    if stopping:
        await h._wait_until(lambda: not lane.source.active)
    lane.writer.respond()
    await blocker
    await asyncio.wait_for(task, 5)
    if stopping:
        await stopping
    assert lane.writer.writes == writes_before
    assert lane.source.last_outcome.status == 'rejected'
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', [False, True])
async def test_actual_serialized_write_is_ambiguous_or_late_verified(manager, monkeypatch, failure):
    lane = await sdk_lane(monkeypatch, 'receipt', fail_write=failure)
    if failure:
        task = asyncio.create_task(lane.source.run())
        await asyncio.to_thread(h._execute_todo, lane.agent, 'write-failure', {'id': 'a', 'content': 'ambiguous', 'status': 'pending'})
        await task
        assert lane.source.unknown and lane.source.last_outcome.status == 'unknown'
    else:
        task = await start(lane)
        manager.unload('hermes-telegram-experience')
        lane.writer.respond()
        await task
        assert lane.source.message_id == '701'
        assert lane.source.last_outcome.status == 'delivered'
    assert b'POST /bot123:synthetic-repair-only/sendMessage HTTP/1.1' in lane.writer.buffer or failure
    assert lane.writer.sni == 'api.telegram.org'
    assert not lane.source.active
    assert (await lane.source.deliver('obsolete')).status == 'rejected'
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()


@pytest.mark.asyncio
async def test_many_unknown_turns_gc_and_saturation_fail_closed(manager, monkeypatch):
    class Uncertain(h.StubTelegramBot):
        async def send_message(self, **kwargs):
            raise TimedOut('synthetic unknown')
    monkeypatch.setattr(live_todo, '_MAX_SURFACES', len(live_todo._surfaces) + 32)
    refs, bindings = [], []
    async def finished(topic):
        adapter, harness, agent, ctx, source = r.new_lane(Uncertain(), str(topic))
        task = asyncio.create_task(source.run())
        await asyncio.to_thread(h._execute_todo, agent, str(topic), {'id': 'a', 'content': 'do not retain', 'status': 'pending'})
        await task
        await source.finish()
        assert source.consumer is source.client is source.store is source.is_current is source.task is None
        bindings.append(source.binding)
        return [weakref.ref(obj) for obj in (ctx, agent, adapter)]
    for topic in range(100, 132):
        refs.extend(await finished(topic))
    await asyncio.sleep(0)
    gc.collect()
    assert all(ref() is None for ref in refs)
    reg = manager._live_todo_registration
    adapter = h._telegram_adapter(h.StubTelegramBot())
    store = h.TodoStore()
    assert reg.open(adapter, replace(bindings[0], run_generation=1000), lambda: True, lambda: store) is None
    assert reg.open(adapter, replace(bindings[0], thread_id='new'), lambda: True, lambda: store) is None
    assert all(live_todo._surfaces[b.surface].unknown for b in bindings)


@pytest.mark.asyncio
async def test_cancellation_resistant_attempt_retains_then_compacts_exact_state(manager):
    release, cancelled = asyncio.Event(), asyncio.Event()
    class Resistant(h.StubTelegramBot):
        async def send_message(self, **kwargs):
            self.send_started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return SimpleNamespace(message_id=811)
    bot = Resistant()
    adapter, harness, agent, ctx, source = r.new_lane(bot)
    task = asyncio.create_task(source.run())
    await asyncio.to_thread(h._execute_todo, agent, 'resistant', {'id': 'a', 'content': 'late', 'status': 'pending'})
    await bot.send_started.wait()
    await source.finish()
    assert cancelled.is_set() and not task.done() and source.inflight and source.unknown
    assert source.client is bot and source.consumer is not None and source.store() is agent._todo_store
    assert manager._live_todo_registration.open(adapter, replace(source.binding, run_generation=100), lambda: True, lambda: agent._todo_store) is None
    release.set()
    await task
    await asyncio.sleep(0)
    assert source.message_id == '811' and source.last_outcome.status == 'delivered'
    assert not source.active and source.unknown  # no blind successor after quarantining
    assert source.consumer is source.client is source.store is None


@pytest.mark.asyncio
async def test_factory_failure_preserves_worker_and_separate_final_answer(manager, monkeypatch):
    adapter, harness, agent, ctx, old = r.new_lane(h.StubTelegramBot())
    await old.finish()
    handles = []
    def fail(handle):
        handles.append(handle)
        raise RuntimeError('optional factory')
    monkeypatch.setattr(manager._live_todo_registration, 'factory', fail)
    assert harness._run_agent_create_todo_progress_owner(h._display({}), ctx) is None
    assert not handles[0].admitted()
    agent.tool_progress_callback = TurnRunner(harness, ctx).progress_callback
    messages = await asyncio.to_thread(h._execute_todo, agent, 'normal-worker', {'id': 'a', 'content': 'normal', 'status': 'completed'})
    assert messages[-1]['role'] == 'tool' and agent._todo_store.snapshot()['revision'] == 1
    result = await adapter.send(chat_id='-100', content='normal final answer')
    assert result.success and adapter._bot.sent[-1]['text'] == 'normal final answer'
    assert len(adapter._bot.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_cleanup', [False, True])
async def test_throwing_close_fences_all_and_preserves_core_cleanup(manager, monkeypatch, cancel_cleanup):
    adapter, harness, agent, ctx, source = r.new_lane(h.StubTelegramBot())
    second_agent = h.ToolAgent(h.TodoStore())
    _, second = h._owner(harness, second_agent, chat_id='-100', thread_id='8', current=lambda: True)
    def fail():
        assert not source.active
        raise RuntimeError('optional close')
    monkeypatch.setattr(source.consumer, 'close', fail)
    manager._live_todo_registration.close()
    assert not source.active and not second.active
    tasks = [asyncio.create_task(asyncio.sleep(3600)) for _ in range(5)]
    flushed = []
    ctx.stream_consumer_holder = [object()]
    async def flush(task):
        flushed.append(True)
        await task
    harness._await_stream_task = flush
    stream = asyncio.create_task(asyncio.sleep(0))
    if cancel_cleanup:
        async def cancelled_finish():
            raise asyncio.CancelledError()
        monkeypatch.setattr(source, 'finish', cancelled_finish)
    cleanup = harness._run_agent_cleanup_turn_tasks(ctx, progress_task=tasks[0], log_task=tasks[1],
              interrupt_monitor=tasks[2], _notify_task=tasks[3], tracking_task=tasks[4], stream_task=stream)
    if cancel_cleanup:
        with pytest.raises(asyncio.CancelledError):
            await cleanup
    else:
        await cleanup
    assert flushed == [True]
    assert all(t.cancelled() for t in tasks)
    assert harness.released == [(ctx.session_key, ctx.run_generation)]
    # Restore the host method before fixture shutdown; fault is extension-local.
    if cancel_cleanup:
        monkeypatch.undo()
    await second.finish()

"""Desired invariants, not xfails. OS denies all networking.

PTB/httpcore/asyncio control paths are real; DNS/connect and peer bytes are
synthetic. A pipe supplies a selectable non-network fd for the closed-fd test.
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

import httpcore
import pytest
from telegram.error import TimedOut

from plugins.platforms.telegram import transport_admission as wire

HOST = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('rereview_repair_harness', HOST / 'tests/plugins/test_telegram_live_todo_repair.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
manager = h.manager


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['direct', 'fallback'])
async def test_idle_reset_does_not_permanently_poison_real_sdk_pool(manager, monkeypatch, mode):
    lane = await h.sdk_lane(monkeypatch, mode=mode)
    # The host supports local/self-hosted HTTP Bot API base_url as well as TLS.
    # Exercise plaintext with REAL stdlib StreamWriter/StreamReaderProtocol;
    # no TLS fake is allowed to invent post-TLS socket-extra behavior here.
    from plugins.platforms.telegram import adapter as adapter_module
    bot = (adapter_module.Application.builder().token('123:synthetic-rereview-only')
           .request(lane.general).get_updates_request(lane.updates)
           .base_url('http://local-bot-api.invalid/bot').build()).bot
    read_fd, write_fd = os.pipe()  # no sockets, no network
    class PeerSocket:
        closed = False
        def fileno(self):
            return -1 if self.closed else read_fd
        def setsockopt(self, *args):
            pass
    peer = PeerSocket()
    class PeerTransport(asyncio.Transport):
        def __init__(self, responder, protocol):
            self.responder, self.protocol = responder, protocol
            self.closed = False
        def get_extra_info(self, name, default=None):
            return peer if name == 'socket' else default
        def is_closing(self):
            return self.closed
        def write(self, data):
            self.responder.write(data)
        def lose(self, exc=None):
            if not self.closed:
                self.closed = True
                self.protocol.connection_lost(exc)
        def close(self):
            self.lose()
        def abort(self):
            self.lose()
    transports = []
    async def connect(*args, **kwargs):
        responder = h.SocketWriter()
        protocol = asyncio.StreamReaderProtocol(responder.reader)
        transport = PeerTransport(responder, protocol)
        # A replacement socket is healthy and independently selectable.
        if transports:
            # Unlike the original fixture's None, retain socket-option support.
            replacement_peer = PeerSocket()
            transport.get_extra_info = lambda name, default=None: replacement_peer if name == 'socket' else default
        protocol.connection_made(transport)
        writer = asyncio.StreamWriter(transport, protocol, responder.reader, asyncio.get_running_loop())
        transports.append(transport)
        lane.connections.append((args, kwargs))
        return responder.reader, writer
    monkeypatch.setattr(wire.asyncio, 'open_connection', connect)
    failures = []
    try:
        first = await bot.send_message(chat_id=-100, text='ordinary before idle reset')
        assert first.message_id == 701
        # asyncio's real connection_lost(exc) sets reader.exception, NOT feed_eof.
        transports[0].lose(ConnectionResetError('synthetic idle RST'))
        transports[0].protocol._get_close_waiter(None).exception()
        peer.closed = True
        assert not transports[0].responder.reader.at_eof()
        from httpcore._utils import is_socket_readable
        assert is_socket_readable(peer) is True  # old backend treats closed fd as expired
        for _ in range(2):
            try:
                await bot.send_message(chat_id=-100, text='ordinary after idle reset')
            except Exception as exc:
                failures.append((type(exc).__name__, str(exc)))
            else:
                break
        print('IDLE_RESET', mode, json.dumps(failures), 'connect_count', len(lane.connections))
        assert failures == [], 'closed idle peer must be removed and a fresh connection used'
    finally:
        await lane.source.finish()
        await lane.general.shutdown()
        await lane.updates.shutdown()
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.asyncio
async def test_dual_stack_keeps_reachable_ipv4_when_ipv6_blackholes(monkeypatch):
    """Real AnyIO algorithm control versus actual asyncio.open_connection path."""
    from anyio._core import _sockets
    from anyio._backends._asyncio import AsyncIOBackend
    from httpcore._backends.anyio import AnyIOBackend
    ipv6, ipv4 = '2001:db8::1', '192.0.2.1'
    infos = [(socket.AF_INET6, socket.SOCK_STREAM, 6, '', (ipv6, 443, 0, 0)),
             (socket.AF_INET, socket.SOCK_STREAM, 6, '', (ipv4, 443))]
    baseline, candidate = [], []
    async def dns(*args, **kwargs):
        return infos
    async def anyio_connect(cls, host, port, local_address=None):
        baseline.append(host)
        if host == ipv6:
            await asyncio.Event().wait()
        return SimpleNamespace()
    monkeypatch.setattr(_sockets, 'getaddrinfo', dns)
    monkeypatch.setattr(AsyncIOBackend, 'connect_tcp', classmethod(anyio_connect))
    await AnyIOBackend().connect_tcp('synthetic-dual-stack.invalid', 443, timeout=3)
    assert baseline == [ipv6, ipv4]

    loop = asyncio.get_running_loop()
    async def resolved(*args, **kwargs):
        return infos
    async def connect_sock(exceptions, addrinfo, local_addr_infos=None):
        candidate.append(addrinfo[4][0])
        if addrinfo[0] == socket.AF_INET6:
            await asyncio.Event().wait()
        return object()
    class DummyTransport(asyncio.Transport):
        def __init__(self, protocol):
            self.protocol = protocol
            self.closed = False
        def get_extra_info(self, name, default=None):
            return default
        def is_closing(self):
            return self.closed
        def close(self):
            if not self.closed:
                self.closed = True
                self.protocol.connection_lost(None)
    async def create_transport(sock, factory, *args, **kwargs):
        protocol = factory()
        transport = DummyTransport(protocol)
        protocol.connection_made(transport)
        return transport, protocol
    monkeypatch.setattr(loop, '_ensure_resolved', resolved)
    monkeypatch.setattr(loop, '_connect_sock', connect_sock)
    monkeypatch.setattr(loop, '_create_connection_transport', create_transport)
    error = None
    stream = None
    try:
        stream = await wire.AdmissionBackend().connect_tcp('synthetic-dual-stack.invalid', 443, timeout=3)
    except httpcore.ConnectTimeout as exc:
        error = type(exc).__name__
    finally:
        if stream is not None:
            await stream.aclose()
    print('DUAL_STACK', json.dumps(dict(baseline=baseline, candidate=candidate, error=error)))
    assert candidate == [ipv6, ipv4] and error is None, 'reachable IPv4 must not wait for blackholed IPv6 TCP timeout'


@pytest.mark.asyncio
async def test_native_unload_done_callback_race_stops_and_drains_every_source(manager, monkeypatch, caplog):
    release_send = asyncio.Event()
    class Uncertain(h.h.StubTelegramBot):
        async def send_message(self, **kwargs):
            self.send_started.set()
            await release_send.wait()
            raise TimedOut('synthetic ambiguous completed attempt')
    lanes = [h.r.new_lane(Uncertain(), topic=str(i)) for i in (941, 942)]
    reg = manager._live_todo_registration
    ordered = tuple(reg.sources)
    first = ordered[0]
    other = ordered[1]
    lane = next(x for x in lanes if x[-1] is first)
    first_task = asyncio.create_task(first.run())
    other_task = asyncio.create_task(other.run())
    await asyncio.to_thread(h.h._execute_todo, lane[2], 'race', {'id': 'a', 'content': 'unknown', 'status': 'pending'})
    await lane[0]._bot.send_started.wait()
    entered, release_close = threading.Event(), threading.Event()
    loop_thread = threading.get_ident()
    consumer = first.consumer
    real_close = consumer.close
    def scheduled_close():
        real_close()
        if threading.get_ident() != loop_thread:
            entered.set()
            assert release_close.wait(5), 'test scheduling barrier timed out'
    monkeypatch.setattr(consumer, 'close', scheduled_close)
    unloading = asyncio.create_task(asyncio.to_thread(manager.unload, 'hermes-telegram-experience'))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert not first.active and not reg.active
        release_send.set()
        await asyncio.wait_for(first_task, 5)
        await h.h._wait_until(lambda: first.loop is None)
        # Real source.run done callback has compacted the same source while
        # the unload thread is between source.close() and source.loop access.
        release_close.set()
        assert await asyncio.wait_for(unloading, 5)
        await asyncio.sleep(0)
        print('UNLOAD_RACE', json.dumps(dict(compacted=first.loop is None, other_active=other.active,
              other_task_done=other_task.done(), log=caplog.text)))
        assert not other.active and other_task.done(), 'one compacted source must not abort teardown of remaining sources'
    finally:
        release_close.set()
        release_send.set()
        await unloading
        await other.finish()
        await first.finish()
        await asyncio.gather(first_task, other_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_request_context_resets_after_ambiguous_attempt(manager, monkeypatch):
    lane = await h.sdk_lane(monkeypatch, fail_write=True)
    task = asyncio.create_task(lane.source.run())
    await asyncio.to_thread(h.h._execute_todo, lane.agent, 'ctx', {'id': 'a', 'content': 'ambiguous', 'status': 'pending'})
    await task
    assert lane.source.unknown
    assert wire.operation_admission.get() is None
    replacement = h.SocketWriter()
    async def connect(*args, **kwargs):
        return replacement.reader, replacement
    monkeypatch.setattr(wire.asyncio, 'open_connection', connect)
    result = await lane.adapter._bot.send_message(chat_id=-100, text='ordinary after failed optional request')
    assert result.message_id == 701 and replacement.writes
    await lane.source.finish()
    await lane.general.shutdown()
    await lane.updates.shutdown()

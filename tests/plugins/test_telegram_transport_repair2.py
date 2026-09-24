"""N1/N2: real SDK/pool and stdlib scheduling, synthetic non-network peers.

The TLS fixture uses real SSLProtocol closure, not an encrypted handshake.
Connection races retain asyncio's _connect_sock and staggered_race algorithms;
only DNS, socket allocation/connect and transport construction are substituted.
"""
import asyncio
import importlib.util
import os
from pathlib import Path
import socket
import threading
from types import SimpleNamespace

import httpcore
import httpx
import pytest

from plugins.platforms.telegram import adapter as adapter_module
from plugins.platforms.telegram import transport_admission as wire

spec = importlib.util.spec_from_file_location('repair2_harness', Path(__file__).with_name('test_telegram_live_todo_repair.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
manager = h.manager


class ReusableResponder(h.SocketWriter):
    def respond(self):
        super().respond()
        self.buffer = b''
        self.responded = False


@pytest.mark.asyncio
@pytest.mark.parametrize('tls', [False, True])
@pytest.mark.parametrize('reset', [False, True])
async def test_sdk_idle_reuse_and_closed_tls_expiry(manager, monkeypatch, tls, reset):
    lane = await h.sdk_lane(monkeypatch, mode='direct')
    loop = asyncio.get_running_loop()
    read_fd, write_fd = os.pipe()
    peers = []

    class PeerTransport(asyncio.Transport):
        def __init__(self, protocol, responder):
            self.protocol, self.responder = protocol, responder
            self.closed = False
            self.sock = SimpleNamespace(fileno=lambda: -1 if self.closed else read_fd,
                                        setsockopt=lambda *args: None)
            self.tls_protocol = None

        def get_extra_info(self, name, default=None):
            return self.sock if name == 'socket' else default

        def is_closing(self):
            return self.closed

        def write(self, data):
            self.responder.write(data)

        def lose(self, exc=None):
            if not self.closed:
                self.closed = True
                (self.tls_protocol or self.protocol).connection_lost(exc)

        def close(self):
            self.lose()

        abort = close

    class Writer(asyncio.StreamWriter):
        async def start_tls(self, context, **kwargs):
            # Preserve actual TLS extras and connection_lost, substitute only
            # the handshake/encryption so this test cannot open a socket.
            from asyncio.sslproto import SSLProtocol
            peer = self.transport
            protocol = SSLProtocol(loop, self._protocol, context, None,
                                   server_hostname=kwargs['server_hostname'], call_connection_made=False)
            protocol._transport = peer
            app = protocol._get_app_transport()
            app.write = peer.write
            app.close = peer.close
            app.abort = peer.abort
            peer.tls_protocol = protocol
            self._transport = app

    async def connect(*args, **kwargs):
        responder = ReusableResponder()
        protocol = asyncio.StreamReaderProtocol(responder.reader)
        peer = PeerTransport(protocol, responder)
        protocol.connection_made(peer)
        writer = Writer(peer, protocol, responder.reader, loop)
        peers.append((peer, writer))
        return responder.reader, writer

    monkeypatch.setattr(wire.asyncio, 'open_connection', connect)
    bot = (adapter_module.Application.builder().token('123:synthetic-repair2-only')
           .request(lane.general).get_updates_request(lane.updates)
           .base_url(('https' if tls else 'http') + '://bot-api.invalid/bot').build()).bot
    try:
        assert (await bot.send_message(chat_id=-100, text='before')).message_id == 701
        if reset:
            peer, writer = peers[0]
            peer.lose(ConnectionResetError('synthetic idle reset'))
            await asyncio.sleep(0)  # TLS forwards connection_lost to app via call_soon
            writer._protocol._get_close_waiter(None).exception()
            assert writer.is_closing() and not peer.responder.reader.at_eof()
            if tls:
                assert writer.get_extra_info('socket') is None
        for _ in range(2):
            assert (await bot.send_message(chat_id=-100, text='after')).message_id == 701
        assert len(peers) == (2 if reset else 1), 'healthy peers must pool; reset peers must expire once'
    finally:
        await lane.source.finish()
        await lane.general.shutdown()
        await lane.updates.shutdown()
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.asyncio
@pytest.mark.parametrize('state', ['eof', 'error', 'closing', 'no-socket', 'negative-fd', 'none-fd',
                                      'bad-fd', 'ready', 'idle'])
async def test_readability_contract_without_network(state):
    read_fd, write_fd = os.pipe()
    reader = asyncio.StreamReader()
    fd = read_fd
    if state == 'eof':
        reader.feed_eof()
    elif state == 'error':
        reader.set_exception(ConnectionResetError('fixture'))
    elif state == 'negative-fd':
        fd = -1
    elif state == 'none-fd':
        fd = None
    elif state == 'bad-fd':
        os.close(read_fd)
    elif state == 'ready':
        os.write(write_fd, b'x')
    writer = SimpleNamespace(is_closing=lambda: state == 'closing',
                             get_extra_info=lambda name: None if state == 'no-socket' else SimpleNamespace(fileno=lambda: fd))
    try:
        assert wire.AdmissionStream(reader, writer).get_extra_info('is_readable') is (state != 'idle')
    finally:
        if state != 'bad-fd':
            os.close(read_fd)
        os.close(write_fd)


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['direct', 'proxy'])
@pytest.mark.parametrize('outcome', ['success', 'cancel', 'timeout', 'revoked'])
async def test_dual_stack_loser_cleanup_and_cancellation(monkeypatch, route, outcome):
    loop = asyncio.get_running_loop()
    read_fd, write_fd = os.pipe()
    v6, v4 = '2001:db8::1', '192.0.2.1'
    sockets, attempts, resolutions = [], [], []
    both_started = asyncio.Event()
    admission_source = SimpleNamespace(active=True, lock=threading.RLock(),
                                       registration=SimpleNamespace(lock=threading.RLock()))
    admission_source.admitted = lambda: admission_source.active
    admission = wire.OperationAdmission(admission_source)

    class Socket:
        def __init__(self, family, type, proto):
            self.family, self.type, self.proto = family, type, proto
            self.closed = False
            self.bound = None
            self.options = []
            sockets.append(self)

        def setblocking(self, flag):
            assert flag is False

        def fileno(self):
            return -1 if self.closed else read_fd

        def bind(self, addr):
            self.bound = addr

        def setsockopt(self, *option):
            self.options.append(option)

        def close(self):
            self.closed = True

    async def resolve(address, **kwargs):
        resolutions.append(address)
        port = address[1]
        # Two leading IPv6 entries verify interleaving, not just racing.
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, '', (v6, port, 0, 0)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2001:db8::2', port, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', (v4, port))]

    async def connect(sock, address):
        attempts.append(address[0])
        if len(attempts) >= 2:
            both_started.set()
        if sock.family == socket.AF_INET6 or outcome in {'cancel', 'timeout'}:
            await asyncio.Event().wait()
        if outcome == 'revoked':
            admission_source.active = False

    class Transport(asyncio.Transport):
        def __init__(self, protocol, sock):
            self.protocol, self.sock = protocol, sock
            self.responder = ReusableResponder()
            self.responder.reader = protocol._stream_reader

        def get_extra_info(self, name, default=None):
            return self.sock if name == 'socket' else default

        def is_closing(self):
            return self.sock.closed

        def write(self, data):
            self.responder.write(data)

        def close(self):
            if not self.sock.closed:
                self.sock.close()
                self.protocol.connection_lost(None)

        abort = close

    async def make_transport(sock, factory, *args, **kwargs):
        protocol = factory()
        transport = Transport(protocol, sock)
        protocol.connection_made(transport)
        return transport, protocol

    monkeypatch.setattr(loop, '_ensure_resolved', resolve)
    monkeypatch.setattr(socket, 'socket', Socket)
    monkeypatch.setattr(loop, 'sock_connect', connect)
    monkeypatch.setattr(loop, '_create_connection_transport', make_transport)
    option = (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    client = None
    stream = None
    token = wire.operation_admission.set(admission)
    try:
        if route == 'direct':
            call = wire.AdmissionBackend().connect_tcp('dual.invalid', 443, timeout=3,
                                                        local_address='local.invalid', socket_options=[option])
        else:
            client = httpx.AsyncClient(transport=wire.AdmissionHTTPTransport(
                proxy='http://dual-proxy.invalid:8080', socket_options=[option]), timeout=3)
            call = client.get('http://bot-api.invalid/health')
        task = asyncio.create_task(call)
        if outcome == 'cancel':
            await asyncio.wait_for(both_started.wait(), 5)
            task.cancel()
        if outcome == 'success':
            result = await asyncio.wait_for(task, 5)
            if route == 'direct':
                stream = result
                assert sockets[1].bound == (v4, 0)
                assert sockets[1].options == [option]
            else:
                assert result.status_code == 200
                assert resolutions[0] == ('dual-proxy.invalid', 8080)
                # httpcore 1.0.9 omits options in BOTH proxy constructors.
                # Verify parity with the old HTTPX proxy pool, rather than
                # claiming that the backend receives options it never sees.
                before_control = len(sockets)
                control = httpcore.AsyncHTTPProxy('http://dual-proxy.invalid:8080',
                    network_backend=wire.AdmissionBackend(), socket_options=[option])
                try:
                    response = await control.request('GET', 'http://bot-api.invalid/health')
                    assert response.status == 200
                    assert sockets[before_control + 1].options == sockets[1].options == []
                finally:
                    await control.aclose()
            assert sockets[0].closed and not sockets[1].closed
        else:
            error = {'cancel': asyncio.CancelledError, 'revoked': wire.AdmissionRevoked,
                     'timeout': httpcore.ConnectTimeout if route == 'direct' else httpx.ConnectTimeout}[outcome]
            with pytest.raises(error):
                await asyncio.wait_for(task, 5)
            assert all(sock.closed for sock in sockets)
            assert admission.dispatched is False, 'DNS/connect cancellation is definitely unsent'
        assert attempts[:2] == [v6, v4]
    finally:
        wire.operation_admission.reset(token)
        if stream is not None:
            await stream.aclose()
        if client is not None:
            await client.aclose()
        os.close(read_fd)
        os.close(write_fd)
    assert all(sock.closed for sock in sockets)

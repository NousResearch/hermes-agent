"""Real loopback transport regression for a non-reading SSH-forward peer."""
import asyncio
import logging
from types import SimpleNamespace
import socket

import pytest


@pytest.fixture(autouse=True)
def _restore_uvicorn_loggers():
    saved = {
        name: (lg.level, lg.propagate, list(lg.handlers))
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")
        for lg in (logging.getLogger(name),)
    }
    yield
    for name, (level, propagate, handlers) in saved.items():
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            if h not in handlers:
                h.close()
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate


@pytest.mark.platforms('windows')
@pytest.mark.parametrize('close_header', [False, True])
def test_nonreading_http_peer_is_aborted(monkeypatch, close_header):
    import psutil
    from hermes_cli import web_server

    async def exercise():
        sent = asyncio.Event()
        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'x' * (8 * 1024 ** 2)})
            sent.set()

        # Zero send buffer is opt-in (default kernel buffering avoids blocking the loop on large replies);
        # this regression exercises the opt-in path.
        monkeypatch.setenv('HERMES_DASHBOARD_UNBUFFERED_SEND', '1')
        monkeypatch.setattr(web_server, 'app', SimpleNamespace(state=SimpleNamespace(auth_required=False)))
        monkeypatch.setattr(web_server, 'load_config', dict)
        config, server = web_server._build_uvicorn_server('127.0.0.1', 0)
        config.app = app
        config.lifespan = 'off'
        config.load()

        # Shorten the production timer, not the assertion's wall-clock budget.
        monkeypatch.setattr(config.http_protocol_class, 'write_stall_timeout', 0.1, raising=False)
        protocols = []
        def factory():
            protocol = config.http_protocol_class(config, server.server_state, {})
            protocols.append(protocol)
            return protocol
        own_process = psutil.Process()
        pool_before = own_process.memory_info().nonpaged_pool
        listener = await asyncio.get_running_loop().create_server(factory, '127.0.0.1', 0)
        client = socket.socket()
        client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        client.setblocking(False)
        try:
            await asyncio.get_running_loop().sock_connect(client, listener.sockets[0].getsockname())
            header = b'Connection: close\r\n' if close_header else b''
            await asyncio.get_running_loop().sock_sendall(client, b'GET / HTTP/1.1\r\nHost: localhost\r\n' + header + b'\r\n')
            await asyncio.wait_for(sent.wait(), 3)
            protocol = protocols[0]
            lost = asyncio.Event()
            original = protocol.connection_lost
            def record_lost(exc):
                original(exc)
                lost.set()
            protocol.connection_lost = record_lost

            await asyncio.wait_for(lost.wait(), 3)
            assert not server.server_state.connections
            assert protocol.transport.get_write_buffer_size() == 0
            # A completed/closed transport alone can still retain the full body in AfdB.
            assert own_process.memory_info().nonpaged_pool - pool_before < 1024 ** 2
        finally:
            client.close()
            for protocol in protocols:
                protocol.transport.abort()
            listener.close()
            await listener.wait_closed()
    asyncio.run(exercise())


def test_healthy_http_reader_receives_complete_response(monkeypatch):
    from hermes_cli import web_server

    async def exercise():
        body = b'healthy-reader' * 100000
        async def app(scope, receive, send):
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-length', str(len(body)).encode())]})
            await send({'type': 'http.response.body', 'body': body})
        # Zero send buffer is opt-in (default kernel buffering avoids blocking the loop on large replies);
        # this regression exercises the opt-in path.
        monkeypatch.setenv('HERMES_DASHBOARD_UNBUFFERED_SEND', '1')
        monkeypatch.setattr(web_server, 'app', SimpleNamespace(state=SimpleNamespace(auth_required=False)))
        monkeypatch.setattr(web_server, 'load_config', dict)
        config, server = web_server._build_uvicorn_server('127.0.0.1', 0, ssh_isolated=True)
        config.app = app
        config.lifespan = 'off'
        config.load()
        listener = await asyncio.get_running_loop().create_server(
            lambda: config.http_protocol_class(config, server.server_state, {}), '127.0.0.1', 0)
        reader, writer = await asyncio.open_connection(*listener.sockets[0].getsockname())
        try:
            writer.write(b'GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), 10)
            headers, received = response.split(b'\r\n\r\n', 1)
            assert headers.startswith(b'HTTP/1.1 200')
            assert received == body
        finally:
            writer.close()
            await writer.wait_closed()
            listener.close()
            await listener.wait_closed()
    asyncio.run(exercise())


def test_websocket_upgrade_still_exchanges_messages(monkeypatch):
    from hermes_cli import web_server
    from websockets.asyncio.client import connect

    async def exercise():
        async def app(scope, receive, send):
            assert scope['type'] == 'websocket'
            assert (await receive())['type'] == 'websocket.connect'
            await send({'type': 'websocket.accept'})
            message = await receive()
            await send({'type': 'websocket.send', 'text': message['text']})
            await send({'type': 'websocket.close', 'code': 1000})
        # Zero send buffer is opt-in (default kernel buffering avoids blocking the loop on large replies);
        # this regression exercises the opt-in path.
        monkeypatch.setenv('HERMES_DASHBOARD_UNBUFFERED_SEND', '1')
        monkeypatch.setattr(web_server, 'app', SimpleNamespace(state=SimpleNamespace(auth_required=False)))
        monkeypatch.setattr(web_server, 'load_config', dict)
        config, server = web_server._build_uvicorn_server('127.0.0.1', 0, ssh_isolated=True)
        config.app = app
        config.lifespan = 'off'
        config.load()
        listener = await asyncio.get_running_loop().create_server(
            lambda: config.http_protocol_class(config, server.server_state, {}), '127.0.0.1', 0)
        try:
            port = listener.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}/') as websocket:
                await websocket.send('upgrade-control')
                assert await asyncio.wait_for(websocket.recv(), 3) == 'upgrade-control'
        finally:
            for protocol in list(server.server_state.connections):
                protocol.transport.abort()
            listener.close()
            await listener.wait_closed()
    asyncio.run(exercise())


def test_resumed_writer_cancels_stall_timer():
    from unittest.mock import Mock
    from uvicorn import Config
    from uvicorn.server import ServerState
    from hermes_cli.web_server_slow_client import BoundedH11Protocol

    async def exercise():
        async def app(scope, receive, send):
            pass
        protocol = BoundedH11Protocol(Config(app, log_config=None), ServerState(), {})
        transport = Mock()
        transport.get_extra_info.return_value = None
        protocol.connection_made(transport)
        protocol.pause_writing()
        timer = protocol._write_stall_timer
        protocol.resume_writing()
        assert timer.cancelled()
        assert protocol._write_stall_timer is None
        assert not protocol.flow.write_paused
        protocol.pause_writing()
        timer = protocol._write_stall_timer
        protocol.connection_lost(None)
        assert timer.cancelled()
        transport.abort.assert_not_called()
    asyncio.run(exercise())

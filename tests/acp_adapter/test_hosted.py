"""Exercise the real socket/SDK path, including isolation and rejected peers/headers."""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from acp_adapter.hosted import HostedACPServer
from acp_adapter.relay import HEADER_VERSION


@pytest.mark.asyncio
async def test_cleanup_aborts_before_protocol_close_and_still_closes_agent():
    from acp_adapter.hosted import _close_projection
    calls = []

    async def protocol_close():
        assert calls == ['abort']
        calls.append('protocol')
        raise OSError('peer stopped reading')

    async def agent_close():
        calls.append('agent')

    async def writer_closed():
        calls.append('writer')

    writer = SimpleNamespace(transport=SimpleNamespace(abort=lambda: calls.append('abort')),
                             wait_closed=writer_closed)
    connection = SimpleNamespace(_conn=SimpleNamespace(close=protocol_close))
    await _close_projection(writer, connection, SimpleNamespace(aclose=agent_close))
    assert calls == ['abort', 'protocol', 'agent', 'writer']


@pytest.mark.asyncio
async def test_stop_cancels_clients_before_waiting_for_server_connections():
    # Python 3.12+ wait_closed waits for clients as well as the listening socket.
    closed = asyncio.Event()
    active = asyncio.Event()

    async def client():
        active.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def wait_closed():
        await closed.wait()

    server = HostedACPServer(SimpleNamespace(), '.')
    server.server = SimpleNamespace(close=lambda: None, wait_closed=wait_closed)
    task = asyncio.create_task(client())
    server.tasks.add(task)
    await active.wait()
    try:
        await asyncio.wait_for(server.stop(), 5)
        assert task.cancelled()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def socket_home():
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory(prefix='acp-') as root:
        yield Path(root)


@pytest.mark.asyncio
async def test_socket_handshake_initialize_isolation_and_cleanup(socket_home, monkeypatch):
    from acp_adapter import auth
    monkeypatch.setattr(auth, 'build_auth_methods', list)
    runner = SimpleNamespace()
    server = HostedACPServer(runner, socket_home)
    assert await server.start()
    assert server.path.stat().st_mode & 0o777 == 0o600
    writers = []
    try:
        for index in range(2):
            reader, writer = await asyncio.open_unix_connection(str(server.path))
            writers.append(writer)
            header = {'hermes_acp_relay': HEADER_VERSION, 'env': {'HERMES_YOLO_MODE': str(index)},
                      'cwd': str(socket_home)}
            writer.write((json.dumps(header) + '\n' + json.dumps({
                'jsonrpc': '2.0', 'id': index, 'method': 'initialize',
                'params': {'protocolVersion': 1, 'clientCapabilities': {}}}) + '\n').encode())
            await writer.drain()
            response = json.loads(await asyncio.wait_for(reader.readline(), 10))
            assert response['id'] == index
            assert response['result']['agentInfo']['name'] == 'hermes-agent'
        assert runner._idle_exit_clients == 2
    finally:
        await server.stop()
        for writer in writers:
            writer.close()
            await writer.wait_closed()
    assert not server.path.exists()
    assert not server.tasks
    assert runner._idle_exit_clients == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('header', [b'{}\n', b'[]\n', b'x' * (64 * 1024 + 1) + b'\n'])
async def test_invalid_handshake_is_closed(socket_home, header):
    server = HostedACPServer(SimpleNamespace(), socket_home)
    assert await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(server.path))
        writer.write(header)
        await writer.drain()
        assert await asyncio.wait_for(reader.read(), 10) == b''
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_wrong_uid_is_closed_before_handshake(socket_home, monkeypatch):
    from hermes_cli import gateway_runtime_discovery
    monkeypatch.setattr(gateway_runtime_discovery, 'socket_peer_uid', lambda sock: os.getuid() + 1)
    server = HostedACPServer(SimpleNamespace(), socket_home)
    assert await server.start()
    try:
        reader, writer = await asyncio.open_unix_connection(str(server.path))
        assert await asyncio.wait_for(reader.read(), 10) == b''
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()

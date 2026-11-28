"""A detached worker handshake never confirms a cancelled or rejected attachment."""

import asyncio
import os
import socket
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.managed_worker import encode_frame
from gateway import session_detached_workers as detached
from persistence.core.errors import AuthorityRefusal


@pytest.mark.platforms('linux')
def test_pre_hello_eof_retries_same_identity(monkeypatch):
    record = dict(pid=os.getpid(), birth=1.0, execution_id='fixture', generation=3)
    first, dead = socket.socketpair()
    replacement, worker = socket.socketpair()
    with first, dead, replacement, worker:
        dead.close()
        worker.sendall(encode_frame({'type': 'hello', **record}))
        connections = Mock(side_effect=[first, replacement])
        monkeypatch.setattr(detached, '_connect_spared_worker', connections)
        monkeypatch.setattr(detached, '_worker_alive', lambda *_args, **_kwargs: True)
        sock, _ = detached._handshake(record)
        assert sock is replacement and first.fileno() == -1
        assert connections.call_count == 2


@pytest.mark.platforms('linux')
@pytest.mark.parametrize('field', ['pid', 'uid'])
def test_peer_credentials_refuse_before_hello(monkeypatch, field):
    record = dict(pid=os.getpid(), birth=1.0, execution_id='fixture', generation=3)
    pid = record['pid'] + (field == 'pid')
    uid = os.getuid() + (field == 'uid')
    peer = Mock(spec=socket.socket)
    peer.getsockopt.return_value = struct.pack('3i', pid, uid, 0)
    connect = Mock(return_value=peer)
    monkeypatch.setattr(detached, '_connect_spared_worker', connect)
    with pytest.raises(AuthorityRefusal, match='permission_denied'):
        detached._handshake(record)
    connect.assert_called_once()
    peer.close.assert_called_once()
    peer.recv.assert_not_called()
    peer.sendall.assert_not_called()


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_cancelled_handshake_closes_without_confirmation(monkeypatch):
    record = dict(pid=os.getpid(), birth=1.0, execution_id='fixture', generation=3)
    reached = asyncio.Event()
    loop = asyncio.get_running_loop()
    local, worker = socket.socketpair()
    with local, worker:
        def connect(*_):
            loop.call_soon_threadsafe(reached.set)
            return local

        monkeypatch.setattr(detached, '_connect_spared_worker', connect)
        owner = SimpleNamespace(epoch=2, _managed_reattaching={'s': False}, pending_stops={})
        task = asyncio.create_task(detached.reattach(owner, SimpleNamespace(session_id='s'),
                                                   {'generation': 3}, record, Mock()))
        try:
            await asyncio.wait_for(reached.wait(), 4)
            task.cancel()
            worker.sendall(encode_frame({'type': 'hello', **record}))
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 4)
            worker.settimeout(2)
            assert worker.recv(1024) == b''
            assert local.fileno() == -1
            assert not getattr(owner, '_managed_workers', {})
        finally:
            worker.close()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.platforms('linux')
def test_linux_detach_requires_explicit_opt_in():
    assert detached.detach_enabled(SimpleNamespace(config=lambda: {'gateway': {'detached_workers': True}}))
    assert not detached.detach_enabled(SimpleNamespace(config=lambda: {'gateway': {'detached_workers': 1}}))
    assert not detached.detach_enabled(SimpleNamespace(config=dict))


@pytest.mark.platforms('not linux')
def test_detach_refused_off_linux():
    assert not detached.detach_enabled(SimpleNamespace(config=lambda: {'gateway': {'detached_workers': True}}))

"""An orphaned worker's fallback listener directory exists only while it waits for an owner.

``socket_path`` names ``hmw-<random>/w.sock`` under a short root without creating it; the worker
creates that directory 0700 when it starts listening and removes it once reattached, or just before
it ends itself when no owner came within the grace. An existing directory it did not make (the
profile's outbox) must be private and is left in place.
"""
import contextlib
import io
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.managed_worker import encode_frame, read_frame
from agent.managed_worker_detach.channel import DetachableChannel
from tests.waiting import wait_until

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def base():
    path = Path(tempfile.mkdtemp(prefix='hdl-', dir='/tmp'))  # AF_UNIX: keep the path short
    try:
        yield path
    finally:
        shutil.rmtree(path)


@contextlib.contextmanager
def _listening(path):
    channel = DetachableChannel(io.BytesIO(), str(path), {'pid': os.getpid()})
    # Orphaned without redirecting the pytest process's stdio or arming its grace timer.
    channel.orphaned = True
    channel.timer = SimpleNamespace(cancel=lambda: None)
    listener = threading.Thread(target=channel._listen, name='fixture-listener', daemon=True)
    listener.start()
    try:
        yield channel, listener
    finally:
        if listener.is_alive():
            with contextlib.suppress(OSError, EOFError):
                client, _ = _reattach(path, 0)
                client.close()
        listener.join(10)


def _reattach(path, epoch, before_reply=None):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    deadline = time.monotonic() + 10
    while True:
        try:
            client.connect(str(path))
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() >= deadline:
                client.close()
                raise
            time.sleep(0.01)
    try:
        with client.makefile('rb') as reader:
            hello = read_frame(reader)
        if before_reply is not None:
            before_reply()
        client.sendall(encode_frame({'type': 'reattached', 'epoch': epoch}))
        return client, hello
    except BaseException:
        client.close()
        raise


@pytest.mark.platforms('linux')
def test_fallback_directory_exists_only_until_reattached(base):
    path = base / 'hmw-0123456789abcdef' / 'w.sock'
    with _listening(path) as (channel, listener):
        def check_mode():
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        client, hello = _reattach(path, 2, before_reply=check_mode)
        with client:
            listener.join(10)
            assert not listener.is_alive() and hello == {'type': 'hello', 'pid': os.getpid()}
            assert channel.epoch == 2
            assert not path.parent.exists()


@pytest.mark.platforms('linux')
def test_existing_private_outbox_is_used_and_kept(base):
    outbox = base / 'outbox'
    outbox.mkdir(mode=0o700)
    path = outbox / 'control.sock'
    with _listening(path) as (_channel, listener):
        client, _hello = _reattach(path, 2)
        with client:
            listener.join(10)
            assert not listener.is_alive()
            assert outbox.is_dir() and not path.exists()


@pytest.mark.platforms('linux')
def test_a_directory_another_mode_left_is_refused(base):
    parent = base / 'hmw-0123456789abcdef'
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    channel = DetachableChannel(io.BytesIO(), str(parent / 'w.sock'), {'pid': os.getpid()})
    refused = []

    def listen():
        try:
            channel._listen()
        except PermissionError as exc:
            refused.append(exc)
    listener = threading.Thread(target=listen, name='fixture-listener', daemon=True)
    listener.start()
    listener.join(10)
    assert refused and not listener.is_alive()
    assert os.listdir(parent) == []


# An orphaned worker nobody reattaches: it publishes its listener, then the grace timer ends it.
ORPHAN = """import os, sys, threading, time
import agent.managed_worker as worker
import agent.managed_worker_detach.channel as detach
original = worker._die_with_owner
def arm_after_rendezvous(seconds, before):
    def arm():
        sys.stdin.buffer.read(1)
        original(3, before=before)
    threading.Thread(target=arm, daemon=True).start()
    return None
worker._die_with_owner = arm_after_rendezvous
detach.DetachableChannel(open(os.devnull, 'wb'), sys.argv[1], {'pid': os.getpid()}).orphan()
time.sleep(60)
"""

@pytest.mark.platforms('linux')
def test_grace_expiry_removes_the_listener_before_the_worker_ends(base):
    path = base / 'hmw-0123456789abcdef' / 'w.sock'
    with subprocess.Popen([sys.executable, '-c', ORPHAN, str(path)], cwd=ROOT, start_new_session=True,
                          stdin=subprocess.PIPE, stdout=subprocess.DEVNULL) as worker:
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(10)
            try:
                def connect():
                    try:
                        client.connect(str(path))
                        return True
                    except (FileNotFoundError, ConnectionRefusedError):
                        return False
                wait_until(connect, timeout=10, what='the listener to accept connections')
                with client.makefile('rb') as reader:
                    assert read_frame(reader)['type'] == 'hello'
                client.sendall(encode_frame({'type': 'invalid'}))
            finally:
                client.close()
            worker.stdin.write(b'x')
            worker.stdin.flush()
            assert worker.wait(timeout=60) != 0
        finally:
            if worker.poll() is None:
                worker.kill()
    assert not path.parent.exists()


@pytest.mark.platforms('linux')
def test_consecutive_handoffs_do_not_lose_listener_directory(base, monkeypatch):
    path = base / 'hmw-0123456789abcdef' / 'w.sock'
    channel = DetachableChannel(io.BytesIO(), str(path), {'pid': os.getpid()})
    channel.orphaned = True
    channel.timer = SimpleNamespace(cancel=lambda: None)
    cleanup_entered, allow_cleanup = threading.Event(), threading.Event()
    original = channel._release_directory
    calls = 0

    def paused_cleanup():
        nonlocal calls
        calls += 1
        if calls == 1:
            cleanup_entered.set()
            assert allow_cleanup.wait(10)
        original()

    monkeypatch.setattr(channel, '_release_directory', paused_cleanup)
    first = threading.Thread(target=channel._listen, daemon=True)
    first.start()
    clients = []
    second = None
    try:
        client, hello = _reattach(path, 2)
        clients.append(client)
        assert hello == {'type': 'hello', 'pid': os.getpid()}
        assert cleanup_entered.wait(10)
        attempted = threading.Event()
        acquired = []
        orphaned = threading.Event()

        def next_owner():
            locked = channel.lock.acquire(blocking=False)
            acquired.append(locked)
            if locked:
                channel.lock.release()
            attempted.set()
            with channel.lock:
                channel.orphaned = True
                channel.timer = SimpleNamespace(cancel=lambda: None)
                orphaned.set()
            channel._listen()

        second = threading.Thread(target=next_owner, daemon=True)
        second.start()
        assert attempted.wait(10)
        # The first handoff must not become visible while its directory is still owned.
        assert acquired == [False]
        assert path.parent.is_dir()
        allow_cleanup.set()
        first.join(10)
        assert not first.is_alive() and orphaned.wait(10)
        client, hello = _reattach(path, 3)
        clients.append(client)
        second.join(10)
        assert not second.is_alive() and hello == {'type': 'hello', 'pid': os.getpid()}
        assert channel.epoch == 3 and channel.attachments == 2
        assert not path.parent.exists()
    finally:
        allow_cleanup.set()
        for client in clients:
            client.close()
        for listener in (first, second):
            if listener is not None and listener.is_alive():
                with contextlib.suppress(OSError, EOFError):
                    client, _ = _reattach(path, 0)
                    client.close()
                listener.join(10)

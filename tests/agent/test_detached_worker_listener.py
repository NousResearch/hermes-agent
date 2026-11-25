"""An orphaned worker's fallback listener directory exists only while it waits for an owner.

``socket_path`` names ``hmw-<random>/w.sock`` under a short root without creating it; the worker
creates that directory 0700 when it starts listening and removes it once reattached, or just before
it ends itself when no owner came within the grace. An existing directory it did not make (the
profile's outbox) must be private and is left in place.
"""
import io
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.managed_worker import encode_frame, read_frame
from agent.managed_worker_detach import DetachableChannel
from tests.waiting import wait_until

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def base():
    path = Path(tempfile.mkdtemp(prefix='hdl-', dir='/tmp'))  # AF_UNIX: keep the path short
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _listening(path):
    channel = DetachableChannel(io.BytesIO(), str(path), {'pid': os.getpid()})
    # Orphaned without redirecting the pytest process's stdio or arming its grace timer.
    channel.orphaned = True
    channel.timer = SimpleNamespace(cancel=lambda: None)
    listener = threading.Thread(target=channel._listen, name='fixture-listener', daemon=True)
    listener.start()
    return channel, listener


def _reattach(path, epoch):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(str(path))
    hello = read_frame(client.makefile('rb'))
    client.sendall(encode_frame({'type': 'reattached', 'epoch': epoch}))
    return client, hello


@pytest.mark.platforms('linux')
def test_fallback_directory_exists_only_until_reattached(base):
    path = base / 'hmw-0123456789abcdef' / 'w.sock'
    channel, listener = _listening(path)
    wait_until(path.exists, timeout=10, what='the orphaned worker to publish its listener')
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    client, hello = _reattach(path, 2)
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
    _channel, listener = _listening(path)
    wait_until(path.exists, timeout=10, what='the listener in the outbox')
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
ORPHAN = """import os, sys, time
import agent.managed_worker_detach as detach
detach.ADOPT_GRACE_SECONDS = 3
detach.DetachableChannel(open(os.devnull, 'wb'), sys.argv[1], {'pid': os.getpid()}).orphan()
time.sleep(60)
"""


@pytest.mark.platforms('linux')
def test_grace_expiry_removes_the_listener_before_the_worker_ends(base):
    path = base / 'hmw-0123456789abcdef' / 'w.sock'
    with subprocess.Popen([sys.executable, '-c', ORPHAN, str(path)], cwd=ROOT, start_new_session=True,
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL) as worker:
        try:
            wait_until(path.exists, timeout=30, what='the orphaned worker to publish its listener')
            assert worker.wait(timeout=60) != 0
        finally:
            if worker.poll() is None:
                worker.kill()
    assert not path.parent.exists()

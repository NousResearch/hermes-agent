"""The listener path a detached reservation records fits ``sun_path``, on every fallback.

A long profile home pushes the outbox socket past the bound, so the path moves to a randomly named
directory under a short root. The recorded path must still fit, measured in filesystem bytes, and
be bindable; the owner creates nothing there (the worker makes that directory only while it waits
for a new owner, ``tests/agent/test_detached_worker_listener.py``).
"""
import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest

from gateway import session_detached_workers as detached

EXECUTION = 'admission-worker:' + 'a' * 32
TAIL = len('/hmw-0123456789abcdef/w.sock')


@pytest.fixture
def base():
    # AF_UNIX paths: the per-test tmp_path alone can already be most of sun_path.
    path = Path(tempfile.mkdtemp(prefix='hsp-', dir='/tmp'))
    try:
        yield path
    finally:
        shutil.rmtree(path)


def _size(path):
    return len(os.fsencode(str(path)))


def _bind_as_the_worker_would(path):
    parent = os.path.dirname(path)
    os.mkdir(parent, 0o700)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(path)
    finally:
        server.close()
        os.unlink(path)
        os.rmdir(parent)


def _private_dir(path):
    path.mkdir(mode=0o700)
    return path


def _fallback(base, monkeypatch, runtime, temp=None):
    if runtime is None:
        monkeypatch.delenv('XDG_RUNTIME_DIR', raising=False)
    else:
        monkeypatch.setenv('XDG_RUNTIME_DIR', str(runtime))
    if temp is not None:
        monkeypatch.setattr(tempfile, 'tempdir', str(temp))
    return detached.socket_path(base / ('h' * 120), EXECUTION)


def _under(path, root):
    return path is not None and Path(path).parent.parent == root and _size(path) <= detached._MAX_UNIX_PATH


@pytest.mark.platforms('linux')
def test_short_home_keeps_the_outbox_listener(base):
    path = detached.socket_path(base, EXECUTION)
    assert path == str(base / 'worker-outboxes' / EXECUTION.replace(':', '-') / 'control.sock')
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    with server:
        server.bind(path)


@pytest.mark.platforms('linux')
def test_long_home_uses_the_runtime_dir_and_creates_nothing(base, monkeypatch):
    runtime = _private_dir(base / 'run')
    path = _fallback(base, monkeypatch, runtime)
    assert _under(path, runtime)
    assert os.listdir(runtime) == []
    _bind_as_the_worker_would(path)


@pytest.mark.platforms('linux')
def test_long_runtime_dir_falls_through_to_a_root_that_fits(base, monkeypatch):
    """The audit's case: a 96-byte private ``XDG_RUNTIME_DIR`` gave a 116-byte path."""
    runtime = _private_dir(base / ('r' * (95 - _size(base))))
    assert _size(runtime) == 96 and _size(runtime) + TAIL > detached._MAX_UNIX_PATH
    short = _private_dir(base / 't')
    path = _fallback(base, monkeypatch, runtime, temp=short)
    assert _under(path, short)
    assert os.listdir(runtime) == [] and os.listdir(short) == []
    _bind_as_the_worker_would(path)


@pytest.mark.platforms('linux')
def test_multibyte_root_is_measured_in_bytes(base, monkeypatch):
    """Characters undercount a non-ASCII root: the bound is on the bytes ``bind`` receives."""
    runtime = _private_dir(base / ('é' * (100 - TAIL - len(str(base)) - 1)))
    assert len(str(runtime)) + TAIL <= detached._MAX_UNIX_PATH < _size(runtime) + TAIL
    short = _private_dir(base / 't')
    path = _fallback(base, monkeypatch, runtime, temp=short)
    assert _under(path, short)
    _bind_as_the_worker_would(path)


@pytest.mark.platforms('linux')
def test_long_scratch_dir_without_runtime_dir_uses_the_short_root(base, monkeypatch):
    """A gateway's TMPDIR is its profile's scratch dir, as deep as the home: the last root still fits."""
    deep = _private_dir(base / ('d' * (95 - _size(base))))
    short = _private_dir(base / 't')
    monkeypatch.setattr(detached, '_LAST_RESORT_ROOT', str(short))
    path = _fallback(base, monkeypatch, None, temp=deep)
    assert _under(path, short)
    _bind_as_the_worker_would(path)


@pytest.mark.platforms('linux')
def test_unusable_temp_dir_does_not_hide_a_root_that_fits(base, monkeypatch):
    """No usable temp directory (``gettempdir`` raises) must not stop the other roots being tried."""
    runtime = _private_dir(base / 'run')

    def no_temp_dir():
        raise FileNotFoundError('fixture: no usable temporary directory')
    monkeypatch.setattr(tempfile, 'gettempdir', no_temp_dir)
    assert _under(_fallback(base, monkeypatch, runtime), runtime)
    short = _private_dir(base / 't')
    monkeypatch.setattr(detached, '_LAST_RESORT_ROOT', str(short))
    deep = _private_dir(base / ('d' * (95 - _size(base))))
    assert _under(_fallback(base, monkeypatch, deep), short)


@pytest.mark.platforms('linux')
@pytest.mark.skipif(getattr(os, 'geteuid', lambda: None)() == 0, reason='root writes through any mode')
def test_unwritable_root_is_skipped(base, monkeypatch):
    runtime = _private_dir(base / 'run')
    runtime.chmod(0o500)
    short = _private_dir(base / 't')
    try:
        assert _under(_fallback(base, monkeypatch, runtime, temp=short), short)
    finally:
        runtime.chmod(0o700)


@pytest.mark.platforms('linux')
def test_no_root_that_fits_records_no_listener(base, monkeypatch):
    """Every root too deep: no listener (the turn runs undetached) and nothing created."""
    deep = _private_dir(base / ('d' * (95 - _size(base))))
    monkeypatch.setattr(detached, '_LAST_RESORT_ROOT', str(deep))
    assert _fallback(base, monkeypatch, deep, temp=deep) is None
    assert os.listdir(deep) == []

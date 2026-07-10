"""Invariant tests for utils.durable_fsync (macOS F_FULLFSYNC durability)."""

import os
import sys

import pytest

import utils

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="fcntl / F_FULLFSYNC monkeypatching requires a POSIX platform",
)


@pytest.fixture
def real_fd(tmp_path):
    path = tmp_path / "payload.bin"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT, 0o600)
    os.write(fd, b"payload")
    try:
        yield fd
    finally:
        os.close(fd)


def test_full_fsync_used_on_darwin(monkeypatch, real_fd):
    """On macOS the durable path must issue F_FULLFSYNC, not plain fsync."""
    import fcntl

    fcntl_calls = []
    fsync_calls = []
    monkeypatch.setattr(utils.sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, cmd: fcntl_calls.append((fd, cmd)))
    monkeypatch.setattr(utils.os, "fsync", lambda fd: fsync_calls.append(fd))

    utils.durable_fsync(real_fd)

    assert fcntl_calls == [(real_fd, 51)]
    assert fsync_calls == []


def test_falls_back_to_fsync_when_full_fsync_unsupported(monkeypatch, real_fd):
    """A filesystem that rejects F_FULLFSYNC must degrade to os.fsync."""
    import fcntl

    fsync_calls = []

    def _reject(fd, cmd):
        raise OSError("F_FULLFSYNC not supported")

    monkeypatch.setattr(utils.sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", _reject)
    monkeypatch.setattr(utils.os, "fsync", lambda fd: fsync_calls.append(fd))

    utils.durable_fsync(real_fd)

    assert fsync_calls == [real_fd]


def test_plain_fsync_off_darwin(monkeypatch, real_fd):
    """Non-macOS platforms use os.fsync directly and never touch F_FULLFSYNC."""
    fsync_calls = []
    monkeypatch.setattr(utils.sys, "platform", "linux")
    monkeypatch.setattr(utils.os, "fsync", lambda fd: fsync_calls.append(fd))

    utils.durable_fsync(real_fd)

    assert fsync_calls == [real_fd]

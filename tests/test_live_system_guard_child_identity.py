"""Regression tests for stable child identity in the live-system guard."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest


def test_spawn_wrappers_record_pid_and_creation_time(_live_system_guard):
    sync_child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)

    async def _spawn_async_child():
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import sys; sys.stdin.read()", stdin=asyncio.subprocess.PIPE
        )
        child.terminate()
        await child.wait()
        return child.pid

    try:
        sync_identity = next(
            identity
            for identity in _live_system_guard["owned_children"]
            if identity[0] == sync_child.pid
        )
        assert sync_identity[1] > 0
        async_pid = asyncio.run(_spawn_async_child())
        async_identity = next(
            identity
            for identity in _live_system_guard["owned_children"]
            if identity[0] == async_pid
        )
        assert async_identity[1] > 0
    finally:
        sync_child.terminate()
        sync_child.wait()


def test_signal_guard_uses_owned_identity_and_fails_closed_on_races(
    _live_system_guard, monkeypatch
):
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    child.terminate()
    child.wait(timeout=10)
    owned_pid, owned_created = next(
        identity
        for identity in _live_system_guard["owned_children"]
        if identity[0] == child.pid
    )
    delivered = []
    _live_system_guard["signal_targets"]["kill"] = (
        lambda pid, sig, *args, **kwargs: delivered.append((pid, sig))
    )
    foreign_parent = SimpleNamespace(pid=owned_pid + 100_000)

    class _Process:
        def __init__(self, pid, created, parents=()):
            self.pid = pid
            self._created = created
            self._parents = parents

        def create_time(self):
            return self._created

        def parents(self):
            if isinstance(self._parents, BaseException):
                raise self._parents
            return list(self._parents)

    # A recorded child remains ours after it has exited/reparented.
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: _Process(pid, owned_created, (foreign_parent,)),
    )
    os.kill(owned_pid, signal.SIGTERM)
    assert delivered == [(owned_pid, signal.SIGTERM)]

    # The same numeric PID with a different birth time is a different process.
    delivered.clear()
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: _Process(pid, owned_created + 1, (foreign_parent,)),
    )
    with pytest.raises(RuntimeError, match="outside the test process subtree"):
        os.kill(owned_pid, signal.SIGTERM)
    assert delivered == []

    # An unrecorded process with no test ancestor remains foreign.
    foreign_pid = owned_pid + 200_000
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda pid: _Process(pid, owned_created + 2, (foreign_parent,)),
    )
    with pytest.raises(RuntimeError, match="outside the test process subtree"):
        os.kill(foreign_pid, signal.SIGTERM)
    assert delivered == []

    # If the process disappears between identity and ancestry reads, report the
    # ordinary stale-PID outcome without ever invoking the signal target.
    calls = 0

    def _disappearing_process(pid):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Process(pid, owned_created + 3, RuntimeError("process vanished"))
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", _disappearing_process)
    with pytest.raises(ProcessLookupError):
        os.kill(foreign_pid, signal.SIGTERM)
    assert delivered == []

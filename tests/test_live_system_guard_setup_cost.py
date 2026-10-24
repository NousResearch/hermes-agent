"""The live-system guard must not walk the process table at every test's setup.

``_live_system_guard`` (tests/_fixtures/live_system_guard.py) is autouse, so
anything it does at setup is paid by every test in the suite. Its snapshot of
the test process's children only serves the tests that deliver a signal, so it
is taken at the first guarded kill, not at setup.

The counter is installed by a MODULE-scoped fixture so it is in place before
the function-scoped guard sets up; a function-scoped autouse fixture here would
run after the guard and see nothing.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import types

import psutil
import pytest

_WALKS: list[int] = []


@pytest.fixture(scope="module", autouse=True)
def _count_children_walks():
    real_children = psutil.Process.children

    def _counting_children(self, *args, **kwargs):
        _WALKS.append(self.pid)
        return real_children(self, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(psutil.Process, "children", _counting_children)
    yield
    patch.undo()


@pytest.fixture(autouse=True)
def _fresh_count():
    # Teardown runs after the test body and before the next test's guard setup,
    # so each test starts with only its own setup's walks recorded.
    yield
    _WALKS.clear()


def _guard_is_active() -> bool:
    # The guard replaces os.kill with a Python function; unguarded it is a C builtin.
    return not isinstance(os.kill, types.BuiltinFunctionType)


def test_guard_setup_does_not_walk_the_process_table():
    assert _guard_is_active()
    assert _WALKS == []


def test_the_first_guarded_kill_still_takes_the_snapshot():
    """Positive control: the counter does see the guard's walk once a kill reaches it."""
    assert _guard_is_active()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        os.kill(child.pid, signal.SIGTERM)
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
    assert os.getpid() in _WALKS


def test_a_mocked_psutil_cannot_widen_the_snapshot(monkeypatch):
    """The lazy snapshot is taken after the test body runs, so it must not go through a mock.

    A fake ``psutil.Process`` whose ``children()`` names a foreign PID would put
    that PID on the allow-list at the first kill and let the signal reach the OS.
    """
    assert _guard_is_active()
    foreign_pid = 424242
    while psutil.pid_exists(foreign_pid):
        foreign_pid += 1

    class _FakeProcess:
        def __init__(self, pid=None):
            self.pid = pid

        def children(self, recursive=False):
            return [types.SimpleNamespace(pid=foreign_pid)]

        def parents(self):
            return []

    monkeypatch.setattr(psutil, "Process", _FakeProcess)
    with pytest.raises(RuntimeError, match="live-system guard"):
        os.kill(foreign_pid, signal.SIGTERM)

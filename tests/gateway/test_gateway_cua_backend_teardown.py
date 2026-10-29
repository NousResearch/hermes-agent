"""The gateway's hard-exit path must stop the cua-driver backends before ``os._exit``.

``gateway/run.py::_exit_after_graceful_shutdown`` ends in ``os._exit``, which bypasses
``atexit`` — and the computer-use teardown is an ``atexit`` hook
(``tools/computer_use/tool.py::_shutdown_backend_atexit``). On macOS the private embedded
daemon is launched through LaunchServices, so it is a child of launchd and not of Hermes:
nothing else reaps it, and it keeps serving on its socket with
``--dangerously-bypass-approvals`` and the CuaDriver TCC grants intact (#126364).

These assert the exit path really calls the teardown, not a snapshot of the source.
"""

from __future__ import annotations

import inspect

import pytest

from gateway import run as gateway_run
from gateway import shutdown_watchdog
from tools.computer_use import tool as cu_tool


class _Recorder:
    """Stands in for the computer-use tool module: records the teardown call."""

    def __init__(self) -> None:
        self.calls = 0

    def stop_all_computer_use_backends(self) -> None:
        self.calls += 1


def test_exit_path_calls_cua_backend_teardown():
    """``os._exit`` skips atexit, so the exit path must stop the backends itself.

    The funnel delegates to the shared ``stop_cua_backends_quietly`` seam rather than calling
    the tool module directly, so the assert names the seam (the chain down to
    ``stop_all_computer_use_backends`` is pinned by ``test_teardown_seam_reaches_the_tool``).
    """
    source = inspect.getsource(gateway_run._exit_after_graceful_shutdown)
    assert "stop_cua_backends_quietly" in source


def test_exit_path_teardown_runs_before_os_exit(monkeypatch):
    """The teardown must be part of the step sequence that precedes ``os._exit``."""
    source = inspect.getsource(gateway_run._exit_after_graceful_shutdown)
    steps_line = next(line for line in source.splitlines() if "for _step in" in line)
    assert "stop_cua_backends_quietly" in steps_line
    # The teardown must not be the very last step before os._exit either way: it has to run while
    # the interpreter is still alive, and os._exit must remain the final statement.
    assert source.rstrip().endswith("os._exit(exit_code)")


def test_public_teardown_api_stops_every_cached_backend():
    """The public seam drains every session backend, not only the legacy slot."""
    first = _Recorder()
    second = _Recorder()
    backends = {"one": first, "two": second}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cu_tool, "_backend", None)
        mp.setattr(cu_tool, "_backends", backends)
        mp.setattr(cu_tool, "_backend_call_locks", {})
        cu_tool.stop_all_computer_use_backends()
    assert backends == {}


def test_public_teardown_api_never_raises():
    """The caller is on an exit path; a failing backend must not break the shutdown."""
    class _Boom:
        def stop(self):
            raise RuntimeError("stop failed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cu_tool, "_backend", None)
        mp.setattr(cu_tool, "_backends", {"one": _Boom()})
        mp.setattr(cu_tool, "_backend_call_locks", {})
        cu_tool.stop_all_computer_use_backends()


def test_watchdog_hard_exit_calls_the_teardown_seam():
    """The shutdown watchdog's forced exit is the other os._exit path and must tear down too."""
    source = inspect.getsource(shutdown_watchdog._hard_exit)
    assert "stop_cua_backends_quietly" in source


def test_teardown_seam_reaches_the_tool():
    """The shared seam is what both os._exit paths call, so it must reach the real teardown."""
    source = inspect.getsource(shutdown_watchdog.stop_cua_backends_quietly)
    assert "stop_all_computer_use_backends" in source

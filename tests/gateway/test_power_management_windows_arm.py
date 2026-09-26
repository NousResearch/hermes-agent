"""Windows power-broadcast monitor: the hidden window must actually arm.

Regression for #100025: with an undeclared ``CreateWindowExW``/``DefWindowProcW``
ctypes marshals the pointer-sized ``hInstance``/``hwnd``/``lparam`` args as C
``int`` on win64, so ``CreateWindowExW`` fails and ``start()`` returns False
with a misleading ``err=0`` -- the native suspend/resume broadcast never arms
and an unclean gateway exit after an overnight sleep is the only symptom the
user sees. These tests pin the arm and the dispatch.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from gateway.power_management import (
    PBT_APMRESUMEAUTOMATIC,
    PBT_APMSUSPEND,
    WM_POWERBROADCAST,
    WindowsPowerMonitor,
)

# ``windows_only`` rather than a ``sys.platform`` skipif: the CI Windows lane
# imports only the files scripts/ci/list_os_marked_tests.py finds this marker
# in, so with a skipif alone the native pump ran on no lane at all -- skipped
# on Linux, never imported on Windows. tests/conftest.py applies the skip on
# every other host, which is all the skipif did.
pytestmark = pytest.mark.windows_only


@pytest.fixture(autouse=True)
def _defuse_fail_closed_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the resume posts below from arming the real fail-closed exit.

    A sync test has no *running* loop, so the resume path resolves a parked
    ``ProactorEventLoop``; the guard's probe then never answers and ~20 s later
    it dumps every thread and ``os._exit(75)``s the pytest process -- measured:
    a run that lingers past the timeout dies with exit code 75. Per-file
    isolation hides that today (the file finishes in ~1.5 s), any longer run
    does not. In production this exit is the point; in a test process it is
    only a fuse.
    """
    monkeypatch.setenv("HERMES_POWER_FAIL_CLOSED", "0")


class _LogInbox(logging.Handler):
    """Collect records for the module logger and signal on the first match."""

    def __init__(self, needle: str) -> None:
        super().__init__(level=logging.DEBUG)
        self.needle = needle
        self.seen: list[str] = []
        self.hit = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        self.seen.append(message)
        if self.needle in message:
            self.hit.set()


def test_hidden_window_arms_and_stops():
    """The monitor must arm: this is what the ctypes truncation broke."""
    monitor = WindowsPowerMonitor(on_suspend=lambda: None, on_resume=lambda: None)
    try:
        assert monitor.start() is True, (
            "WindowsPowerMonitor.start() returned False -- the hidden "
            "WM_POWERBROADCAST window could not be created"
        )
        assert monitor.is_running is True
        assert monitor._hwnd, "armed without a window handle"
    finally:
        monitor.stop()
    assert monitor.is_running is False


def test_pump_delivers_a_power_broadcast_to_the_window_proc():
    """A posted WM_POWERBROADCAST must reach the hidden window's proc.

    This is the dispatch half: arming a window that never receives the
    broadcast would look healthy and still miss every suspend/resume.
    """
    module_logger = logging.getLogger("gateway.power_management")
    monitor = WindowsPowerMonitor(on_suspend=lambda: None, on_resume=lambda: None)
    inbox = _LogInbox("PBT_APMRESUMEAUTOMATIC")
    module_logger.addHandler(inbox)
    previous_level = module_logger.level
    module_logger.setLevel(logging.DEBUG)
    try:
        assert monitor.start() is True

        from gateway.power_management import _win32_dlls

        user32, _kernel32 = _win32_dlls()
        assert user32.PostMessageW(monitor._hwnd, WM_POWERBROADCAST, PBT_APMRESUMEAUTOMATIC, 0)

        assert inbox.hit.wait(timeout=5.0), f"broadcast never reached the window proc; saw {inbox.seen!r}"
    finally:
        module_logger.removeHandler(inbox)
        module_logger.setLevel(previous_level)
        monitor.stop()


def test_resume_broadcast_arms_the_fail_closed_guard(monkeypatch: pytest.MonkeyPatch):
    """Dispatching a resume must also arm the stuck-loop guard (#100025 review).

    Without this call a wedged loop waits forever for a callback it can never
    run -- exactly the half-restored state the reporter asked us to exit out of
    for a supervisor restart. Patched rather than asserted through the real
    guard: the defuse fixture is test hygiene, this pins the wiring.
    """
    import gateway.power_management as pm

    armed: list = []
    monkeypatch.setattr(pm, "_spawn_resume_fail_closed_guard", lambda loop, **kw: armed.append(loop))
    monitor = pm.WindowsPowerMonitor(on_suspend=lambda: None, on_resume=lambda: None)
    try:
        assert monitor.start() is True
        user32, _kernel32 = pm._win32_dlls()
        assert user32.PostMessageW(monitor._hwnd, WM_POWERBROADCAST, pm.PBT_APMRESUMESUSPEND, 0)
        deadline = time.monotonic() + 5.0
        while not armed and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        monitor.stop()
    assert armed, "resume broadcast did not arm the fail-closed guard"
    # Whatever loop it got is not running here -- which is why the guard would
    # exit(75) on a wedged gateway and why the fixture above is needed in tests.
    assert armed[0] is None or armed[0].is_running() is False

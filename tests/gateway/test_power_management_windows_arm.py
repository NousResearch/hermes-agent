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
import sys
import threading

import pytest

from gateway.power_management import (
    PBT_APMRESUMEAUTOMATIC,
    PBT_APMSUSPEND,
    WM_POWERBROADCAST,
    WindowsPowerMonitor,
)

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only native power pump")


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

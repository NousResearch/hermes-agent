"""Fail-closed resume: the wake handler must not need a schedulable loop.

#100025 review asked for exactly this test: when the gateway loop is already
stuck *before* the resume broadcast arrives (stale fds make the selector's
``select()`` return early after the wall-clock jump), ``call_soon_threadsafe``
still succeeds and the resume callback simply never runs. The pump thread must
then dump state, mark the life exited and ``os._exit`` with the
supervisor-restart code instead of sitting in a half-restored state -- the same
contract as ``gateway.shutdown_watchdog.start_loop_liveness_watchdog``.

Cross-platform on purpose: the guard takes any loop-shaped object, so these run
everywhere (the ctypes pump itself is covered in
``test_power_management_windows_arm.py``).
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from gateway.power_management import _spawn_resume_fail_closed_guard
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE


class _StuckLoop:
    """Loop-shaped: schedule-able, but the callback never runs (the wedge)."""

    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, callback, *args) -> None:
        pass


class _LiveLoop:
    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, callback, *args) -> None:
        callback(*args)


class _ClosedLoop:
    def is_closed(self) -> bool:
        return True

    def call_soon_threadsafe(self, callback, *args) -> None:
        raise RuntimeError("Event loop is closed")


@pytest.fixture
def exits(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Capture what the guard would exit with (it runs on its own thread)."""
    seen: dict = {"codes": [], "ledger": [], "event": threading.Event()}

    def _exit(code: int) -> None:
        seen["codes"].append(code)
        seen["event"].set()

    def _mark(exit_code=None, reason="graceful_shutdown", home=None) -> None:
        seen["ledger"].append((exit_code, reason))

    monkeypatch.setattr("os._exit", _exit)
    monkeypatch.setattr("gateway.lifecycle_ledger.mark_exited", _mark)
    monkeypatch.delenv("HERMES_POWER_FAIL_CLOSED", raising=False)
    return seen


def test_stuck_loop_fails_closed_with_supervisor_restart_code(exits, monkeypatch: pytest.MonkeyPatch):
    dumped: list = []
    monkeypatch.setattr("faulthandler.dump_traceback", lambda: dumped.append(True))

    _spawn_resume_fail_closed_guard(_StuckLoop(), timeout=1.0)

    assert exits["event"].wait(5.0), "a wedged loop never triggered the fail-closed exit"
    assert exits["codes"] == [GATEWAY_SERVICE_RESTART_EXIT_CODE], (
        "the guard must exit with the supervisor-restart code, not 0 -- otherwise "
        "the supervisor does not revive the gateway"
    )
    assert exits["ledger"] == [(GATEWAY_SERVICE_RESTART_EXIT_CODE, "power_resume_loop_stuck")]
    assert dumped, "no thread dump before exiting -- the wedge would leave no forensics"


def test_a_running_but_blocked_loop_still_fails_closed(exits):
    """The reporter's exact shape: the loop is alive, it just can't run callbacks.

    Pins that ``call_soon_threadsafe`` does *not* raise here -- if the guard only
    watched for the closed-loop ``RuntimeError`` it would silently no-op in the
    one scenario it exists for.
    """
    loop = asyncio.new_event_loop()
    entered = threading.Event()

    def _drive() -> None:
        def _block() -> None:
            entered.set()
            time.sleep(3.0)

        loop.call_soon(_block)
        loop.run_forever()

    driver = threading.Thread(target=_drive, daemon=True)
    driver.start()
    assert entered.wait(5.0), "loop thread never started"
    try:
        _spawn_resume_fail_closed_guard(loop, timeout=1.0)
        assert exits["event"].wait(5.0), "a running-but-blocked loop must still fail closed"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        driver.join(timeout=5.0)
        loop.close()


def test_live_loop_takes_the_graceful_path(exits):
    _spawn_resume_fail_closed_guard(_LiveLoop(), timeout=1.0)

    assert not exits["event"].wait(2.0), "a healthy loop must not be exited"


def test_closed_loop_is_an_orderly_shutdown_not_a_wedge(exits):
    _spawn_resume_fail_closed_guard(_ClosedLoop(), timeout=1.0)

    assert not exits["event"].wait(2.0), "shutdown must not be reported as a stuck loop"


def test_no_loop_is_a_noop(exits):
    _spawn_resume_fail_closed_guard(None, timeout=1.0)

    assert not exits["event"].wait(1.5)


def test_fail_closed_can_be_disabled(exits, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_POWER_FAIL_CLOSED", "0")

    _spawn_resume_fail_closed_guard(_StuckLoop(), timeout=1.0)

    assert not exits["event"].wait(2.0)


def test_real_ledger_accepts_the_guards_call_form(tmp_path):
    """The guard swallows a ``mark_exited`` failure -- so pin the call form.

    If the signature drifted, the ledger would record nothing and the next boot
    would report the very UNCLEAN death this PR exists to remove.
    """
    from gateway.lifecycle_ledger import mark_exited, read_prior_exit_label

    mark_exited(GATEWAY_SERVICE_RESTART_EXIT_CODE, reason="power_resume_loop_stuck", home=tmp_path)

    assert read_prior_exit_label(tmp_path) == "clean"

"""Exit watchdog: arm on shutdown *intent* (signal), never at chat startup.

Regression coverage for the #65998 class: a ``hermes --tui`` process whose
main thread wedges before ``app.run()`` returns never executes the ``finally``
that calls ``_run_cleanup`` — the only place the exit watchdog used to be
armed — so a "dead" CLI lingered indefinitely (observed ~47 min at 4% CPU).

The fix arms the backstop from the SIGTERM/SIGHUP handlers via
``_arm_exit_watchdog_on_shutdown_signal()``. Arming at *startup* (the
rejected #65998 approach) is specifically forbidden: the watchdog thread
calls ``os._exit(0)`` unconditionally after its sleep, so a startup-armed
timer hard-kills every session that outlives the timeout.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

import cli


@pytest.fixture(autouse=True)
def _reset_arm_flag(monkeypatch):
    """Each test starts with the idempotency flag clear."""
    monkeypatch.setattr(cli, "_signal_watchdog_armed", False)


class TestSignalArmLogic:
    def test_arms_with_double_cleanup_timeout(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXIT_WATCHDOG_S", "7")
        with patch.object(cli, "_arm_exit_watchdog") as arm:
            cli._arm_exit_watchdog_on_shutdown_signal()
        arm.assert_called_once_with(timeout_s=14.0, from_signal=True)



    def test_bad_env_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXIT_WATCHDOG_S", "not-a-number")
        with patch.object(cli, "_arm_exit_watchdog") as arm:
            cli._arm_exit_watchdog_on_shutdown_signal()
        arm.assert_called_once_with(timeout_s=60.0, from_signal=True)

    def test_never_raises_even_if_arm_explodes(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXIT_WATCHDOG_S", "7")
        with patch.object(cli, "_arm_exit_watchdog", side_effect=RuntimeError("boom")):
            cli._arm_exit_watchdog_on_shutdown_signal()  # must not raise


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A minimal stand-in for the wedged-CLI shape: signal handlers mirror the
# production wiring (arm-on-signal, then a graceful unwind that wedges), and
# the main thread parks the way a stuck app.run() does.
_WEDGE_SRC = """
import os, signal, sys, time
sys.path.insert(0, {repo!r})
import cli

def _handler(signum, frame):
    # Production wiring: arm the backstop the moment shutdown intent exists,
    # then attempt a graceful unwind — which, in this repro, wedges (the
    # KeyboardInterrupt lands in a frame that swallows it).
    cli._arm_exit_watchdog_on_shutdown_signal()

signal.signal(signal.SIGTERM, _handler)
print("READY", flush=True)
while True:  # the wedge: never observes any unwind
    time.sleep(0.2)
"""

_CLEANUP_OVERLAP_SRC = """
import os, signal, sys, threading, time
sys.path.insert(0, {repo!r})
import cli


def _start_cleanup():
    # Simulate a slow graceful-cleanup path that starts after signal intent.
    time.sleep(1.0)
    cli._cleanup_in_progress = True
    cli._arm_exit_watchdog(timeout_s=1.5)
    time.sleep(5.0)


def _handler(signum, frame):
    # Arm the broad signal watchdog, then trigger a cleanup window shortly after.
    cli._arm_exit_watchdog_on_shutdown_signal()
    threading.Thread(target=_start_cleanup, daemon=True).start()


signal.signal(signal.SIGTERM, _handler)
print("READY", flush=True)
while True:
    time.sleep(0.2)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_sigterm_on_wedged_process_forces_exit_within_leash():
    """E2E: a wedged process armed via the signal path self-exits at ~2×
    HERMES_EXIT_WATCHDOG_S; without the signal it would live forever."""
    env = dict(os.environ, HERMES_EXIT_WATCHDOG_S="1", PYTHONPATH=_REPO_ROOT)
    # _arm_exit_watchdog refuses to arm under pytest (it would kill the test
    # worker); the subprocess must look like a real CLI.
    env.pop("PYTEST_CURRENT_TEST", None)
    src = _WEDGE_SRC.format(repo=_REPO_ROOT)
    p = subprocess.Popen(
        [sys.executable, "-c", src],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert p.stdout is not None
        assert p.stdout.readline().strip() == "READY"
        # Wedged, no signal yet: must still be alive well past the leash
        # (proves we did NOT arm at startup — the #65998 regression).
        time.sleep(3.0)
        assert p.poll() is None, "watchdog fired without shutdown intent"

        p.send_signal(signal.SIGTERM)
        t0 = time.time()
        rc = p.wait(timeout=10)
        elapsed = time.time() - t0
        assert rc == 0
        # Leash is 2×1s; generous CI slack.
        assert elapsed < 8.0, f"exit took {elapsed:.1f}s; leash should be ~2s"
    finally:
        if p.poll() is None:
            p.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_signal_watchdog_is_skipped_once_cleanup_starts():
    """If cleanup starts after signal intent, the cleanup-owned watchdog should
    govern shutdown and the signal watchdog should not hard-kill midway."""
    env = dict(os.environ, HERMES_EXIT_WATCHDOG_S="1", PYTHONPATH=_REPO_ROOT)
    env.pop("PYTEST_CURRENT_TEST", None)
    src = _CLEANUP_OVERLAP_SRC.format(repo=_REPO_ROOT)
    p = subprocess.Popen(
        [sys.executable, "-c", src],
        env=env,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert p.stdout is not None
        assert p.stdout.readline().strip() == "READY"

        p.send_signal(signal.SIGTERM)
        t0 = time.time()
        rc = p.wait(timeout=10)
        elapsed = time.time() - t0
        assert rc == 0
        # Outer watchdog is at 2s, cleanup watchdog at 2.5s (1s delay + 1.5s
        # timeout). Without the overlap fix this would likely exit earlier at 2s.
        assert elapsed >= 2.2, f"elapsed={elapsed:.2f}s; expected cleanup watchdog to fire"
        assert elapsed <= 5.0, f"elapsed={elapsed:.2f}s; expected bounded shutdown"
    finally:
        if p.poll() is None:
            p.kill()


class TestWindowsSigintEscalation:
    """The Windows-only half of the backstop (#100747).

    ``run()`` binds SIGINT to a silent absorber on Windows because the console
    fires spurious ``CTRL_C_EVENT`` whenever a background thread spawns a
    ``.cmd`` child. That absorber is the ONLY handler a Windows console can
    reach — there is no ``SIGHUP`` and Ctrl+C never raises ``SIGTERM`` — so if
    it never arms the backstop, a wedge before ``_run_cleanup`` leaves a CLI
    no keystroke can end.
    """

    @pytest.fixture(autouse=True)
    def _clean_cleanup_flags(self, monkeypatch):
        monkeypatch.setattr(cli, "_cleanup_in_progress", False)
        monkeypatch.setattr(cli, "_cleanup_done", False)

    def test_idle_press_is_console_noise(self):
        """No shutdown intent: the press stays fully absorbed."""
        stub = type("Stub", (), {"_should_exit": False})()
        assert cli._windows_sigint_is_escalation(stub) is False

    def test_press_after_exit_requested_is_escalation(self):
        """``handle_ctrl_c`` set ``_should_exit`` and ``app.exit()`` wedged."""
        stub = type("Stub", (), {"_should_exit": True})()
        assert cli._windows_sigint_is_escalation(stub) is True

    def test_press_during_cleanup_is_escalation(self, monkeypatch):
        monkeypatch.setattr(cli, "_cleanup_in_progress", True)
        stub = type("Stub", (), {"_should_exit": False})()
        assert cli._windows_sigint_is_escalation(stub) is True

    def test_press_after_cleanup_is_escalation(self, monkeypatch):
        monkeypatch.setattr(cli, "_cleanup_done", True)
        stub = type("Stub", (), {"_should_exit": False})()
        assert cli._windows_sigint_is_escalation(stub) is True

    def test_missing_attribute_is_not_escalation(self):
        """A CLI object without ``_should_exit`` must not arm a hard kill."""
        assert cli._windows_sigint_is_escalation(object()) is False


def test_windows_absorber_arms_backstop_only_on_escalation(monkeypatch):
    """End-to-end handler shape: absorb always, arm only when escalating.

    Mirrors the closure installed in ``HermesCLI.run()`` without booting the
    whole TUI: the absorber must return ``None`` in both cases (never raise,
    never interrupt the agent) and reach the backstop only once shutdown
    intent exists.
    """
    calls = []
    monkeypatch.setattr(
        cli, "_arm_exit_watchdog_on_shutdown_signal", lambda: calls.append(1)
    )
    monkeypatch.setattr(cli, "_cleanup_in_progress", False)
    monkeypatch.setattr(cli, "_cleanup_done", False)
    stub = type("Stub", (), {"_should_exit": False})()

    def _sigint_absorb(signum, frame):
        try:
            if cli._windows_sigint_is_escalation(stub):
                cli._arm_exit_watchdog_on_shutdown_signal()
        except Exception:
            pass
        return

    assert _sigint_absorb(signal.SIGINT, None) is None
    assert calls == []

    stub._should_exit = True
    assert _sigint_absorb(signal.SIGINT, None) is None
    assert calls == [1]

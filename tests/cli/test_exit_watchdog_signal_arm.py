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
        arm.assert_called_once_with(timeout_s=14.0)

    def test_idempotent_across_repeated_signals(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXIT_WATCHDOG_S", "7")
        with patch.object(cli, "_arm_exit_watchdog") as arm:
            cli._arm_exit_watchdog_on_shutdown_signal()
            cli._arm_exit_watchdog_on_shutdown_signal()
            cli._arm_exit_watchdog_on_shutdown_signal()
        assert arm.call_count == 1

    def test_disabled_via_env_zero(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXIT_WATCHDOG_S", "0")
        with patch.object(cli, "_arm_exit_watchdog") as arm:
            cli._arm_exit_watchdog_on_shutdown_signal()
        arm.assert_not_called()

    def test_bad_env_value_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HERMES_EXIT_WATCHDOG_S", "not-a-number")
        with patch.object(cli, "_arm_exit_watchdog") as arm:
            cli._arm_exit_watchdog_on_shutdown_signal()
        arm.assert_called_once_with(timeout_s=60.0)

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

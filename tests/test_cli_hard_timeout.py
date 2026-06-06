"""Tests for the opt-in hard wall-clock timeout watchdog in cli.main()."""

import os
import subprocess
import sys
import threading
import time

import pytest

from cli import _install_hard_timeout_watchdog


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestInstallHardTimeoutWatchdog:
    """Unit tests for the env-var-driven helper that installs the watchdog."""

    def test_no_env_var_returns_none(self):
        assert _install_hard_timeout_watchdog(None) is None
        assert _install_hard_timeout_watchdog("") is None

    def test_non_numeric_env_var_returns_none(self):
        assert _install_hard_timeout_watchdog("not-a-number") is None
        assert _install_hard_timeout_watchdog("abc") is None

    def test_zero_or_negative_returns_none(self):
        assert _install_hard_timeout_watchdog("0") is None
        assert _install_hard_timeout_watchdog("-1") is None
        assert _install_hard_timeout_watchdog("-0.5") is None

    def test_positive_value_starts_daemon_thread(self):
        # Use a long timeout so the watchdog doesn't actually fire during
        # the test run; we only assert that the thread is created and named.
        before = {t.name for t in threading.enumerate()}
        thread = _install_hard_timeout_watchdog("3600")
        try:
            assert thread is not None
            assert thread.is_alive()
            assert thread.daemon is True
            assert thread.name == "hermes-hard-timeout"
            # The thread is newly created (the name isn't a duplicate from
            # before the call).
            assert "hermes-hard-timeout" not in before
        finally:
            # Daemon threads die with the interpreter; nothing to clean up
            # explicitly, but we drop the reference to be tidy.
            del thread


@pytest.mark.timeout(15)
def test_watchdog_force_terminates_wedged_process(tmp_path):
    """End-to-end: a subprocess with HERMES_HARD_TIMEOUT_SEC=1.0 set must exit
    with code 124 within ~2s, even though it would otherwise sleep for 30s."""
    script = tmp_path / "wedge.py"
    script.write_text(
        "import sys, os\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        "from cli import _install_hard_timeout_watchdog\n"
        "_install_hard_timeout_watchdog(os.environ.get('HERMES_HARD_TIMEOUT_SEC'))\n"
        "import time; time.sleep(30)\n"
    )

    env = os.environ.copy()
    env["HERMES_HARD_TIMEOUT_SEC"] = "1.0"

    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        timeout=10,
    )
    elapsed = time.monotonic() - start

    assert result.returncode == 124, (
        f"expected exit 124, got {result.returncode}; "
        f"stderr={result.stderr.decode(errors='replace')!r}"
    )
    # Allow generous slack for slow CI; the point is it killed the process
    # well before the 30s sleep would have completed.
    assert elapsed < 5.0, f"watchdog took {elapsed:.2f}s to fire"
    assert b"Hard timeout" in result.stderr

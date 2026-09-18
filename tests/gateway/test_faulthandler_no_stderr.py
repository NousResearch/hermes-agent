"""Regression: #71671 — gateway must survive faulthandler.enable() with sys.stderr=None."""

from __future__ import annotations

import faulthandler
import sys

import pytest


def test_faulthandler_enable_falls_back_when_stderr_is_none(tmp_path):
    was_enabled = faulthandler.is_enabled()
    if was_enabled:
        faulthandler.disable()

    real_stderr = sys.stderr
    sys.stderr = None
    try:
        with pytest.raises(RuntimeError, match="sys.stderr is None"):
            faulthandler.enable()

        log_path = tmp_path / "gateway_faulthandler.log"
        fh = open(log_path, "a", encoding="utf-8")
        try:
            faulthandler.enable(file=fh, all_threads=True)
            assert faulthandler.is_enabled()
            assert log_path.exists()
        finally:
            faulthandler.disable()
            fh.close()
    finally:
        sys.stderr = real_stderr
        if was_enabled:
            faulthandler.enable()


_SIGUSR2_CHILD = """
import os, signal, sys, time
from types import SimpleNamespace
sys.path.insert(0, sys.argv[2])
from gateway.run_startup import GatewayStartupMixin
runner = object.__new__(GatewayStartupMixin)
runner.config = SimpleNamespace(log_dir=sys.argv[1])
runner._start_install_faulthandler()
print("ready", flush=True)
time.sleep(30)
"""


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGUSR2"), reason="POSIX-only signal")
def test_sigusr2_stack_dump_leaves_the_gateway_running(tmp_path):
    """#110437 — SIGUSR2 must dump stacks to the log and return; chaining to the
    signal's default disposition (terminate) killed the process being inspected."""
    import os
    import signal
    import subprocess
    import time

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    child = subprocess.Popen(
        [sys.executable, "-c", _SIGUSR2_CHILD, str(tmp_path), repo_root],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        child.send_signal(signal.SIGUSR2)
        time.sleep(0.5)
        assert child.poll() is None, f"gateway died on SIGUSR2 (rc={child.returncode})"
    finally:
        child.kill()
        child.wait()
    dump = (tmp_path / "gateway_faulthandler.log").read_text(encoding="utf-8")
    assert "Thread 0x" in dump or "Current thread" in dump

"""#95693 / #93958 — live macOS proof for the parent-death watchdog.

The unit tests in ``test_serve_parent_watchdog.py`` pin ``_is_serve_orphaned``
as a pure function. The bug itself lived one layer up: a daemon thread that
calls ``os._exit(0)`` ~5ms after ``HERMES_BACKEND_READY``, which no in-process
test can observe. This file runs the REAL ``start_server`` in a subprocess, on
the host ``ps``, with the marker the Desktop would hand it rendered in a
different timezone than the backend's own probe.

Contract:

* live parent + TZ-drifted ``ps:`` marker → backend announces READY and is
  still alive well past several watchdog polls;
* that same backend still exits once the parent actually dies (the marker
  degrade did not turn the watchdog off).
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.macos_only


def _lstart(pid: int, tz: str) -> str:
    env = dict(os.environ, TZ=tz)
    out = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], capture_output=True, text=True, env=env, check=True
    ).stdout
    return " ".join(out.split())


def _read_until(proc: subprocess.Popen, token: str, timeout: float = 120.0) -> bool:
    hit = threading.Event()

    def _pump():
        assert proc.stdout is not None
        for line in proc.stdout:
            if token in line:
                hit.set()
                return

    threading.Thread(target=_pump, daemon=True).start()
    return hit.wait(timeout)


def _wait_exit(proc: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.1)
    return False


def test_live_backend_survives_timezone_drifted_parent_marker(tmp_path):
    # Stand-in for the Electron parent: a long-lived process we control.
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    serve = None
    try:
        backend_tz = "America/New_York"
        cached_tz = "Europe/Paris"
        drifted_marker = f"ps:{_lstart(parent.pid, cached_tz)}"
        assert drifted_marker != f"ps:{_lstart(parent.pid, backend_tz)}", (
            "precondition: the two TZ renderings of the same instant must differ"
        )

        home = tmp_path / "hermes_home"
        home.mkdir()
        env = dict(os.environ)
        env.update(
            TZ=backend_tz,
            HERMES_HOME=str(home),
            HERMES_SERVE_HEADLESS="1",
            PYTHONUNBUFFERED="1",
            HERMES_PARENT_PID=str(parent.pid),
            HERMES_PARENT_START_MARKER=drifted_marker,
            HERMES_PARENT_NONCE="nonce-95693",
            HERMES_SERVE_WATCHDOG_POLL_S="0.5",
        )
        env.pop("HERMES_DESKTOP", None)
        code = (
            "from hermes_cli.web_server import start_server\n"
            "start_server(host='127.0.0.1', port=0, open_browser=False, headless=True)\n"
        )
        serve = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        assert _read_until(serve, "HERMES_BACKEND_READY"), "backend never announced READY"

        # Before the fix the watchdog fired on its very first poll. Several
        # polls later the backend must still be here.
        assert not _wait_exit(serve, timeout=4.0), (
            f"backend exited (code {serve.returncode}) with a live parent; "
            f"TZ-drifted marker {drifted_marker!r} was treated as proof of death"
        )

        # The degrade to PID liveness must still reap a genuinely dead parent.
        parent.kill()
        parent.wait(timeout=10)
        assert _wait_exit(serve, timeout=15.0), "backend outlived its dead parent"
        assert serve.returncode == 0
    finally:
        for proc in (serve, parent):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

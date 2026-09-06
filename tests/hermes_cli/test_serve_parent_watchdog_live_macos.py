"""#95693 / #93958 — hermetic macOS parent-watchdog regression proof.

The historical test launched a real web server, invoked the host ``ps``, and
used real process signals. That is incompatible with the repository's
fail-closed test boundary: macOS tests cannot receive host process-enumeration
or network capabilities merely to validate marker comparison.

This subprocess proof drives the production watchdog thread and its real
``os._exit`` termination path. The only injected seam is the parent-death
observation, controlled by a test-owned sentinel file.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.macos_only


def test_backend_watchdog_thread_survives_then_exits_on_injected_parent_death(tmp_path):
    sentinel = tmp_path / "parent-dead"
    code = r"""
import os
from pathlib import Path
import time
from hermes_cli.web_server_lifecycle import _start_parent_death_watchdog

sentinel = Path(os.environ["HERMES_TEST_PARENT_DEAD_SENTINEL"])
os.environ["HERMES_PARENT_PID"] = "4242"
os.environ["HERMES_PARENT_START_MARKER"] = "ps:Sat Sep 06 14:00:00 2026"
os.environ["HERMES_PARENT_NONCE"] = "hermetic-test-nonce"
os.environ["HERMES_SERVE_WATCHDOG_POLL_S"] = "0.5"
_start_parent_death_watchdog(orphan_probe=lambda _pid, _marker: sentinel.exists())
print("WATCHDOG_STARTED", flush=True)
while True:
    time.sleep(0.1)
"""
    env = os.environ.copy()
    env["HERMES_TEST_PARENT_DEAD_SENTINEL"] = str(sentinel)
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "WATCHDOG_STARTED"
    time.sleep(1.1)
    assert process.poll() is None
    sentinel.write_text("dead\n", encoding="utf-8")
    assert process.wait(timeout=5) == 0

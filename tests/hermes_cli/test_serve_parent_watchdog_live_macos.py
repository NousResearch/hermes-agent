"""#95693 / #93958 — hermetic macOS parent-watchdog regression proof.

The historical test launched a real web server, invoked the host ``ps``, and
used real process signals. That is incompatible with the repository's
fail-closed test boundary: macOS tests cannot receive host process-enumeration
or network capabilities merely to validate marker comparison.

This subprocess proof retains the production decision path while injecting the
two OS observations it consumes. A timezone-drifted ``ps:`` marker stays
inconclusive while its parent is alive, then becomes orphaned when the injected
liveness probe reports the parent gone.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.macos_only


def test_backend_watchdog_degrades_timezone_drift_to_injected_pid_liveness():
    code = r"""
import json
from hermes_cli.web_server_lifecycle import _is_serve_orphaned

expected = "ps:Sat Sep 06 14:00:00 2026"
actual = "ps:Sat Sep 06 20:00:00 2026"
alive = _is_serve_orphaned(
    4242,
    expected,
    pid_exists=lambda _pid: True,
    process_start_marker=lambda _pid: actual,
)
dead = _is_serve_orphaned(
    4242,
    expected,
    pid_exists=lambda _pid: False,
    process_start_marker=lambda _pid: actual,
)
print(json.dumps({"alive_is_orphaned": alive, "dead_is_orphaned": dead}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "alive_is_orphaned": False,
        "dead_is_orphaned": True,
    }

"""Fatal-signal forensics at the gateway process entry (#126099).

``GatewayStartupMixin`` arms faulthandler once config resolves (#70344); the window before that —
module imports of C extensions, config load, the startup watchdog — used to have no fatal-signal
dump at all, so a native crash there left the next boot's respawn-storm warning as the only
symptom. ``arm_faulthandler_at_process_entry`` closes that window, with the same stderr-less log
file fallback the startup install uses (#71671)."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_CHILD = """
import os, signal, sys
from gateway.run_startup import arm_faulthandler_at_process_entry
arm_faulthandler_at_process_entry()
print("armed", flush=True)
os.kill(os.getpid(), signal.SIGILL)
"""

_CHILD_NO_STDERR = """
import os, signal, sys
sys.stderr = None  # Windows VBS / pythonw / detached service
from gateway.run_startup import arm_faulthandler_at_process_entry
arm_faulthandler_at_process_entry()
print("armed", flush=True)
os.kill(os.getpid(), signal.SIGILL)
"""


def _run_child(script: str, tmp_path: Path, hermes_home: Path | None = None):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path),
           "PYTHONPATH": str(REPO_ROOT)}
    if hermes_home is not None:
        env["HERMES_HOME"] = str(hermes_home)
    return subprocess.run(
        [sys.executable, "-c", script], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=60,
    )


@pytest.mark.platforms("posix")
def test_fatal_signal_after_entry_arming_dumps_into_the_log(tmp_path):
    """A fatal signal with no config, no GatewayRunner and no startup phase must still dump the
    signal name and the crashing frame — the evidence a supervisor-only record lacks."""
    proc = _run_child(_CHILD, tmp_path)
    assert proc.returncode == -signal.SIGILL
    assert "Fatal Python error: Illegal instruction" in proc.stderr
    assert "Current thread" in proc.stderr
    assert 'File "<string>", line 6 in <module>' in proc.stderr


@pytest.mark.platforms("posix")
def test_entry_arming_falls_back_to_the_log_file_without_stderr(tmp_path):
    """A stderr-less process (Windows VBS / pythonw / detached service) must still get the dump —
    in ``gateway_faulthandler.log`` — instead of dying on the enable call (#71671 class)."""
    home = tmp_path / "home"
    (home / "logs").mkdir(parents=True)
    proc = _run_child(_CHILD_NO_STDERR, tmp_path, hermes_home=home)
    assert proc.returncode == -signal.SIGILL
    dump = home / "logs" / "gateway_faulthandler.log"
    assert dump.exists(), f"no fallback dump written (stderr={proc.stderr[-200:]!r})"
    assert "Fatal Python error: Illegal instruction" in dump.read_text(encoding="utf-8")

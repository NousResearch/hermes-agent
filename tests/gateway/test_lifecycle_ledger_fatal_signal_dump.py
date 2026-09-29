"""Fatal-signal forensics for the gateway lifecycle ledger (#126099): an uncatchable native death
(SIGILL/SIGSEGV — e.g. an illegal instruction in a C dependency under virtualization) runs no
Python handler, so without faulthandler the only evidence is the supervisor's own record plus the
next boot's respawn-storm warning. Armed, the signal name and an all-thread traceback land in the
gateway log where the operator can see them."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_CHILD = """
import os, signal, sys
from gateway.lifecycle_ledger import enable_fatal_signal_dump
enable_fatal_signal_dump()
print("armed", flush=True)
os.kill(os.getpid(), signal.SIGILL)
"""


@pytest.mark.platforms("posix")
def test_fatal_signal_leaves_a_named_traceback_in_the_gateway_log():
    """A fatal signal after arming must die by the signal AND dump its name + traceback to stderr
    (the gateway log), instead of leaving the respawn storm as the only symptom."""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=REPO_ROOT,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(REPO_ROOT),
             "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == -signal.SIGILL
    # The dump names the signal — the evidence a supervisor-only record (s6-svdt) lacks.
    assert "Fatal Python error: Illegal instruction" in proc.stderr
    # All-thread dump covering the child's own frames at crash time.
    assert "Current thread" in proc.stderr
    assert 'File "<string>", line 6 in <module>' in proc.stderr

"""A run claim the process image before an in-place relaunch took is not held by the relaunched one.

``hermes_cli/relaunch.py`` replaces the process with ``os.execvp``: same pid, same start time. A
claim taken before the exec (a curator pass or a plugin idle task at CLI startup, then ``/update``
or a session resume) was never released, and its owner looked alive, so every idle job of that home
stayed suppressed until the relaunched process exited.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.platforms("linux")  # pid start + machine identity come from /proc

_ROOT = Path(__file__).resolve().parents[2]
_CLAIM_THEN_EXEC = """
import os, sys
from pathlib import Path
from agent.idle_schedule import claim_run
lock = sys.argv[1]
assert claim_run(Path(lock))
os.execv(sys.executable, [sys.executable, "-c", sys.argv[2], lock])
"""
_AFTER_EXEC = """
import sys
from pathlib import Path
from agent.idle_schedule import claim_run
print(claim_run(Path(sys.argv[1])), flush=True)
"""


def test_the_relaunched_image_takes_over_its_predecessors_claim(tmp_path):
    lock = tmp_path / ".task.lock"
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    out = subprocess.run([sys.executable, "-c", _CLAIM_THEN_EXEC, str(lock), _AFTER_EXEC],
                         check=False, capture_output=True, text=True, env=env, cwd=_ROOT, timeout=60)
    assert out.stdout.strip() == "True", out.stderr

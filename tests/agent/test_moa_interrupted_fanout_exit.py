"""An interrupted MoA fan-out must not hold the process open on the reference it abandoned.

On interrupt the fan-out returns at once and leaves in-flight reference calls running (they
cannot be killed). With stdlib ThreadPoolExecutor workers, the interpreter's exit hook still
joins that worker, so quitting right after an interrupt waited for the provider call to finish,
up to the reference timeout (900 s by default).
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_SCRIPT = f"""
import sys, threading, time
sys.path.insert(0, {str(_REPO_ROOT)!r})
from types import SimpleNamespace
from agent import moa_loop

started = threading.Event()

def stuck_reference(slot, messages, **kwargs):
    started.set()
    time.sleep(120)  # a provider call that never comes back

moa_loop._run_reference = stuck_reference
agent = SimpleNamespace(_interrupt_requested=False)

def interrupt_when_running():
    started.wait(10)
    agent._interrupt_requested = True

threading.Thread(target=interrupt_when_running, daemon=True).start()
out = moa_loop._run_references_parallel(
    [{{"provider": "openrouter", "model": "slow/model"}}], [{{"role": "user", "content": "hi"}}],
    agent=agent)
print("returned", len(out), flush=True)
"""


def test_interrupted_fanout_does_not_block_interpreter_exit():
    proc = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True,
                          timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "returned 1" in proc.stdout

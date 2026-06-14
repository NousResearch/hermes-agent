"""Runtime health contract for Hermes OS dashboard polling."""

import os
import subprocess
import time

from .contracts import RuntimeStatus
from .wrapper import default_launcher_path


def check_runtime_health(launcher_path=None, timeout_seconds=3):
    launcher = launcher_path or default_launcher_path()
    if not os.path.exists(launcher):
        return RuntimeStatus(
            available=False,
            provider="official-hermes-agent",
            recent_errors=["hermes-agent launcher not found"],
        )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [launcher, "--help"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception as exc:
        return RuntimeStatus(
            available=False,
            provider="official-hermes-agent",
            latency_ms=int((time.monotonic() - started) * 1000),
            recent_errors=[str(exc)],
        )
    return RuntimeStatus(
        available=completed.returncode == 0,
        provider="official-hermes-agent",
        version="unknown",
        latency_ms=int((time.monotonic() - started) * 1000),
        recent_errors=[] if completed.returncode == 0 else [completed.stderr.strip()],
    )

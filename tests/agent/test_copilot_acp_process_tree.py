"""Process-lifecycle regressions for the Copilot ACP subprocess transport."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import psutil

from agent.copilot_acp_client import CopilotACPClient


_PARENT_SCRIPT = r"""
import subprocess
import sys
import time
from pathlib import Path

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
Path(sys.argv[1]).write_text(str(child.pid), encoding="utf-8")
time.sleep(60)
"""


def _pid_alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() not in {
            psutil.STATUS_DEAD,
            psutil.STATUS_ZOMBIE,
        }
    except psutil.NoSuchProcess:
        return False


def _wait_until(predicate, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def test_close_reaps_acp_descendants(tmp_path: Path) -> None:
    """An ACP transport owns its whole subprocess tree, not only the launcher.

    A CLI may spawn helpers while servicing a turn.  Closing the transport must
    therefore leave no descendant running after the direct ACP child exits.
    """
    child_pid_file = tmp_path / "child.pid"
    client = CopilotACPClient(
        acp_command=sys.executable,
        acp_args=["-c", _PARENT_SCRIPT, str(child_pid_file)],
        acp_cwd=str(tmp_path),
    )
    child_pid: int | None = None

    try:
        with patch("agent.copilot_acp_client._acp_supported", return_value=True):
            parent = client._spawn()

        assert _wait_until(child_pid_file.exists), "helper child pid was never published"
        child_pid = int(child_pid_file.read_text(encoding="utf-8"))
        assert parent.poll() is None
        assert _pid_alive(child_pid)

        client.close()

        assert parent.poll() is not None
        assert _wait_until(lambda: not _pid_alive(child_pid)), (
            "CopilotACPClient.close() terminated only the direct ACP process; "
            "its descendant survived the transport lifecycle"
        )
    finally:
        client.close()
        if child_pid is not None and _pid_alive(child_pid):
            try:
                process = psutil.Process(child_pid)
                process.kill()
                process.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass

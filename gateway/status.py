"""
Gateway runtime status helpers.

Provides PID-file based detection of whether the gateway daemon is running,
used by send_message's check_fn to gate availability in the CLI.
"""

import os
from pathlib import Path

_PID_FILE = Path.home() / ".hermes" / "gateway.pid"


def write_pid_file() -> None:
    """Write the current process PID to the gateway PID file."""
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))


def remove_pid_file() -> None:
    """Remove the gateway PID file if it exists."""
    try:
        _PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def is_gateway_running() -> bool:
    """Check if the gateway daemon is currently running."""
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal sent
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        # Stale PID file -- process is gone
        remove_pid_file()
        return False

"""
Gateway runtime status helpers.

Provides PID-file based detection of whether the gateway daemon is running,
used by send_message's check_fn to gate availability in the CLI.

The PID file lives at ``{HERMES_HOME}/gateway.pid``.  HERMES_HOME defaults to
``~/.hermes`` but can be overridden via the environment variable.  This means
separate HERMES_HOME directories naturally get separate PID files — a property
that will be useful when we add named profiles (multiple agents running
concurrently under distinct configurations).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _get_pid_path() -> Path:
    """Return the path to the gateway PID file, respecting HERMES_HOME."""
    home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return home / "gateway.pid"


def write_pid_file() -> None:
    """Write the current process PID to the gateway PID file."""
    pid_path = _get_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def remove_pid_file() -> None:
    """Remove the gateway PID file if it exists."""
    try:
        _get_pid_path().unlink(missing_ok=True)
    except Exception:
        pass


def _is_hermes_gateway_process(pid: int) -> bool:
    """Check if a process with the given PID is actually a hermes gateway.
    
    This prevents false positives when a PID is reused by an unrelated process
    after the original gateway crashed without cleanup.
    """
    # Patterns that indicate a hermes gateway process
    gateway_patterns = (
        "hermes_cli.main gateway",
        "hermes gateway",
        "gateway/run.py",
        "gateway.run",
    )
    
    try:
        # On Linux, check /proc/{pid}/cmdline directly (fast, no subprocess)
        if sys.platform.startswith("linux"):
            cmdline_path = Path(f"/proc/{pid}/cmdline")
            if cmdline_path.exists():
                cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
                return any(p in cmdline for p in gateway_patterns)
        
        # macOS / fallback: use ps command
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            cmdline = result.stdout.strip()
            return any(p in cmdline for p in gateway_patterns)
        
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError, OSError):
        pass
    
    # If we can't determine, assume it's NOT a gateway (safer to allow startup)
    return False


def get_running_pid() -> Optional[int]:
    """Return the PID of a running gateway instance, or ``None``.

    Checks the PID file and verifies:
    1. The process is actually alive
    2. The process is actually a hermes gateway (not a PID collision)
    
    Cleans up stale PID files automatically.
    """
    pid_path = _get_pid_path()
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal sent
        
        # Process exists — verify it's actually a hermes gateway
        if not _is_hermes_gateway_process(pid):
            # PID exists but it's not a hermes gateway — stale PID file
            remove_pid_file()
            return None
        
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        # Stale PID file — process is gone
        remove_pid_file()
        return None


def is_gateway_running() -> bool:
    """Check if the gateway daemon is currently running."""
    return get_running_pid() is not None

"""Credential proxy daemon lifecycle management.

start()     — spawn the proxy as a detached background process, wait for ready
stop()      — SIGTERM the daemon, clean up state files
status()    — return {running, pid, socket_path}
is_running()— quick bool check used by other components
"""

import asyncio
import logging
import os
import signal
import sys

from hermes_constants import get_hermes_home

_STATE_DIR = get_hermes_home() / "state"
_PID_FILE = _STATE_DIR / "cred-proxy.pid"
_SOCKET_PATH = _STATE_DIR / "cred-proxy.sock"
_LOG_FILE = _STATE_DIR / "cred-proxy.log"


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------

def _write_pid() -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))


def _read_pid() -> int | None:
    try:
        return int(_PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return None


def _remove_pid() -> None:
    try:
        _PID_FILE.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Socket path helper
# ---------------------------------------------------------------------------

def get_socket_path() -> str:
    """Return the Unix socket path for the credential proxy."""
    return str(_SOCKET_PATH)


# ---------------------------------------------------------------------------
# State cleanup
# ---------------------------------------------------------------------------

def _cleanup_state_files() -> None:
    _remove_pid()
    try:
        _SOCKET_PATH.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_running() -> bool:
    """Return True if the credential proxy daemon is running.

    Checks that PID file exists, the process is alive, and the socket file
    exists.  Removes stale files if the process is dead.
    """
    pid = _read_pid()
    if pid is None:
        return False
    if not _SOCKET_PATH.exists():
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        _cleanup_state_files()
        return False
    except PermissionError:
        # Process exists but we can't send signals — still running
        return True


def status() -> dict:
    """Return {running: bool, pid: int|None, socket_path: str}."""
    running = is_running()
    return {
        "running": running,
        "pid": _read_pid() if running else None,
        "socket_path": str(_SOCKET_PATH),
    }


def stop() -> None:
    """Send SIGTERM to the daemon and remove state files."""
    pid = _read_pid()
    if pid is None:
        print("Credential proxy is not running.")
        return
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped credential proxy (PID {pid}).")
    except ProcessLookupError:
        print("Credential proxy process not found (already stopped?).")
    except PermissionError:
        print(f"Permission denied when signalling PID {pid}.")
        return  # Process is still alive — leave state files intact
    _cleanup_state_files()


def start() -> None:
    """Start the credential proxy daemon as a detached background process.

    Spawns ``python -m cred_proxy`` with start_new_session=True so it
    survives the calling process exiting.  Waits up to 3 s for the daemon
    to write its PID file and create the socket before returning.
    """
    if is_running():
        print("Credential proxy is already running.")
        return

    import subprocess
    import time

    cmd = [sys.executable, "-m", "cred_proxy"]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        print(f"Failed to start credential proxy: {exc}")
        return

    # Wait up to 3 s for the daemon to write its PID file and bind the socket
    for _ in range(30):
        time.sleep(0.1)
        if is_running():
            pid = _read_pid()
            print(f"Credential proxy started (PID {pid}).")
            return

    print("Warning: Could not confirm credential proxy started. Check logs at:")
    print(f"  {_LOG_FILE}")


# ---------------------------------------------------------------------------
# Internal: run the server (called from __main__.py)
# ---------------------------------------------------------------------------

def _run_server() -> None:
    """Configure logging and run the asyncio HTTP proxy (blocks forever).

    Binds a Unix domain socket at ``_SOCKET_PATH``.  The on_started callback
    writes the PID file once the server is bound, so callers polling
    is_running() only see the daemon as ready once it is live.
    """
    from .server import run_proxy

    _STATE_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(_LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    def _on_sigterm(signum, frame):
        # sys.exit() raises SystemExit, unwinding asyncio.run() into the finally block.
        sys.exit(0)

    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except (OSError, ValueError):
        pass  # Windows or restricted environment

    def _on_started(socket_path: str) -> None:
        """Called once the proxy server is bound and listening."""
        _write_pid()

    try:
        asyncio.run(run_proxy(unix_socket=str(_SOCKET_PATH), on_started=_on_started))
    finally:
        _cleanup_state_files()

"""Credential proxy daemon lifecycle management.

Provides ``hermes cred-proxy start|stop|status`` subcommands.
Mirrors the structure of ``hermes_cli/gateway.py`` but much simpler —
the proxy is a lightweight asyncio process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from proxy.config import get_proxy_pid_path, get_proxy_socket_path, is_proxy_enabled


def _is_running() -> int | None:
    """Return the PID if the proxy daemon is running, else None."""
    pid_path = get_proxy_pid_path()
    if not pid_path.exists():
        return None
    try:
        pid = int(pid_path.read_text().strip())
        os.kill(pid, 0)  # check if alive
        return pid
    except (ValueError, OSError):
        # Stale PID file
        pid_path.unlink(missing_ok=True)
        return None


def _start() -> None:
    """Start the credential proxy daemon."""
    if not is_proxy_enabled():
        print("Credential proxy is not enabled in config.yaml.", file=sys.stderr)
        print("Add to your config.yaml:", file=sys.stderr)
        print("  credential_proxy:", file=sys.stderr)
        print("    enabled: true", file=sys.stderr)
        sys.exit(1)

    pid = _is_running()
    if pid is not None:
        print(f"Credential proxy already running (PID {pid}).")
        return

    socket_path = get_proxy_socket_path()
    pid_path = get_proxy_pid_path()

    # Ensure state dir exists
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    # Spawn the proxy as a detached subprocess
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "proxy.daemon",
            "--socket", str(socket_path),
            "--pid-file", str(pid_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    print(f"Credential proxy started (PID {proc.pid}).")
    print(f"Socket: {socket_path}")


def _stop() -> None:
    """Stop the credential proxy daemon."""
    pid = _is_running()
    if pid is None:
        print("Credential proxy is not running.")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Credential proxy stopped (PID {pid}).")
    except OSError as exc:
        print(f"Failed to stop proxy (PID {pid}): {exc}", file=sys.stderr)

    # Clean up
    get_proxy_pid_path().unlink(missing_ok=True)
    socket_path = get_proxy_socket_path()
    if socket_path.exists():
        socket_path.unlink(missing_ok=True)


def _status() -> None:
    """Show credential proxy status."""
    pid = _is_running()
    socket_path = get_proxy_socket_path()
    enabled = is_proxy_enabled()

    if pid:
        print(f"Status:  running (PID {pid})")
    else:
        print("Status:  stopped")

    print(f"Enabled: {enabled}")
    print(f"Socket:  {socket_path}")

    if socket_path.exists():
        print("Socket:  exists ✓")
    else:
        print("Socket:  not found")


def cred_proxy_command(args) -> None:
    """Dispatch ``hermes cred-proxy`` subcommands.

    Args:
        args: argparse Namespace with ``cred_proxy_command`` attribute.
    """
    subcmd = getattr(args, "cred_proxy_command", None)
    if not subcmd:
        print("Usage: hermes cred-proxy <start|stop|status>", file=sys.stderr)
        sys.exit(1)

    subcmd = subcmd.lower()
    if subcmd == "start":
        _start()
    elif subcmd == "stop":
        _stop()
    elif subcmd == "status":
        _status()
    else:
        print(f"Unknown subcommand: {subcmd}", file=sys.stderr)
        print("Usage: hermes cred-proxy <start|stop|status>", file=sys.stderr)
        sys.exit(1)

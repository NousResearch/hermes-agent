"""Detached-process fallback when user-scope systemd is unreachable (Linux).

Port of the idea in qwibitai/nanoclaw#3768/#3760: when the host has systemd as PID 1 but
``systemctl --user`` cannot reach a user instance (no linger, SSH session without a user
bus, polkit denying ``enable-linger``), setup used to install the unit and then abort the
start with a wall of D-Bus remediation, leaving the gateway stopped — while the same host
on macOS 26+ gets a CLI-managed detached process (``_launchd_fallback_to_detached``).
This module gives Linux the same fallback, and marks it so ``hermes gateway status`` can
explain why the unit is inactive while a gateway is running.

Lives beside :mod:`hermes_cli.gateway` (facade); imports from it happen inside functions
to avoid a module-level cycle.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

from hermes_cli.config import get_hermes_home

_MARKER_NAME = ".gateway-systemd-user-unavailable"
# The detached child claims gateway.pid/gateway.lock once its event loop is up; on a cold
# venv that import chain is several seconds, so give it a real budget (nanoclaw waits 30s).
_START_TIMEOUT_S = 30.0


def marker_path() -> Path:
    return get_hermes_home() / _MARKER_NAME


def write_marker(reason: str) -> None:
    from datetime import datetime, timezone
    payload = {"written_at": datetime.now(timezone.utc).isoformat(), "reason": reason}
    with contextlib.suppress(OSError):
        marker_path().write_text(json.dumps(payload), encoding="utf-8")


def clear_marker() -> None:
    """A later successful ``systemctl --user start`` means the OS recovered; forget the fallback."""
    with contextlib.suppress(OSError):
        marker_path().unlink(missing_ok=True)


def marker_exists() -> bool:
    return marker_path().exists()


def wait_for_detached_gateway(timeout: float = _START_TIMEOUT_S) -> int | None:
    """PID once the spawned gateway owns the runtime lock + pid record, else None at *timeout*.
    A Popen that returned is not a running gateway (config errors exit within a second)."""
    from gateway.status import get_running_pid
    deadline = time.monotonic() + timeout
    while True:
        pid = get_running_pid()
        if pid is not None:
            return pid
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def fallback_to_detached(reason: str, *, exit_on_failure: bool = True) -> bool:
    """Start the gateway as a CLI-managed detached process because user systemd is unreachable.

    Prints what happened and what is lost (no auto-start at login / no crash restart), records
    the marker for ``status``, and waits until the child has actually claimed the pid file — a
    fallback that "started" a gateway which died on its first config read would be worse than
    the original error. Returns True when a live gateway pid was observed."""
    from hermes_cli import gateway as gw
    from hermes_constants import display_hermes_home as _dhh

    print(f"⚠ User systemd is not reachable in this session ({reason}).")
    print("  Starting the gateway as a background process instead.")
    if gw._spawn_detached_gateway():
        pid = wait_for_detached_gateway()
        if pid is not None:
            write_marker(reason)
            print(f"✓ Started gateway as a detached background process (PID {pid})")
            print("  It will NOT auto-start at login or auto-restart on crash until user systemd is available.")
            print(f"  Fix for a supervised service:  sudo loginctl enable-linger $USER  # then: hermes gateway start")
            print(f"  Logs: {_dhh()}/logs/gateway.log")
            print("  Stop it with: hermes gateway stop")
            return True
        print(f"✗ The detached gateway exited before claiming its pid file; see {_dhh()}/logs/gateway.error.log")
    else:
        print("✗ Failed to start the gateway as a background process.")
    print(f"  Try manually: nohup hermes gateway run --replace > {_dhh()}/logs/gateway.log 2>&1 &")
    if exit_on_failure:
        sys.exit(1)
    return False

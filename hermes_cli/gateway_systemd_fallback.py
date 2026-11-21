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
    from datetime import UTC, datetime
    payload = {"written_at": datetime.now(UTC).isoformat(), "reason": reason}
    with contextlib.suppress(OSError):
        marker_path().write_text(json.dumps(payload), encoding="utf-8")


def clear_marker() -> None:
    """A later successful ``systemctl --user start`` means the OS recovered; forget the fallback."""
    with contextlib.suppress(OSError):
        marker_path().unlink(missing_ok=True)


def marker_exists() -> bool:
    return marker_path().exists()


def marker_supervisor() -> str | None:
    """``"user systemd"`` while this fallback's marker is set (``status`` names who cannot supervise)."""
    return "user systemd" if marker_exists() else None


def raise_user_systemd_unavailable(username: str, *, reason: str, fix_hint: str) -> None:
    """Build a user-facing error message and raise UserSystemdUnavailableError."""
    from hermes_cli import gateway as gw
    msg = (
        f"{reason}\n"
        "  systemctl --user cannot reach the user D-Bus session in this shell.\n"
        "\n"
        "  To fix:\n"
        f"{fix_hint}\n"
        "\n"
        "  Alternative: run the gateway in the foreground (stays up until\n"
        "  you exit / close the terminal):\n"
        "    hermes gateway run"
    )
    raise gw.UserSystemdUnavailableError(msg)


def _first_line(exc: BaseException) -> str:
    return str(exc).splitlines()[0].strip() if str(exc) else type(exc).__name__


def install_deferred(unit_path: Path) -> bool:
    """User-scope install with no reachable user bus: True after leaving the unit on disk unenabled.
    daemon-reload/enable would die with "Failed to connect to bus" and abort setup; systemd picks the
    unit up once the user instance exists, and ``systemd_start`` degrades to a detached process."""
    from hermes_cli import gateway as gw
    try:
        gw._preflight_user_systemd()  # tries enable-linger first; a user bus may appear
    except gw.UserSystemdUnavailableError as exc:
        gw.print_warning(f"  User systemd is not reachable in this session ({_first_line(exc)})")
        gw.print_info(f"  Unit written to {unit_path} but not enabled yet. Enable later:")
        gw.print_info(f"    systemctl --user daemon-reload && systemctl --user enable {gw.get_service_name()}")
        return True
    return False


def start_preamble_or_detach(system: bool) -> bool | None:
    """``systemd_start``'s scope preamble; ``None`` once a detached gateway was started instead.
    Same policy as launchd on macOS 26+ (``_launchd_degrade_or_raise``): a host whose service manager
    cannot supervise us still gets a running gateway, honestly labelled. Only when a user unit of THIS
    install exists — "not installed" keeps its own message, another install's unit is refused."""
    from hermes_cli import gateway as gw
    try:
        return gw._systemd_scope_preamble("start", system, preflight_user=True)
    except gw.UserSystemdUnavailableError as exc:
        if not gw.get_systemd_unit_path(system=False).exists():
            raise
        gw.gateway_service_owner.refuse_foreign_systemd_unit("start", False)
        fallback_to_detached(_first_line(exc), exit_on_failure=False)
        return None


def print_detached_status(scope_label: str, start_cmd: str) -> bool:
    """``status`` for an inactive user unit: True after explaining a running detached fallback gateway."""
    from gateway.status import get_running_pid
    if not (marker_exists() and (pid := get_running_pid(cleanup_stale=False))):
        return False
    print(f"⚠ {scope_label} gateway service is stopped, but a detached fallback gateway is running (PID {pid})")
    print("  User systemd was unreachable when it started; auto-start at login and crash restart are NOT available.")
    print(f"  Fix for a supervised service:  sudo loginctl enable-linger $USER  # then: {start_cmd}")
    return True


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
            print("  Fix for a supervised service:  sudo loginctl enable-linger $USER  # then: hermes gateway start")
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

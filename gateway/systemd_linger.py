"""Systemd linger management for gateway and worker persistence."""
from __future__ import annotations

import getpass
import shutil
import subprocess
import time
from pathlib import Path

from gateway import service_identity, systemd_runtime
from hermes_constants import is_termux

_CAPTURE_TEXT = dict(
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)


def _enable_linger(username: str) -> subprocess.CompletedProcess:
    """Run loginctl enable-linger for one user."""
    return subprocess.run(
        ["loginctl", "enable-linger", username],
        check=False,
        timeout=30,
        **_CAPTURE_TEXT,
    )


def _completed_process_detail(result) -> str:
    return (result.stderr or result.stdout or f"exit {result.returncode}").strip()


def _target_bus_ready(uid: int, timeout: float = 5.0) -> bool:
    """Wait for another account's D-Bus without adopting it into this process."""
    bus = Path(f"/run/user/{uid}/bus")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if bus.exists():
                return True
        except OSError:
            pass
        time.sleep(0.2)
    try:
        return bus.exists()
    except OSError:
        return False


def _lookup_uid(username: str) -> int:
    import pwd

    return pwd.getpwnam(username).pw_uid


def _print_enable_warning(
    username: str,
    detail: str | None = None,
    *,
    system: bool = False,
) -> None:
    print()
    if system:
        print(
            f"⚠ Linger not enabled for {username} — cron and Kanban workers "
            "cannot start (no user D-Bus)."
        )
    else:
        print("⚠ Linger not enabled — gateway may stop when you close this terminal.")
    if detail:
        print(f"  Auto-enable failed: {detail}")
    print()
    print(
        "  Enable it manually:"
        if system
        else "  On headless servers (VPS, cloud instances) run:"
    )
    print(f"    sudo loginctl enable-linger {username}")
    print()
    print("  Then restart the gateway:")
    sudo = "sudo " if system else ""
    user_flag = "" if system else "--user "
    print(f"    {sudo}systemctl {user_flag}restart {service_identity.service_name()}.service")
    print()


def ensure_linger_enabled(
    username: str | None = None,
    *,
    system: bool = False,
) -> bool:
    """Ensure linger for a user; return True only when this call enabled it."""
    if is_termux() or not systemd_runtime.is_linux():
        return False

    if username is None:
        username = getpass.getuser()
    enabled_msg = (
        f"✓ Systemd linger is enabled for {username} (worker D-Bus available)"
        if system
        else "✓ Systemd linger is enabled (service survives logout)"
    )
    if Path(f"/var/lib/systemd/linger/{username}").exists():
        print(enabled_msg)
        return False

    linger_enabled, linger_detail = systemd_runtime.linger_status(username)
    if linger_enabled is True:
        print(enabled_msg)
        return False

    if not shutil.which("loginctl"):
        _print_enable_warning(
            username,
            linger_detail or "loginctl not found",
            system=system,
        )
        return False

    if system:
        print(
            f"Enabling linger for {username} so cron and Kanban workers can "
            "reach systemd-run --user..."
        )
    else:
        print("Enabling linger so the gateway survives SSH logout...")
    try:
        result = _enable_linger(username)
    except Exception as exc:
        _print_enable_warning(username, str(exc), system=system)
        return False

    if result.returncode != 0:
        _print_enable_warning(
            username,
            _completed_process_detail(result) or linger_detail,
            system=system,
        )
        return False
    print(
        f"✓ Enabled linger for {username}"
        if system
        else "✓ Linger enabled — gateway will persist after logout"
    )
    return True


def ensure_system_service_linger(username: str) -> None:
    """Provision the target user's user manager for restart-safe workers."""
    if not ensure_linger_enabled(username, system=True):
        return
    uid = _lookup_uid(username)
    if _target_bus_ready(uid):
        print(
            f"✓ /run/user/{uid}/bus is up — cron and Kanban workers can use "
            "systemd-run --user"
        )
    else:
        print(f"⚠ /run/user/{uid}/bus did not appear within 5s.")
        print(f"  Start the user manager: sudo systemctl start user@{uid}.service")
    if systemd_runtime.unit_is_active(system=True):
        print(
            "  The running gateway was started without a user D-Bus; "
            "restart it to pick one up:"
        )
        print(f"    sudo systemctl restart {service_identity.service_name()}.service")

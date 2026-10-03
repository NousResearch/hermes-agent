"""Systemd runtime discovery and command execution."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gateway.service_identity import service_name
from gateway.systemd_identity import unit_path
from hermes_constants import is_container, is_termux, is_wsl

_CAPTURE_TEXT = dict(
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
)


class UserSystemdUnavailableError(RuntimeError):
    """systemctl --user cannot reach the current user's systemd instance."""


class SystemctlUnavailableError(RuntimeError):
    """systemctl is not installed."""

    def __init__(self) -> None:
        super().__init__("systemctl is not available on this system")


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def _path_exists_safe(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _runtime_dir_is_ours(runtime_dir: str) -> bool:
    try:
        return Path(runtime_dir).stat().st_uid == os.getuid()
    except OSError:
        return False


def _user_runtime_dir() -> Path:
    return Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")


def _user_systemd_socket_ready() -> bool:
    runtime = _user_runtime_dir()
    return _path_exists_safe(runtime / "bus") or _path_exists_safe(runtime / "systemd" / "private")


def ensure_user_env() -> None:
    uid = os.getuid()
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if (not xdg or not _runtime_dir_is_ours(xdg)) and _runtime_dir_is_ours(f"/run/user/{uid}"):
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    if "DBUS_SESSION_BUS_ADDRESS" not in os.environ:
        bus_path = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{uid}")) / "bus"
        if _path_exists_safe(bus_path):
            os.environ["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"


def systemctl_cmd(system: bool = False) -> list[str]:
    if not system:
        ensure_user_env()
    return ["systemctl"] if system else ["systemctl", "--user"]


def run_systemctl(
    args: list[str],
    *,
    system: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(systemctl_cmd(system) + args, **kwargs)
    except FileNotFoundError:
        raise SystemctlUnavailableError() from None


def _systemd_operational(system: bool = False) -> bool:
    try:
        result = run_systemctl(
            ["is-system-running"],
            system=system,
            timeout=5,
            **_CAPTURE_TEXT,
        )
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return False
    return result.stdout.strip().lower() in {
        "running", "degraded", "starting", "initializing",
    }


def supports_services() -> bool:
    if not is_linux() or is_termux() or shutil.which("systemctl") is None:
        return False
    if is_wsl():
        return _systemd_operational(system=True)
    if is_container():
        return _systemd_operational(False) or _systemd_operational(True)
    return True


def unit_is_active(system: bool = False) -> bool:
    from gateway.service_identity import service_name
    from gateway.systemd_identity import unit_path

    if not unit_path(system).exists():
        return False
    try:
        result = run_systemctl(
            ["is-active", service_name()],
            system=system,
            timeout=10,
            **_CAPTURE_TEXT,
        )
    except (RuntimeError, subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "active"


def installed_scopes() -> list[str]:
    scopes: list[str] = []
    seen: set[Path] = set()
    for system, label in ((False, "user"), (True, "system")):
        path = unit_path(system=system)
        if path not in seen and path.exists():
            scopes.append(label)
            seen.add(path)
    return scopes


def has_conflicting_units() -> bool:
    return len(installed_scopes()) > 1


def select_scope(system: bool = False) -> bool:
    return system or (unit_path(True).exists() and not unit_path(False).exists())


def _completed_process_detail(result) -> str:
    return (result.stderr or result.stdout or f"exit {result.returncode}").strip()


def linger_status(username: str | None = None) -> tuple[bool | None, str]:
    if is_termux():
        return None, "not supported in Termux"
    if not is_linux():
        return None, "not supported on this platform"
    if not shutil.which("loginctl"):
        return None, "loginctl not found"
    if username is None:
        username = os.getenv("USER") or os.getenv("LOGNAME")
    if not username:
        try:
            import pwd
            username = pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            return None, "could not determine current user"
    try:
        result = subprocess.run(
            ["loginctl", "show-user", username, "--property=Linger", "--value"],
            check=False,
            timeout=10,
            **_CAPTURE_TEXT,
        )
    except Exception as exc:
        return None, str(exc)
    if result.returncode != 0:
        return None, _completed_process_detail(result) or "loginctl query failed"
    value = (result.stdout or "").strip().lower()
    if value in {"yes", "true", "1"}:
        return True, ""
    if value in {"no", "false", "0"}:
        return False, ""
    return None, f"unexpected loginctl output: {value or '<empty>'}"


def _wait_for_user_socket(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _user_systemd_socket_ready():
            ensure_user_env()
            return True
        time.sleep(0.2)
    return _user_systemd_socket_ready()


def preflight_user(*, auto_enable_linger: bool = True) -> None:
    ensure_user_env()
    if _user_systemd_socket_ready():
        return
    import getpass
    username = getpass.getuser()
    enabled, detail = linger_status()
    sudo_hint = f"  sudo loginctl enable-linger {username}"
    if enabled is True:
        if _wait_for_user_socket(3.0):
            return
        raise UserSystemdUnavailableError(
            "User systemd control sockets are missing even though linger is enabled.\n"
            "  systemctl --user cannot reach the user D-Bus session in this shell.\n\n"
            "  To fix:\n"
            f"  systemctl start user@{os.getuid()}.service\n"
            "  (may require sudo; try again after the command succeeds)\n\n"
            "  Alternative: run the gateway in the foreground (stays up until\n"
            "  you exit / close the terminal):\n"
            "    hermes gateway run"
        )
    if auto_enable_linger and shutil.which("loginctl"):
        try:
            result = subprocess.run(
                ["loginctl", "enable-linger", username],
                check=False,
                timeout=30,
                **_CAPTURE_TEXT,
            )
        except Exception as exc:
            reason = f"loginctl enable-linger failed ({exc})."
        else:
            if result.returncode == 0 and _wait_for_user_socket(5.0):
                print(f"✓ Enabled linger for {username} — user D-Bus now available")
                return
            reason = (
                "Linger was enabled, but the user D-Bus socket did not appear."
                if result.returncode == 0
                else f"loginctl enable-linger was denied: {_completed_process_detail(result)}"
            )
    else:
        reason = f"User D-Bus session is not available ({detail or 'linger disabled'})."
    raise UserSystemdUnavailableError(
        f"{reason}\n"
        "  systemctl --user cannot reach the user D-Bus session in this shell.\n\n"
        "  To fix:\n"
        f"{sudo_hint}\n\n"
        "  Alternative: run the gateway in the foreground (stays up until\n"
        "  you exit / close the terminal):\n"
        "    hermes gateway run"
    )


def scope_label(system: bool = False) -> str:
    return "system" if system else "user"


def sync_home_from_unit(system: bool) -> None:
    if not system:
        return
    from gateway.service_identity import hermes_home_pinned_by_unit
    home = (hermes_home_pinned_by_unit(unit_path(True)) or "").strip()
    if not home:
        result = run_systemctl(
            ["show", service_name(), "--no-pager", "--property", "Environment"],
            system=True,
            timeout=10,
            **_CAPTURE_TEXT,
        )
        if result.returncode == 0:
            for token in result.stdout.split():
                if token.startswith("HERMES_HOME="):
                    home = token.split("=", 1)[1].strip()
                    break
    if home and os.environ.get("HERMES_HOME", "").strip() != home:
        os.environ["HERMES_HOME"] = home

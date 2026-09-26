"""Systemd service identity and unit-path primitives."""
from __future__ import annotations

import getpass
import os
from pathlib import Path

from gateway import service_identity


class SystemScopeRequiresRootError(RuntimeError):
    """System-scope gateway operation attempted as non-root."""

    def __str__(self) -> str:
        return self.args[0] if self.args else ""


def user_unit_dir() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "systemd" / "user"


def unit_path(system: bool = False) -> Path:
    base = service_identity.SYSTEM_UNIT_DIR if system else user_unit_dir()
    return base / f"{service_identity.service_name()}.service"


def require_root(action: str) -> None:
    if os.geteuid() != 0:
        raise SystemScopeRequiresRootError(
            f"System gateway {action} requires root. Re-run with sudo.",
            action,
        )


def system_service_identity(
    run_as_user: str | None = None,
) -> tuple[str, str, str, int]:
    username = (
        run_as_user
        or os.getenv("SUDO_USER")
        or os.getenv("USER")
        or os.getenv("LOGNAME")
        or getpass.getuser()
    ).strip()
    if not username:
        raise ValueError("Could not determine which user the gateway service should run as")
    if username == "root" and not run_as_user:
        raise ValueError(
            "Refusing to install the gateway system service as root; "
            "pass --run-as-user root to override (e.g. in LXC containers)"
        )
    if username == "root":
        print("⚠ Installing gateway service to run as root.")
        print("  This is fine for LXC/container environments but not recommended on bare-metal hosts.")
    import grp
    import pwd

    try:
        user_info = pwd.getpwnam(username)
    except KeyError as exc:
        raise ValueError(f"Unknown user: {username}") from exc
    return (
        username,
        grp.getgrgid(user_info.pw_gid).gr_name,
        user_info.pw_dir,
        user_info.pw_uid,
    )


def read_unit_user(path: Path) -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("User="):
            return line.split("=", 1)[1].strip() or None
    return None


def pinned_home(path: Path) -> str | None:
    return service_identity.hermes_home_pinned_by_unit(path)

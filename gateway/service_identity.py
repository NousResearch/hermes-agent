"""Gateway service identity and profile-to-service naming.

This module owns the stable service-name mapping shared by systemd, launchd,
Windows service discovery, and explicit runtime-service lookup.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from hermes_constants import (
    _get_platform_default_hermes_home,
    get_default_hermes_root,
    get_hermes_home,
    profile_name_for_home,
    sudo_invoker_default_home,
)
from profiles.paths import profile_name_from_home

SERVICE_BASE = "hermes-gateway"
SYSTEM_UNIT_DIR = Path("/etc/systemd/system")


def unit_environment_value(unit_path: Path, name: str) -> str | None:
    """Read one Environment NAME=value directive from a systemd unit."""
    try:
        text = unit_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        body = line.strip()
        if not body.startswith("Environment="):
            continue
        body = body[len("Environment="):].strip()
        if body.startswith('"') and body.endswith('"'):
            body = body[1:-1].replace(r'\\"', '"').replace(r"\\\\", "\\").replace("%%", "%")
        if body.startswith(f"{name}="):
            return body.split("=", 1)[1].strip() or None
    return None


def hermes_home_pinned_by_unit(unit_path: Path) -> str | None:
    return unit_environment_value(unit_path, "HERMES_HOME")


def _native_service_homes() -> set[Path]:
    homes = {_get_platform_default_hermes_home().resolve()}
    sudo_home = sudo_invoker_default_home()
    if sudo_home is not None:
        homes.add(sudo_home.resolve())
    return homes


def _bare_unit_pinned_home() -> Path | None:
    """Home pinned by the bare system unit when root on Linux."""
    if not sys.platform.startswith("linux") or os.geteuid() != 0:
        return None
    pinned = hermes_home_pinned_by_unit(SYSTEM_UNIT_DIR / f"{SERVICE_BASE}.service")
    if not pinned:
        return None
    try:
        return Path(pinned).expanduser().resolve()
    except (RuntimeError, ValueError):
        return None


def service_suffix() -> str:
    """Suffix for the current HERMES_HOME; empty means the bare service name."""
    home = get_hermes_home().resolve()
    if home in _native_service_homes() or home == _bare_unit_pinned_home():
        return ""
    name = profile_name_from_home(home, get_default_hermes_root().resolve())
    return name or hashlib.sha256(str(home).encode()).hexdigest()[:8]


def service_suffix_for_home(home: Path) -> str:
    """Service suffix for an explicit canonical home, independent of caller env."""
    default = _get_platform_default_hermes_home().resolve()
    home = home.resolve()
    if home == default:
        return ""
    root = default
    if not home.is_relative_to(default):
        root = home.parent.parent if home.parent.name == "profiles" else home
    if home != root:
        name = profile_name_from_home(home, root)
        if name:
            return name
    return hashlib.sha256(str(home).encode()).hexdigest()[:8]


def current_profile_name() -> str:
    """Canonical current profile id, falling back to the service suffix."""
    return profile_name_for_home(get_hermes_home()) or service_suffix()


def profile_arg(
    hermes_home: str | None = None,
    default_root: str | Path | None = None,
) -> str:
    """Return --profile <name> for a named profile home, else an empty string."""
    home = Path(hermes_home or str(get_hermes_home())).resolve()
    default = Path(default_root).resolve() if default_root else get_default_hermes_root().resolve()
    if home == default:
        return ""
    name = profile_name_from_home(home, default)
    return f"--profile {name}" if name else ""


def service_name() -> str:
    suffix = service_suffix()
    return f"{SERVICE_BASE}-{suffix}" if suffix else SERVICE_BASE

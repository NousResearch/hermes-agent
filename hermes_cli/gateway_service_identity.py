"""Gateway service identity helpers.

This module is intentionally small and import-safe so both the main
``hermes gateway`` command and the standalone ``scripts/hermes-gateway`` entry
point can share service naming without importing the full CLI gateway module.
"""

from __future__ import annotations

import hashlib
import html
import os
import plistlib
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from hermes_constants import get_hermes_home


_SERVICE_BASE = "hermes-gateway"
_LAUNCHD_LABEL_BASE = "ai.hermes.gateway"
_PROFILE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class GatewayServiceIdentity:
    """Names for the gateway service under the active Hermes home."""

    suffix: str
    systemd_service_name: str
    launchd_label: str


def _resolved_hermes_home(hermes_home: str | Path | None = None) -> Path:
    return Path(hermes_home).resolve() if hermes_home is not None else get_hermes_home().resolve()


def _native_default_hermes_homes() -> set[Path]:
    candidates: set[Path] = set()

    try:
        candidates.add((Path.home() / ".hermes").resolve())
    except OSError:
        pass

    try:
        import pwd

        user_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        candidates.add((user_home / ".hermes").resolve())

        sudo_user = os.environ.get("SUDO_USER", "").strip()
        if sudo_user and sudo_user != "root":
            sudo_home = Path(pwd.getpwnam(sudo_user).pw_dir)
            candidates.add((sudo_home / ".hermes").resolve())
    except (ImportError, KeyError, OSError):
        pass

    return candidates


def gateway_profile_name(hermes_home: str | Path | None = None) -> str | None:
    """Return the profile name when *hermes_home* is ``<root>/profiles/<name>``."""
    home = _resolved_hermes_home(hermes_home)
    name = home.name
    if home.parent.name == "profiles" and _PROFILE_NAME_RE.match(name):
        return name
    return None


def gateway_service_suffix(hermes_home: str | Path | None = None) -> str:
    """Return the stable suffix for service names under *hermes_home*.

    The default ``~/.hermes`` keeps the historical unsuffixed service names.
    Named profiles use their profile name. Any other custom Hermes home gets a
    short hash so multiple installations can coexist without service collisions.
    """
    home = _resolved_hermes_home(hermes_home)
    if home in _native_default_hermes_homes():
        return ""

    profile_name = gateway_profile_name(home)
    if profile_name:
        return profile_name

    return hashlib.sha256(str(home).encode()).hexdigest()[:8]


def is_custom_root_hermes_home(hermes_home: str | Path | None = None) -> bool:
    """Return True when *hermes_home* is a non-default, non-profile root."""
    home = _resolved_hermes_home(hermes_home)
    return (
        home not in _native_default_hermes_homes()
        and gateway_profile_name(home) is None
    )


def get_gateway_service_identity(
    hermes_home: str | Path | None = None,
) -> GatewayServiceIdentity:
    """Return systemd and launchd names for *hermes_home*."""
    suffix = gateway_service_suffix(hermes_home)
    return GatewayServiceIdentity(
        suffix=suffix,
        systemd_service_name=(
            f"{_SERVICE_BASE}-{suffix}" if suffix else _SERVICE_BASE
        ),
        launchd_label=(
            f"{_LAUNCHD_LABEL_BASE}-{suffix}" if suffix else _LAUNCHD_LABEL_BASE
        ),
    )


def gateway_profile_arg(hermes_home: str | Path | None = None) -> str:
    """Return ``--profile <name>`` for named profile homes, otherwise ``""``."""
    profile_name = gateway_profile_name(hermes_home)
    return f"--profile {profile_name}" if profile_name else ""


def gateway_launchd_plist_path(
    user_home: str | Path | None = None,
    hermes_home: str | Path | None = None,
) -> Path:
    """Return the launchd plist path for *hermes_home* under *user_home*."""
    home = Path(user_home) if user_home is not None else Path.home()
    label = get_gateway_service_identity(hermes_home).launchd_label
    return home / "Library" / "LaunchAgents" / f"{label}.plist"


def _gateway_script_project_dirs(text: str) -> set[Path]:
    dirs: set[Path] = set()
    for match in re.finditer(r"(/[^\s\"'<]+/scripts/hermes-gateway)(?:[\s\"'<]|$)", text):
        script_path = Path(match.group(1))
        if script_path.name == "hermes-gateway" and script_path.parent.name == "scripts":
            dirs.add(script_path.parent.parent)
    return dirs


def _project_env_matches_hermes_home(project_dir: Path, resolved_home: Path) -> bool:
    env_path = project_dir / ".env"
    if not env_path.exists():
        return False

    try:
        from dotenv import dotenv_values

        value = (dotenv_values(env_path).get("HERMES_HOME") or "").strip()
    except Exception:
        return False
    if not value:
        return False

    configured_home = Path(value).expanduser()
    if not configured_home.is_absolute():
        configured_home = project_dir / configured_home

    try:
        return configured_home.resolve() == resolved_home
    except OSError:
        return False


def _gateway_script_definition_matches_hermes_home(
    text: str,
    hermes_home: str | Path | None = None,
) -> bool:
    resolved_home = _resolved_hermes_home(hermes_home)
    return any(
        _project_env_matches_hermes_home(project_dir, resolved_home)
        for project_dir in _gateway_script_project_dirs(text)
    )


def _hermes_home_value_matches(value: object, resolved_home: Path) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        return Path(text).expanduser().resolve() == resolved_home
    except OSError:
        return False


def _explicit_hermes_home_values(text: str) -> list[str]:
    values: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Environment="):
            continue
        try:
            env_values = shlex.split(stripped.split("=", 1)[1])
        except ValueError:
            env_values = []
        for value in env_values:
            if value.startswith("HERMES_HOME="):
                values.append(value.split("=", 1)[1])

    try:
        payload = plistlib.loads(text.encode("utf-8"))
    except Exception:
        payload = None
    if isinstance(payload, dict):
        env = payload.get("EnvironmentVariables")
        if isinstance(env, dict) and "HERMES_HOME" in env:
            values.append(str(env.get("HERMES_HOME") or ""))

    values.extend(
        html.unescape(match)
        for match in re.findall(
            r"<key>\s*HERMES_HOME\s*</key>\s*<string>(.*?)</string>",
            text,
            flags=re.S,
        )
    )
    return values


def service_definition_declares_hermes_home(path: Path) -> bool:
    """Return True when a service definition explicitly sets HERMES_HOME."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, PermissionError):
        return False
    return bool(_explicit_hermes_home_values(text))


def service_definition_matches_hermes_home(
    path: Path,
    hermes_home: str | Path | None = None,
) -> bool:
    """Return True when a systemd unit or launchd plist targets *hermes_home*."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, PermissionError):
        return False
    resolved_home = _resolved_hermes_home(hermes_home)

    explicit_values = _explicit_hermes_home_values(text)
    if explicit_values:
        return any(
            _hermes_home_value_matches(value, resolved_home)
            for value in explicit_values
        )

    if _gateway_script_definition_matches_hermes_home(text, resolved_home):
        return True

    return False

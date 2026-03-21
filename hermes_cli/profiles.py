"""
Profile management for multiple isolated Hermes instances.

Each profile is a fully independent HERMES_HOME directory with its own
config.yaml, .env, memory, sessions, skills, gateway, cron, and logs.
Profiles live under ``~/.hermes/profiles/<name>/`` by default.

The "default" profile is ``~/.hermes`` itself — backward compatible,
zero migration needed.
"""

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Directories bootstrapped inside every new profile
_PROFILE_DIRS = [
    "memories",
    "sessions",
    "skills",
    "skins",
    "logs",
    "plans",
    "workspace",
    "audio_cache",
    "image_cache",
]

# Files copied during clone (if they exist in the source)
_CLONE_CONFIG_FILES = [
    "config.yaml",
    ".env",
    "SOUL.md",
]

# Optional data dirs to clone when --clone-data is requested
_CLONE_DATA_DIRS = [
    "memories",
    "skills",
    "skins",
]


def _get_profiles_root() -> Path:
    """Return the directory where profiles are stored.

    Always ``~/.hermes/profiles/`` — anchored to the user's home,
    NOT to the current HERMES_HOME (which may itself be a profile).
    """
    return Path.home() / ".hermes" / "profiles"


def _get_default_hermes_home() -> Path:
    """Return the default (pre-profile) HERMES_HOME path."""
    return Path.home() / ".hermes"


def validate_profile_name(name: str) -> None:
    """Raise ``ValueError`` if *name* is not a valid profile identifier."""
    if name == "default":
        return  # special alias for ~/.hermes
    if not _PROFILE_ID_RE.match(name):
        raise ValueError(
            f"Invalid profile name {name!r}. Must match "
            f"[a-z0-9][a-z0-9_-]{{0,63}}"
        )


def get_profile_dir(name: str) -> Path:
    """Resolve a profile name to its HERMES_HOME directory."""
    if name == "default":
        return _get_default_hermes_home()
    return _get_profiles_root() / name


def profile_exists(name: str) -> bool:
    """Check whether a profile directory exists."""
    return get_profile_dir(name).is_dir()


@dataclass
class ProfileInfo:
    """Summary information about a profile."""
    name: str
    path: Path
    is_default: bool
    gateway_running: bool
    model: Optional[str]
    provider: Optional[str]
    has_env: bool


def _read_config_model(profile_dir: Path) -> tuple:
    """Read model/provider from a profile's config.yaml. Returns (model, provider)."""
    config_path = profile_dir / "config.yaml"
    if not config_path.exists():
        return None, None
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, str):
            return model_cfg, None
        if isinstance(model_cfg, dict):
            return model_cfg.get("model"), model_cfg.get("provider")
        return None, None
    except Exception:
        return None, None


def _check_gateway_running(profile_dir: Path) -> bool:
    """Check if a gateway is running for a given profile directory."""
    pid_file = profile_dir / "gateway.pid"
    if not pid_file.exists():
        return False
    try:
        raw = pid_file.read_text().strip()
        if not raw:
            return False
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        pid = int(data["pid"])
        os.kill(pid, 0)  # existence check
        return True
    except (json.JSONDecodeError, KeyError, ValueError, TypeError,
            ProcessLookupError, PermissionError, OSError):
        return False


def list_profiles() -> List[ProfileInfo]:
    """Return info for all profiles, including the default."""
    profiles = []

    # Default profile
    default_home = _get_default_hermes_home()
    if default_home.is_dir():
        model, provider = _read_config_model(default_home)
        profiles.append(ProfileInfo(
            name="default",
            path=default_home,
            is_default=True,
            gateway_running=_check_gateway_running(default_home),
            model=model,
            provider=provider,
            has_env=(default_home / ".env").exists(),
        ))

    # Named profiles
    profiles_root = _get_profiles_root()
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if not _PROFILE_ID_RE.match(name):
                continue
            model, provider = _read_config_model(entry)
            profiles.append(ProfileInfo(
                name=name,
                path=entry,
                is_default=False,
                gateway_running=_check_gateway_running(entry),
                model=model,
                provider=provider,
                has_env=(entry / ".env").exists(),
            ))

    return profiles


def create_profile(
    name: str,
    clone_from: Optional[str] = None,
    clone_data: bool = False,
) -> Path:
    """Create a new profile directory with bootstrapped structure.

    Parameters
    ----------
    name:
        Profile identifier (lowercase, alphanumeric, hyphens, underscores).
    clone_from:
        If set, copy config files from this existing profile.
        Use ``"default"`` to clone from the main ``~/.hermes``.
    clone_data:
        If True (and clone_from is set), also copy memories, skills, skins.

    Returns
    -------
    Path
        The newly created profile directory.
    """
    validate_profile_name(name)

    if name == "default":
        raise ValueError("Cannot create a profile named 'default' — it is the built-in profile (~/.hermes).")

    profile_dir = get_profile_dir(name)
    if profile_dir.exists():
        raise FileExistsError(f"Profile '{name}' already exists at {profile_dir}")

    # Bootstrap directory structure
    profile_dir.mkdir(parents=True, exist_ok=True)
    for subdir in _PROFILE_DIRS:
        (profile_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Clone from source profile
    if clone_from is not None:
        validate_profile_name(clone_from)
        source_dir = get_profile_dir(clone_from)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source profile '{clone_from}' does not exist at {source_dir}")

        # Copy config files
        for filename in _CLONE_CONFIG_FILES:
            src = source_dir / filename
            if src.exists():
                shutil.copy2(src, profile_dir / filename)

        # Copy data directories
        if clone_data:
            for dirname in _CLONE_DATA_DIRS:
                src = source_dir / dirname
                if src.is_dir() and any(src.iterdir()):
                    dst = profile_dir / dirname
                    # Remove the empty bootstrapped dir, copy the full tree
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)

    return profile_dir


def delete_profile(name: str) -> Path:
    """Delete a profile directory.

    Parameters
    ----------
    name:
        Profile identifier.

    Returns
    -------
    Path
        The path that was removed.

    Raises
    ------
    ValueError
        If trying to delete the default profile.
    FileNotFoundError
        If the profile does not exist.
    """
    validate_profile_name(name)

    if name == "default":
        raise ValueError("Cannot delete the default profile (~/.hermes).")

    profile_dir = get_profile_dir(name)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile '{name}' does not exist.")

    # Safety: check if gateway is running
    if _check_gateway_running(profile_dir):
        raise RuntimeError(
            f"Profile '{name}' has a running gateway. "
            f"Stop it first: hermes -p {name} gateway stop"
        )

    shutil.rmtree(profile_dir)
    return profile_dir


def resolve_profile_env(profile_name: str) -> str:
    """Resolve a profile name to a HERMES_HOME path string.

    This is called early in the CLI entry point, before any hermes modules
    are imported, to set the HERMES_HOME environment variable.
    """
    validate_profile_name(profile_name)
    profile_dir = get_profile_dir(profile_name)

    if profile_name != "default" and not profile_dir.is_dir():
        raise FileNotFoundError(
            f"Profile '{profile_name}' does not exist. "
            f"Create it with: hermes profile create {profile_name}"
        )

    return str(profile_dir)


def get_active_profile_name() -> str:
    """Infer the current profile name from HERMES_HOME.

    Returns ``"default"`` if HERMES_HOME is not set or points to ``~/.hermes``.
    Returns the profile name if HERMES_HOME points into ``~/.hermes/profiles/<name>``.
    Returns ``"custom"`` if HERMES_HOME is set to an unrecognized path.
    """
    hermes_home = Path(os.getenv("HERMES_HOME", str(_get_default_hermes_home())))
    resolved = hermes_home.resolve()

    default_resolved = _get_default_hermes_home().resolve()
    if resolved == default_resolved:
        return "default"

    profiles_root = _get_profiles_root().resolve()
    try:
        rel = resolved.relative_to(profiles_root)
        parts = rel.parts
        if len(parts) == 1 and _PROFILE_ID_RE.match(parts[0]):
            return parts[0]
    except ValueError:
        pass

    return "custom"

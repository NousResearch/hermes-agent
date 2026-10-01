"""Profile root/home path resolution."""

from __future__ import annotations

import os
from pathlib import Path

from hermes_constants import get_default_hermes_root, named_profile_is_live

from .names import (
    _PROFILE_ID_RE,
    _canon_valid,
    _invalid_profile_name_error,
    _missing_profile_error,
    normalize_profile_name,
)


def _get_default_hermes_home() -> Path:
    return get_default_hermes_root()


def _get_profiles_root() -> Path:
    return _get_default_hermes_home() / "profiles"


def _get_active_profile_path() -> Path:
    return _get_default_hermes_home() / "active_profile"


def get_profile_dir(name: str) -> Path:
    canon = normalize_profile_name(name)
    if canon == "default":
        return _get_default_hermes_home()
    if not _PROFILE_ID_RE.match(canon):
        raise _invalid_profile_name_error(canon)
    return _get_profiles_root() / canon


def profile_root_for_env_home(env_home: str, default_root: Path) -> Path:
    env_home = env_home.strip()
    if not env_home:
        return default_root
    env_path = Path(env_home)
    return env_path.parent.parent if env_path.parent.name == "profiles" else env_path


def resolve_profile_env(profile_name: str) -> str:
    canon = _canon_valid(profile_name)
    root = profile_root_for_env_home(
        os.environ.get("HERMES_HOME", ""),
        _get_default_hermes_home(),
    )
    if canon == "default":
        return str(root)
    profile_dir = root / "profiles" / canon
    if not named_profile_is_live(profile_dir):
        raise _missing_profile_error(canon)
    return str(profile_dir)

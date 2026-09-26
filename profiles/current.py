"""Sticky, home-derived, and task-scoped profile identity."""

from __future__ import annotations

import os
from pathlib import Path

from hermes_constants import get_hermes_home, get_hermes_home_override

from .names import _PROFILE_ID_RE, _canon_valid, _missing_profile_error
from .paths import _get_active_profile_path, _get_default_hermes_home, _get_profiles_root
from .registry import profile_exists


def get_active_profile(root: Path | None = None) -> str:
    path = root / "active_profile" if root is not None else _get_active_profile_path()
    try:
        return path.read_text(encoding="utf-8").strip() or "default"
    except (UnicodeDecodeError, OSError):
        return "default"


def set_active_profile(name: str) -> None:
    canon = _canon_valid(name)
    if canon != "default" and not profile_exists(canon):
        raise _missing_profile_error(canon)
    path = _get_active_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if canon == "default":
        path.unlink(missing_ok=True)
    else:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(canon + "\n", encoding="utf-8")
        tmp.replace(path)


def get_active_profile_name() -> str:
    resolved = get_hermes_home().resolve()
    if resolved == _get_default_hermes_home().resolve():
        return "default"
    profiles_root = _get_profiles_root().resolve()
    try:
        parts = resolved.relative_to(profiles_root).parts
        if len(parts) == 1 and _PROFILE_ID_RE.match(parts[0]):
            return parts[0]
    except ValueError:
        pass
    return "custom"


def current_profile_name(default: str | None = None) -> str | None:
    if get_hermes_home_override() is None:
        for env_name in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
            value = (os.environ.get(env_name) or "").strip()
            if value:
                return value
    try:
        return get_active_profile_name() or default
    except Exception:
        return default

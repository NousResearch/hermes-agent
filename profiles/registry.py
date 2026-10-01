"""Profile existence, enumeration, and tombstone-aware registry primitives."""

from __future__ import annotations

import contextlib
from pathlib import Path

from hermes_constants import (
    get_hermes_home,
    named_profile_has_identity,
    named_profile_is_deleted,
    named_profile_is_live,
)

from .names import _PROFILE_ID_RE, _canon_valid, _unknown_profile_error, normalize_profile_name
from .paths import _get_profiles_root, get_profile_dir


def _existing_profile_dir(name: str) -> tuple[str, Path]:
    canon = _canon_valid(name)
    profile_dir = get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise _unknown_profile_error(canon)
    return canon, profile_dir


def profile_exists(name: str) -> bool:
    try:
        canon = normalize_profile_name(name)
        profile_dir = get_profile_dir(canon)
    except ValueError:
        return False
    if canon == "default":
        return True
    return named_profile_is_live(profile_dir)


def profile_matches_home(name: str, home: Path | None = None) -> bool:
    try:
        target = get_profile_dir(name)
        if home is None:
            home = get_hermes_home()
        return (
            Path(target).expanduser().resolve(strict=False)
            == Path(home).expanduser().resolve(strict=False)
        )
    except Exception:
        return False


def _iter_named_profile_dirs(*, live_only: bool = True) -> list[Path]:
    profiles_root = _get_profiles_root()
    if not profiles_root.is_dir():
        return []
    return [
        entry
        for entry in sorted(profiles_root.iterdir())
        if entry.is_dir()
        and entry.name != "default"
        and _PROFILE_ID_RE.match(entry.name)
        and named_profile_has_identity(entry)
        and not (live_only and named_profile_is_deleted(entry))
    ]


def list_profile_names() -> list[str]:
    names = ["default"]
    with contextlib.suppress(OSError):
        names.extend(entry.name for entry in _iter_named_profile_dirs())
    return names

"""Profile scope for the persistence boundary.

``ProfileScope`` names the one profile a store handle is bound to: ``PersistenceBackend.open(scope)`` returns that profile's store handle, and no repository verb takes a scope. ``profile_dir`` is the profile's home folder resolved once, when the scope is built, by the ``hermes_constants.hermes_home_key`` rule (``expanduser``, strict ``resolve`` with a lenient fallback for a folder not on disk yet); every path a backend derives for the profile starts from it, so repointing a symlink on the given path later moves no scope already built. ``hermes_home_key`` is that folder's ``normcase``, and equality and hashing use that key only. ``name`` is a label: it never selects storage and is not part of equality, so two scopes on one folder with different labels are the same scope. A caller working on several profiles opens one handle per profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ProfileScope", "canonical_home_key"]


def _canonical_dir(path: Path) -> Path:
    expanded = Path(path).expanduser()
    try:
        return expanded.resolve(strict=True)
    except OSError:
        return expanded.resolve(strict=False)


def canonical_home_key(path: Path) -> str:
    """The canonical store key of a profile home: the ``hermes_constants.hermes_home_key`` rule without its memo."""
    return os.path.normcase(str(_canonical_dir(path)))


@dataclass(frozen=True, slots=True, eq=False)
class ProfileScope:
    """The profile a store handle is bound to: its resolved home folder and a label."""

    profile_dir: Path
    name: str = field(compare=False)
    hermes_home_key: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not str(self.profile_dir):
            raise ValueError("ProfileScope.profile_dir must be non-empty")
        if not self.name:
            raise ValueError("ProfileScope.name must be non-empty")
        object.__setattr__(self, "profile_dir", _canonical_dir(self.profile_dir))
        object.__setattr__(self, "hermes_home_key", os.path.normcase(str(self.profile_dir)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProfileScope):
            return NotImplemented
        return self.hermes_home_key == other.hermes_home_key

    def __hash__(self) -> int:
        return hash(self.hermes_home_key)

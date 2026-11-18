"""Profile scope for the persistence boundary.

``ProfileScope`` names the one profile a store handle is bound to: ``PersistenceBackend.open(scope)`` returns that profile's store handle, and no repository verb takes a scope. ``profile_dir`` is the profile's home folder resolved once, when the scope is built, by the ``hermes_constants.hermes_home_key`` rule (``expanduser``, strict ``resolve`` with a lenient fallback for a folder not on disk yet); every path a backend derives for the profile starts from it, so repointing a symlink on the given path later moves no scope already built. ``hermes_home_key`` is that folder's ``normcase``, and equality and hashing use that key only. ``name`` is a label: it never selects storage and is not part of equality, so two scopes on one folder with different labels are the same scope. A caller working on several profiles opens one handle per profile.

``pin`` is the hosted extension: a :class:`StorePin` naming the ``(tenant, agent)`` a hosted activity's store serves. A
local scope has none (``pin is None``) and behaves exactly as above. A pinned scope's store is chosen by its pin alone,
never by ``profile_dir`` (a hosted scope has no home of its own), so equality, hashing and ``store_key`` use the pin.
A backend that cannot serve a pin refuses it at ``open`` with :class:`persistence.ScopePinRefused`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ProfileScope", "StorePin", "canonical_home_key"]


def _canonical_dir(path: Path) -> Path:
    expanded = Path(path).expanduser()
    try:
        return expanded.resolve(strict=True)
    except OSError:
        return expanded.resolve(strict=False)


def canonical_home_key(path: Path) -> str:
    """The canonical store key of a profile home: the ``hermes_constants.hermes_home_key`` rule without its memo."""
    return os.path.normcase(str(_canonical_dir(path)))


@dataclass(frozen=True, slots=True)
class StorePin:
    """The ``(tenant, agent)`` a hosted store handle serves, taken from the bound ``hermes_scope.Scope``, never from a
    path, a config file or a caller-supplied id."""

    tenant: str
    agent: str

    def __post_init__(self) -> None:
        if not isinstance(self.tenant, str) or not self.tenant:
            raise ValueError("StorePin.tenant must be a non-empty string")
        if not isinstance(self.agent, str) or not self.agent:
            raise ValueError("StorePin.agent must be a non-empty string")

    @property
    def key(self) -> str:
        """The pin's store key: distinct from every local ``hermes_home_key`` and from every other pin."""
        return f"pin:{len(self.tenant)}:{self.tenant}/{self.agent}"


@dataclass(frozen=True, slots=True, eq=False)
class ProfileScope:
    """The profile a store handle is bound to: its resolved home folder and a label."""

    profile_dir: Path
    name: str = field(compare=False)
    hermes_home_key: str = field(init=False, repr=False, compare=False)
    pin: StorePin | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not str(self.profile_dir):
            raise ValueError("ProfileScope.profile_dir must be non-empty")
        if not self.name:
            raise ValueError("ProfileScope.name must be non-empty")
        object.__setattr__(self, "profile_dir", _canonical_dir(self.profile_dir))
        object.__setattr__(self, "hermes_home_key", os.path.normcase(str(self.profile_dir)))
        if self.pin is not None and not isinstance(self.pin, StorePin):
            raise TypeError("ProfileScope.pin must be a StorePin or None")

    @property
    def store_key(self) -> str:
        """The key of the store this scope names: ``hermes_home_key`` for a local scope, the pin's key for a pinned one."""
        return self.hermes_home_key if self.pin is None else self.pin.key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProfileScope):
            return NotImplemented
        return self.store_key == other.store_key

    def __hash__(self) -> int:
        return hash(self.store_key)

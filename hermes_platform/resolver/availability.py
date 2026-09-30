"""Whether a catalog entry can be offered on this host, from its `app` and `requires` blocks.

`availability` runs `locate` and `inspect` only. It never probes, never spawns, never connects, and holds
no cache: the callers that need a TTL (the tool registry) already have one.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Protocol

from hermes_platform.host import facts
from hermes_platform.resolver.app import AppDef, AppResolver
from hermes_platform.resolver.core import Candidate, CheckState, Resolution

AvailabilityState = Literal[
    "available",
    "installed_not_running",
    "missing_app",
    "version_too_old",
    "unsupported_os",
    "no_requirements",
]

OFFERABLE: frozenset[str] = frozenset({"available", "no_requirements"})


class _HasRequirements(Protocol):
    """The slice of a catalog entry `availability` reads; keeps this module free of `hermes_cli` imports."""

    @property
    def requires_app(self) -> bool: ...

    @property
    def min_version(self) -> str | None: ...

    def app_for(self, os_family: str) -> AppDef | None: ...


@dataclass(frozen=True)
class Availability:
    state: AvailabilityState
    version: str | None = None
    path: str | None = None
    min_version: str | None = None

    @property
    def offerable(self) -> bool:
        return self.state in OFFERABLE

    def __bool__(self) -> bool:
        raise TypeError("Availability is not a boolean; read .offerable or .state")

    def as_dict(self) -> dict:
        return {"state": self.state, "version": self.version, "path": self.path, "min_version": self.min_version}


def _version_parts(version: str) -> list[int] | None:
    if not re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,9})*", version):
        return None
    return [int(component) for component in version.split(".")]


def version_at_least(found: str, minimum: str) -> bool:
    """Compare dot-separated decimal components, failing closed on invalid input."""
    a, b = _version_parts(found), _version_parts(minimum)
    if a is None or b is None:
        return False
    width = max(len(a), len(b))
    a += [0] * (width - len(a))
    b += [0] * (width - len(b))
    return a >= b


def availability(entry: _HasRequirements, *, os_family: str | None = None) -> Availability:
    if not entry.requires_app:
        return Availability("no_requirements")
    osf = os_family or facts.os_family()
    definition = entry.app_for(osf)
    if definition is None:
        return Availability("unsupported_os", min_version=entry.min_version)
    resolver = AppResolver(definition)
    res = resolver.locate()
    looked_at = res.command[0] if res.command else (res.candidates[0].value if res.candidates else None)
    if not res.found:
        return Availability("missing_app", path=looked_at, min_version=entry.min_version)
    if definition.version_kind == "none":
        return Availability("available", path=looked_at, min_version=entry.min_version)

    def version_of(copy: Candidate) -> str | None:
        obs = resolver.inspect(Resolution(res.kind, (copy,))).version
        return obs.value if obs.state is CheckState.PRESENT else None

    if not entry.min_version:
        return Availability("available", version=version_of(res.present[0]), path=looked_at)
    # Every present copy is read, so an old copy found first does not hide a newer one found later;
    # location order only breaks ties among copies that qualify.
    copies = [(copy.value, version_of(copy)) for copy in res.present]
    for path, version in copies:
        if version is not None and version_at_least(version, entry.min_version):
            return Availability("available", version=version, path=path, min_version=entry.min_version)
    readable = [(parts, path, version) for path, version in copies
                if version is not None and (parts := _version_parts(version)) is not None]
    _, path, version = max(readable, key=lambda item: item[0]) if readable else (None, looked_at, None)
    return Availability("version_too_old", version=version, path=path, min_version=entry.min_version)

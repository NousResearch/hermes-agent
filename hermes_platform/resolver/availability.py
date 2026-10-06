"""Whether a catalog entry can be offered on this host, from its `app` and `requires` blocks.

`availability` runs `locate`, `inspect` and the host's cached GPU fact only. It never probes, never spawns,
never connects, and holds no cache: the callers that need a TTL (the tool registry) already have one.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Protocol

from hermes_platform.host import facts
from hermes_platform.resolver.app import AppDef, AppResolver
from hermes_platform.resolver.core import CheckState

AvailabilityState = Literal[
    "available",
    "installed_not_running",
    "missing_app",
    "version_too_old",
    "unsupported_os",
    "unsupported_gpu",
    "no_requirements",
]

OFFERABLE: frozenset[str] = frozenset({"available", "no_requirements"})


class _HasRequirements(Protocol):
    """The slice of a catalog entry `availability` reads; keeps this module free of `hermes_cli` imports."""

    @property
    def requires_app(self) -> bool: ...

    @property
    def min_version(self) -> str | None: ...

    @property
    def required_gpu(self) -> str | None: ...

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


def version_at_least(found: str, minimum: str) -> bool:
    """Compare dot-separated decimal components, failing closed on invalid input."""

    def parts(version: str) -> list[int] | None:
        if not re.fullmatch(r"[0-9]{1,9}(?:\.[0-9]{1,9})*", version):
            return None
        return [int(component) for component in version.split(".")]

    a, b = parts(found), parts(minimum)
    if a is None or b is None:
        return False
    width = max(len(a), len(b))
    a += [0] * (width - len(a))
    b += [0] * (width - len(b))
    return a >= b


def availability(entry: _HasRequirements, *, os_family: str | None = None) -> Availability:
    gpu = entry.required_gpu
    if not entry.requires_app and gpu is None:
        return Availability("no_requirements")
    osf = os_family or facts.os_family()
    definition = entry.app_for(osf) if entry.requires_app else None
    if entry.requires_app and definition is None:
        return Availability("unsupported_os", min_version=entry.min_version)
    # An unreadable GPU (`unknown`) passes: refusing on a failed registry or sysfs read would lock out
    # the machines that have the GPU, and the MCP connection check still applies.
    if gpu is not None and facts.gpu_class() not in (gpu, "unknown"):
        return Availability("unsupported_gpu", min_version=entry.min_version)
    if definition is None:
        return Availability("no_requirements")
    resolver = AppResolver(definition)
    res = resolver.locate()
    looked_at = res.command[0] if res.command else (res.candidates[0].value if res.candidates else None)
    if not res.found:
        return Availability("missing_app", path=looked_at, min_version=entry.min_version)
    version = None
    if entry.min_version or definition.version_kind != "none":
        obs = resolver.inspect(res).version
        version = obs.value if obs.state is CheckState.PRESENT else None
    if entry.min_version and (version is None or not version_at_least(version, entry.min_version)):
        return Availability("version_too_old", version=version, path=looked_at, min_version=entry.min_version)
    return Availability("available", version=version, path=looked_at, min_version=entry.min_version)

"""Resolver protocols and the results of the two non-passive tiers.

`locate` reads file metadata only. `inspect` may open files and call OS APIs in-process.
`probe` is fresh, never cached, and the only tier that may open a socket or spawn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from hermes_platform.resolver.core import LookupContext, Observation, Resolution


class Effort(Enum):
    """What `probe` may do. `NETWORK` never implies subprocess; `DIAGNOSTIC` alone may spawn."""

    LOCAL = "local"
    NETWORK = "network"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class Inspection:
    version: Observation[str]
    signer: Observation[str]

    @property
    def ok(self) -> bool:
        return self.version.state.value != "error" and self.signer.state.value != "error"


@dataclass(frozen=True)
class Probe:
    running: Observation[bool]
    answering: Observation[bool]
    endpoint: Observation[str]

    def __bool__(self) -> bool:  # pragma: no cover - the raise is the behavior
        raise TypeError("Probe is not a boolean; read .running/.answering")


@runtime_checkable
class Resolver(Protocol):
    name: str

    def locate(self, ctx: LookupContext | None = None) -> Resolution: ...

    def inspect(self, res: Resolution, ctx: LookupContext | None = None) -> Inspection: ...


@runtime_checkable
class Probeable(Resolver, Protocol):
    def probe(self, res: Resolution, *, effort: Effort, deadline_s: float = 3.0) -> Probe: ...

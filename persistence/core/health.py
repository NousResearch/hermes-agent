"""Store health: the corruption latch and the readiness probe, answered without opening, creating, migrating or repairing a store."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Protocol, runtime_checkable


__all__ = ["StorageHealth", "StorageProbe", "StorageState"]


class StorageState(Enum):
    """The scope's latched store state: ``CORRUPT`` once the backend observed structural damage; it never clears in-process."""

    OK = "ok"
    CORRUPT = "corrupt"


@dataclass(frozen=True, slots=True)
class StorageProbe:
    """One readiness probe: ``detail`` is ``"not initialized"`` (no store yet), ``"corrupt"`` (latched or newly found damaged), the failing error's class name, or ``None`` when healthy."""

    status: Literal["ok", "degraded"]
    detail: Optional[str]


@runtime_checkable
class StorageHealth(Protocol):
    def storage_state(self) -> StorageState:
        """The latched state of the scope's store; process-local, no store access."""
        ...

    def probe(self) -> StorageProbe:
        """Probe readiness: a store that does not exist yet is ``ok`` with ``"not initialized"``; a latched store is ``degraded`` with ``"corrupt"``; otherwise one read that never creates or migrates, whose structural failure latches and reports ``"corrupt"`` and whose other failures report their class name."""
        ...

"""Ownership leases for replaceable runtime registrations.

The coordinator models registration *generations*, not just value identity.
That distinction matters when the same provider singleton is registered again
after an older ownership generation was unloaded.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


def same_registration(left: Any, right: Any) -> bool:
    """Compare opaque registry snapshots using identity only."""
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            same_registration(a, b) for a, b in zip(left, right)
        )
    return left is right


@dataclass
class ReplacementLease:
    """One ownership generation in a replaceable registry slot."""

    coordinator: "ReplacementCoordinator"
    slot: Hashable
    current: Any
    previous: Any
    restore: Callable[[Any], bool]
    finalize: Callable[[], None] | None = None
    predecessor: "ReplacementLease | None" = None
    active: bool = field(default=True, init=False)

    def dispose(self) -> None:
        self.coordinator.dispose(self)


class ReplacementCoordinator:
    """Link and remove registration generations in arbitrary unload order."""

    def __init__(self) -> None:
        self._active: dict[Hashable, list[ReplacementLease]] = {}
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self):
        """Serialize a registry snapshot/write/acquire with lease disposal."""
        with self._lock:
            yield

    def acquire(
        self,
        slot: Hashable,
        *,
        current: Any,
        previous: Any,
        restore: Callable[[Any], bool],
        finalize: Callable[[], None] | None = None,
    ) -> ReplacementLease:
        """Attach a new live generation to the matching active predecessor."""
        with self._lock:
            leases = self._active.setdefault(slot, [])
            predecessor = next(
                (
                    lease
                    for lease in reversed(leases)
                    if lease.active and same_registration(lease.current, previous)
                ),
                None,
            )
            lease = ReplacementLease(
                coordinator=self,
                slot=slot,
                current=current,
                previous=previous,
                restore=restore,
                finalize=finalize,
                predecessor=predecessor,
            )
            leases.append(lease)
            return lease

    def dispose(self, lease: ReplacementLease) -> None:
        """Remove *lease*, restoring the nearest still-live predecessor."""
        with self._lock:
            if not lease.active:
                return
            leases = self._active.get(lease.slot, [])
            latest = next(
                (candidate for candidate in reversed(leases) if candidate.active),
                None,
            )
            lease.active = False

            # An older generation can share the exact same object identity as
            # a newer one. Registry-level CAS cannot distinguish those leases,
            # so only the latest live generation is allowed to mutate the slot.
            try:
                try:
                    if latest is lease:
                        replacement = lease.previous
                        predecessor = lease.predecessor
                        while predecessor is not None:
                            if predecessor.active:
                                replacement = predecessor.current
                                break
                            replacement = predecessor.previous
                            predecessor = predecessor.predecessor

                        lease.restore(replacement)
                finally:
                    if lease.finalize is not None:
                        lease.finalize()
            finally:
                if leases:
                    self._active[lease.slot] = [
                        item for item in leases if item.active
                    ]
                    if not self._active[lease.slot]:
                        self._active.pop(lease.slot, None)


replacement_coordinator = ReplacementCoordinator()

"""Ownership leases for replaceable runtime registrations.

The coordinator models registration *generations*, not just value identity: the same provider
singleton may be registered again after an older ownership generation was unloaded.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def same_registration(left: Any, right: Any) -> bool:
    """Compare opaque registry snapshots using identity only (element-wise for tuples)."""
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(same_registration(a, b) for a, b in zip(left, right))
    return left is right



def _hook_source_of(name: str, module: Any) -> tuple[str, str] | None:
    """Return the plugin name and resolved module source used to identify fallback hook ownership."""
    source = getattr(module, "__file__", None)
    return (name, str(Path(source).resolve())) if source else None


@dataclass
class PluginRegistration:
    """One host-owned registration plus its inverse, with idempotent disposal."""

    kind: str
    key: str
    release: Callable[[], None]
    plugin_key: str = ""
    persistent: bool = False
    _disposed: bool = field(default=False, init=False, repr=False)
    _on_dispose: Callable[["PluginRegistration"], None] | None = field(default=None, init=False, repr=False)

    @property
    def active(self) -> bool:
        """Whether this handle still owns an active registration."""
        return not self._disposed

    def dispose(self) -> None:
        """Release this registration once; repeated disposal is harmless."""
        if self._disposed:
            return
        self._disposed = True
        try:
            self.release()
        finally:
            if self._on_dispose is not None:
                self._on_dispose(self)


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

    def acquire(self, slot: Hashable, *, current: Any, previous: Any, restore: Callable[[Any], bool],
                finalize: Callable[[], None] | None = None) -> ReplacementLease:
        """Attach a new live generation to the matching active predecessor."""
        with self._lock:
            leases = self._active.setdefault(slot, [])
            predecessor = next((c for c in reversed(leases) if c.active and same_registration(c.current, previous)), None)
            lease = ReplacementLease(self, slot, current, previous, restore, finalize, predecessor)
            leases.append(lease)
            return lease

    def dispose(self, lease: ReplacementLease) -> None:
        """Remove *lease*, restoring the nearest still-live predecessor.

        ``restore`` -> ``finalize`` -> slot pruning each run even when an earlier step raises.
        """
        with self._lock:
            if not lease.active:
                return
            leases = self._active.get(lease.slot, [])
            latest = next((c for c in reversed(leases) if c.active), None)
            lease.active = False
            # An older generation can share the exact object identity of a newer one; registry-level
            # CAS cannot tell them apart, so only the latest live generation may mutate the slot.
            try:
                try:
                    if latest is lease:
                        # Restore the nearest still-live predecessor, else the last dead generation's previous.
                        replacement, predecessor = lease.previous, lease.predecessor
                        while predecessor is not None and not predecessor.active:
                            replacement, predecessor = predecessor.previous, predecessor.predecessor
                        lease.restore(predecessor.current if predecessor is not None else replacement)
                finally:
                    if lease.finalize is not None:
                        lease.finalize()
            finally:
                live = [item for item in leases if item.active]
                if live:
                    self._active[lease.slot] = live
                elif leases:
                    self._active.pop(lease.slot, None)


replacement_coordinator = ReplacementCoordinator()

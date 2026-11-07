"""What a backend presents to the assembly.

A backend is assembled over the stores it serves and opens one store handle per profile through ``PersistenceBackend.open(scope)``. The handle carries one named repository attribute per capability, the backend's declared capabilities and the awaitable view ``aio``; no verb takes a scope. The admin doors answer the physical admin capability, or ``AdminUnsupported`` on a backend without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .admin import AdminUnsupported, ProfileRepairAdmin, StoreAdmin
from .aio import AwaitableStore
from .capabilities import PersistenceCapabilities
from .context import ProfileScope

__all__ = ["PersistenceBackend", "PersistenceStore"]


@runtime_checkable
class PersistenceStore(Protocol):
    """One store handle bound to one profile: its repositories as named attributes beside these facts."""

    scope: ProfileScope
    read_only: bool
    capabilities: PersistenceCapabilities
    aio: AwaitableStore


@runtime_checkable
class PersistenceBackend(Protocol):
    """One assembled backend over the stores it serves."""

    capabilities: PersistenceCapabilities

    def open(self, scope: ProfileScope) -> PersistenceStore:
        """The store handle bound to ``scope``; opening touches no store, and every repository of the handle acts on that profile only."""
        ...

    def open_store_admin(self, target: ProfileScope | Path, *, read_only: bool = True) -> StoreAdmin | AdminUnsupported:
        """A ``StoreAdmin`` bound to ``target`` when the backend declares ``physical_admin``, else ``AdminUnsupported``."""
        ...

    def open_profile_repair(self, scope: ProfileScope, *, read_only: bool) -> ProfileRepairAdmin | AdminUnsupported:
        """A ``ProfileRepairAdmin`` bound to ``scope`` when the backend declares ``physical_admin``, else ``AdminUnsupported``."""
        ...

    def reopen(self) -> None:
        """Drop every held reference so the next use opens the stores again; a backend not declaring ``reopen_durability`` raises ``CapabilityUnsupported``."""
        ...

    def close(self) -> None:
        """Release everything the backend holds and refuse further use."""
        ...

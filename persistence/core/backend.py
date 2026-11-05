"""What a backend presents to the assembly.

A backend is assembled over the stores it serves and opens one store handle per profile through ``PersistenceBackend.open(scope)``. The handle carries one named repository attribute per capability, each declared on ``PersistenceStore`` with its protocol, the backend's declared capabilities and the awaitable view ``aio``; no verb takes a scope. The admin doors answer the physical admin capability, or ``AdminUnsupported`` on a backend without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..authority.admissions import AdmissionRepository
from ..authority.bindings import SessionBindingRepository
from ..authority.coordination import CoordinationRepository
from ..authority.runtime import SessionRuntimeRepository
from ..conversation.lineage import LineageRepository
from ..conversation.messages import MessageRepository
from ..conversation.search import SessionSearch
from ..conversation.sessions import SessionRepository
from ..conversation.usage import UsageAnalytics, UsageRepository
from ..gateway.hosted_rooms import (
    HostedRoomAttachmentRepository,
    HostedRoomDriverRepository,
    HostedRoomPeerRepository,
    HostedRoomPolicyRepository,
    HostedRoomReplicaRepository,
    HostedRoomRepository,
)
from ..gateway.routing import RoutingRepository
from ..housekeeping.maintenance import SessionMaintenance
from ..housekeeping.profile_identity import ProfileIdentityRepository
from ..housekeeping.state_meta import StateMetaRepository
from ..workers.executions import WorkerExecutionRepository
from .admin import AdminUnsupported, ProfileRepairAdmin, StoreAdmin
from .aio import AwaitableStore
from .capabilities import PersistenceCapabilities
from .context import ProfileScope
from .health import StorageHealth

__all__ = ["PersistenceBackend", "PersistenceStore"]


@runtime_checkable
class PersistenceStore(Protocol):
    """One store handle bound to one profile: its repositories as named attributes beside these facts."""

    scope: ProfileScope
    read_only: bool
    capabilities: PersistenceCapabilities
    aio: AwaitableStore
    health: StorageHealth
    sessions: SessionRepository
    messages: MessageRepository
    lineage: LineageRepository
    search: SessionSearch
    usage: UsageRepository
    usage_analytics: UsageAnalytics
    admissions: AdmissionRepository
    runtime: SessionRuntimeRepository
    bindings: SessionBindingRepository
    coordination: CoordinationRepository
    routing: RoutingRepository
    hosted_rooms: HostedRoomRepository
    hosted_room_peers: HostedRoomPeerRepository
    hosted_room_replicas: HostedRoomReplicaRepository
    hosted_room_driver: HostedRoomDriverRepository
    hosted_room_policy: HostedRoomPolicyRepository
    hosted_room_attachments: HostedRoomAttachmentRepository
    worker_executions: WorkerExecutionRepository
    maintenance: SessionMaintenance
    state_meta: StateMetaRepository
    profile_identity: ProfileIdentityRepository


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

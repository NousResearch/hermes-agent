"""In-memory reference implementation of ``StorageHealth`` (test fixture).

Registered onto the assembled backend as the ``health`` attribute. The reference has no physical store to damage, so its latch never sets; a scope whose namespace does not exist yet probes as not initialized, and probing never creates it.
"""

from __future__ import annotations

from persistence import StorageProbe, StorageState

from ..backend import InMemoryBackend, register_repository

__all__ = ["InMemoryStorageHealth"]


class InMemoryStorageHealth:
    """See module docstring; semantics follow persistence/core/health.py."""

    def __init__(self, backend: InMemoryBackend) -> None:
        self._backend = backend

    def storage_state(self) -> StorageState:
        return StorageState.OK

    def probe(self) -> StorageProbe:
        scope = self._backend.scope
        if not self._backend.scope_exists(scope):
            return StorageProbe(status="ok", detail="not initialized")
        return StorageProbe(status="ok", detail=None)


register_repository("health", InMemoryStorageHealth)

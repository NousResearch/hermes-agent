"""Assembly and profile-keyed storage for the in-memory reference backend.

``InMemoryBackend`` is the reference's file system: it holds every profile's store, keyed by the scope's ``hermes_home_key``, under one tree rooted at ``root``. ``open(scope)`` (``PersistenceBackend.open``) returns the ``InMemoryStore`` handle bound to that one profile, with one named repository attribute per registered capability, and no verb takes a scope. Two handles on one profile see the same store. Each public repository method runs under one backend-wide reentrant lock, so a verb is one atomic step (the reference's transaction) whichever thread calls it; ``executor`` is the backend's pool behind each handle's default awaitable view ``aio``.

Extension is append-only with no shared hotspot: add ONE capability module in this package that calls ``register_repository`` at import time; the package ``__init__`` auto-imports modules in its subpackages, so nobody edits this file, the package ``__init__``, or another capability's module. Capability flags earned by a repository ride on its registration; the base declaration keeps every earnable flag False so the assembled declaration never promises behavior no registered repository executes.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from persistence import AwaitableStore, CapabilityUnsupported, PermanentStoreError, PersistenceCapabilities, ProfileScope, StoreReplaced
from persistence.core.admin import PHYSICAL_ADMIN, AdminUnsupported
from persistence.core.context import canonical_home_key

__all__ = ["REFERENCE_ROOT", "InMemoryBackend", "InMemoryStore", "open_reference_backend", "register_repository"]

# The root of the reference's profile tree: ``<root>`` is the default profile, ``<root>/profiles/<name>`` a named one.
REFERENCE_ROOT = Path("/contract-homes")

# Structural facts of a single-process in-memory store. Every flag a repository
# can earn starts False and is flipped by that repository's registration.
_BASE_CAPABILITIES = PersistenceCapabilities(
    full_text_search=False,
    substring_search=False,
    transactional_claims=False,
    cross_process=False,
    remote=False,
    reopen_durability=False,
    authority_fences=False,
    durable_admissions=False,
    receipted_writes=False,
    physical_admin=False,
)

RepositoryFactory = Callable[["InMemoryStore"], Any]

_REPOSITORY_FACTORIES: dict[str, RepositoryFactory] = {}
_CAPABILITY_OVERRIDES: dict[str, bool] = {}
_PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def register_repository(
    attribute: str,
    factory: RepositoryFactory,
    capabilities: Mapping[str, bool] | None = None,
) -> None:
    """Register one capability repository for assembly onto every new ``InMemoryStore``.

    ``attribute`` is the assembled attribute name (``sessions``, ``messages``, ``search``, ...); ``factory`` receives the bound store handle and returns the repository; ``capabilities`` names the boolean ``PersistenceCapabilities`` fields this repository's real behavior earns. Called at import time from the registering module; double registration of an attribute is a conflict and fails loudly.
    """
    if attribute in _REPOSITORY_FACTORIES:
        raise ValueError(f"reference repository {attribute!r} already registered")
    _REPOSITORY_FACTORIES[attribute] = factory
    for field, value in (capabilities or {}).items():
        if not hasattr(_BASE_CAPABILITIES, field):
            raise ValueError(f"unknown capability field {field!r}")
        _CAPABILITY_OVERRIDES[field] = value


def own_profile_name(profile_dir: Path, root: Path) -> Optional[str]:
    """The profile owning a store at ``profile_dir``: ``default`` at the root, the folder name under ``<root>/profiles`` when it is a valid profile name, else None."""
    if profile_dir == root:
        return "default"
    if profile_dir.parent == root / "profiles" and _PROFILE_NAME.fullmatch(profile_dir.name):
        return profile_dir.name
    return None


class InMemoryBackend:
    """Every profile's in-memory store; ``open(scope)`` binds a handle to one of them. See module docstring."""

    # The reference fence verifies and writes inside one atomic step, so it
    # declares the same-transition fenced model-config patch.
    fenced_config_patch = True

    # Reference-atomicity crash cases inject failures between staged mutations
    # before copy-on-write publication; only this backend exposes that seam.
    injectable_failpoints = True

    def __init__(self, root: Path = REFERENCE_ROOT) -> None:
        self.root = root
        self.capabilities = dataclasses.replace(_BASE_CAPABILITIES, **_CAPABILITY_OVERRIDES)
        self._scopes: dict[str, dict[str, Any]] = {}
        self._files: dict[str, dict[str, Any]] = {}
        self._handles: dict[str, InMemoryStore] = {}
        self._closed = False
        self._replaced: set[str] = set()
        self._closed_under: set[str] = set()
        self._lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reference-store")
        # While a read snapshot is open, ``watched_writes`` counts every outermost public verb that changed stored state.
        self.watched_writes = 0
        self.snapshots_open = 0
        self._depth = 0

    def open(self, scope: ProfileScope) -> InMemoryStore:
        """The store handle bound to ``scope``; the harness opens one per profile and reuses it."""
        if self._closed:
            raise RuntimeError("backend used after teardown")
        handle = self._handles.get(scope.hermes_home_key)
        if handle is None:
            handle = self._handles[scope.hermes_home_key] = InMemoryStore(self, scope)
        return handle

    def scope_store(self, scope: ProfileScope) -> dict[str, Any]:
        """The mutable namespace of one profile's store, created on first touch; repositories keep their tables under their own keys inside it."""
        if self._closed:
            raise RuntimeError("backend used after teardown")
        return self._scopes.setdefault(scope.hermes_home_key, {})

    def replace_store(self, scope: ProfileScope) -> None:
        """The suites' seam for a store replaced under its open handles: reads keep answering from the store the handles opened, and a write that calls ``require_unreplaced`` refuses."""
        self._replaced.add(scope.hermes_home_key)

    def require_unreplaced(self, scope: ProfileScope) -> None:
        """Raise ``StoreReplaced`` once ``replace_store`` replaced ``scope``'s store."""
        if scope.hermes_home_key in self._replaced:
            raise StoreReplaced(f"the store of {scope.name!r} was replaced under its handle")

    def close_store_under(self, scope: ProfileScope) -> None:
        """The suites' seam for a store force-closed under its open handles (a profile delete's sweep): a read that refuses on a closed store calls ``require_held``."""
        self._closed_under.add(scope.hermes_home_key)

    def require_held(self, scope: ProfileScope) -> None:
        """Raise ``PermanentStoreError`` once ``close_store_under`` closed ``scope``'s store."""
        if scope.hermes_home_key in self._closed_under:
            raise PermanentStoreError(f"the store of {scope.name!r} is closed")

    def scope_exists(self, scope: ProfileScope) -> bool:
        """Whether the profile's store exists, without creating it."""
        return scope.hermes_home_key in self._scopes

    def file_store(self, path: Path) -> dict[str, Any]:
        """The mutable namespace of one store file outside any profile's ``state.db`` (the root ``shared-state.db``), keyed by its path, created on first touch."""
        if self._closed:
            raise RuntimeError("backend used after teardown")
        return self._files.setdefault(str(path), {})

    def begin_epoch(self, scope: ProfileScope, instance_id: str) -> int:
        """Begin an authority epoch on ``scope``'s store, the suites' epoch seam: the reference's authority state advances it."""
        from ._authority import authority_state  # _authority imports this module

        with self._lock:
            return int(self.run_verb(authority_state(self, scope).begin_epoch, instance_id))

    def run_verb(self, verb: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one public verb (the caller holds the lock); while a snapshot is open, an outermost verb that changed any stored state counts one watched write."""
        outer = self.snapshots_open and not self._depth
        before = self._stored_state() if outer else None
        self._depth += 1
        try:
            return verb(*args, **kwargs)
        finally:
            self._depth -= 1
            if outer and self._stored_state() != before:
                self.watched_writes += 1

    def _stored_state(self) -> str:
        """Every profile's store and every store file, as text."""
        return repr((self._scopes, self._files))

    def open_store_admin(self, target: ProfileScope | Path, *, read_only: bool = True) -> AdminUnsupported:
        """The reference has no physical admin capability (``physical_admin`` is False): the door answers ``AdminUnsupported``."""
        return AdminUnsupported(PHYSICAL_ADMIN, "in_memory")

    def open_profile_repair(self, scope: ProfileScope, *, read_only: bool) -> AdminUnsupported:
        """As :meth:`open_store_admin`: no physical admin capability."""
        return AdminUnsupported(PHYSICAL_ADMIN, "in_memory")

    def reopen(self) -> None:
        """The in-memory reference has no store to survive a reopen, declares ``reopen_durability=False``, and refuses honestly."""
        raise CapabilityUnsupported("reopen_durability")

    def close(self) -> None:
        """Deterministic teardown: drop every store and refuse further use."""
        self._scopes.clear()
        self._files.clear()
        self._handles.clear()
        self._closed = True
        self.executor.shutdown(wait=True)


class InMemoryStore:
    """One store handle bound to one profile: the reference's ``Persistence``."""

    def __init__(self, backend: InMemoryBackend, scope: ProfileScope) -> None:
        self.scope = scope
        self.read_only = False  # the reference offers no reader door
        self.backend = backend
        self.capabilities = backend.capabilities
        self.executor = backend.executor
        self.fenced_config_patch = backend.fenced_config_patch
        self.injectable_failpoints = backend.injectable_failpoints
        self.owner_profile = own_profile_name(Path(scope.hermes_home_key), Path(canonical_home_key(backend.root)))
        self._lock = backend._lock
        self.aio = AwaitableStore(self)
        for attribute, factory in _REPOSITORY_FACTORIES.items():
            repository = factory(self)
            for name, method in inspect.getmembers(repository, inspect.ismethod):
                if not name.startswith("_"):
                    setattr(repository, name, _serialized(backend, method))
            setattr(self, attribute, repository)

    def scope_store(self, scope: ProfileScope) -> dict[str, Any]:
        """The namespace of ``scope``'s store in the backend (the bound scope for every verb)."""
        return self.backend.scope_store(scope)

    def scope_exists(self, scope: ProfileScope) -> bool:
        return self.backend.scope_exists(scope)


def _serialized(backend: InMemoryBackend, method: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(method)
    def call(*args: Any, **kwargs: Any) -> Any:
        with backend._lock:
            return backend.run_verb(method, *args, **kwargs)

    return call


@contextmanager
def open_reference_backend() -> Iterator[InMemoryBackend]:
    """Fresh reference backend per entry; the conftest registers this factory by default."""
    backend = InMemoryBackend()
    try:
        yield backend
    finally:
        backend.close()

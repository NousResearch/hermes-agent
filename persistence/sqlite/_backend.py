"""The assembled SQLite backend.

One ``SQLiteBackend`` holds the stores one assembly was built for, each profile's ``hermes_home_key`` mapped to one
``state.db`` file. ``open(scope)`` returns a ``SQLiteStore`` handle bound to that one profile: every repository of the
handle acts on that file only, through the registry's single holder for it (a read-only backend opens its own
``mode=ro`` holder instead and never creates, migrates or repairs), and no verb takes a scope. Each handle's ``aio`` is
the default awaitable view, run on the backend's own executor.

Repository modules in this package call ``register_repository`` at import time with a factory that receives the bound
handle; the package ``__init__`` imports every public module, so a family registers without touching the assembly. An
attribute with no registered repository raises ``CapabilityUnsupported`` on use and its capability flags stay ``False``.

SQLite failures cross the repository boundary only as the contract's error family, through ``translated``.
"""

from __future__ import annotations

import dataclasses
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Mapping, NoReturn, Optional

from persistence.sqlite._state.errors import (
    _DISK_IO_ERROR_MARKER, DeletedWalGenerationError, StateDbCorruptError, StateDbReplacedError, _is_no_more_rows,
    classify_persistence_error, is_disk_full_error, is_fts_scoped_corruption_error, is_malformed_schema_error,
    is_sqlite_lock_error,
)
from persistence import (
    AwaitableStore, CapabilityUnsupported, CorruptStoreError, DeletedWalGeneration, DiskFull, MalformedSchema,
    PermanentStoreError, PersistenceCapabilities, PersistenceError, ProfileScope, SearchIndexCorrupt, StoreContention,
    StoreQuarantined, StoreReplaced, StoreUnavailable, TransientStoreError,
)

from . import _registry
from ._holder import StoreHolder

if TYPE_CHECKING:
    from .admin import SQLiteProfileRepairAdmin, SQLiteStoreAdmin

__all__ = [
    "CAPABILITY_ATTRIBUTES",
    "SQLiteBackend",
    "SQLiteStore",
    "register_repository",
    "translated",
    "unfolded_capabilities",
]

# The repository attributes every assembled backend presents, registered or not.
CAPABILITY_ATTRIBUTES: tuple[str, ...] = (
    "sessions",
    "messages",
    "lineage",
    "search",
    "usage",
    "usage_analytics",
    "admissions",
    "runtime",
    "bindings",
    "coordination",
    "worker_executions",
    "maintenance",
    "state_meta",
    "profile_identity",
    "routing",
    "hosted_rooms",
    "hosted_room_peers",
    "hosted_room_replicas",
    "hosted_room_driver",
    "hosted_room_policy",
    "hosted_room_attachments",
    "health",
)

# Facts of a per-file SQLite store, which offers the physical admin capability; every flag a repository earns by its
# behaviour starts False and is set by that repository's registration.
_BASE_CAPABILITIES = PersistenceCapabilities(
    full_text_search=False,
    substring_search=False,
    transactional_claims=False,
    cross_process=True,
    remote=False,
    reopen_durability=True,
    authority_fences=False,
    durable_admissions=False,
    receipted_writes=False,
    physical_admin=True,
)

RepositoryFactory = Callable[["SQLiteStore"], Any]

_REPOSITORY_FACTORIES: dict[str, RepositoryFactory] = {}
_CAPABILITY_OVERRIDES: dict[str, bool] = {}


def register_repository(
    attribute: str, factory: RepositoryFactory, capabilities: Optional[Mapping[str, bool]] = None,
) -> None:
    """Register the repository assembled as ``attribute`` on every new store handle (``factory`` receives the bound
    handle), with the capability flags its behaviour earns. Registering an attribute twice fails."""
    if attribute not in CAPABILITY_ATTRIBUTES:
        raise ValueError(f"unknown capability attribute {attribute!r}")
    if attribute in _REPOSITORY_FACTORIES:
        raise ValueError(f"sqlite repository {attribute!r} already registered")
    _REPOSITORY_FACTORIES[attribute] = factory
    for name, value in (capabilities or {}).items():
        if not hasattr(_BASE_CAPABILITIES, name):
            raise ValueError(f"unknown capability field {name!r}")
        _CAPABILITY_OVERRIDES[name] = value


def unfolded_capabilities() -> frozenset[str]:
    """The attributes with no registered repository."""
    return frozenset(attribute for attribute in CAPABILITY_ATTRIBUTES if attribute not in _REPOSITORY_FACTORIES)


class _UnfoldedCapability:
    """Stand-in for an attribute with no registered repository: every operation raises ``CapabilityUnsupported``."""

    def __init__(self, attribute: str) -> None:
        self._attribute = attribute

    def __getattr__(self, name: str) -> NoReturn:
        if name.startswith("_"):
            raise AttributeError(name)
        raise CapabilityUnsupported(self._attribute)


def _store_error(exc: sqlite3.Error) -> PersistenceError:
    """The contract error for one SQLite error the store raised."""
    message = str(exc)
    if is_disk_full_error(exc):
        return DiskFull(message)
    if is_sqlite_lock_error(exc) or _is_no_more_rows(exc):
        return StoreContention(message)
    if _DISK_IO_ERROR_MARKER in message.lower():
        return TransientStoreError(message)
    if is_fts_scoped_corruption_error(exc):
        return SearchIndexCorrupt(message)
    if is_malformed_schema_error(exc):
        return MalformedSchema(message)
    if classify_persistence_error(exc) == "corrupt":
        return CorruptStoreError(message)
    if "unable to open database file" in message.lower():
        return StoreUnavailable(message)
    return PermanentStoreError(message)


@contextmanager
def translated() -> Iterator[None]:
    """Raise the store's failures as the contract's error family, chaining the original."""
    try:
        yield
    except DeletedWalGenerationError as exc:
        raise DeletedWalGeneration(str(exc)) from exc
    except StateDbReplacedError as exc:
        raise StoreReplaced(str(exc)) from exc
    except StateDbCorruptError as exc:
        raise StoreQuarantined(str(exc)) from exc
    except sqlite3.Error as exc:
        raise _store_error(exc) from exc


# The SQLite backend's own pool for the awaitable form, shared by every backend in the process. Its size is the
# default pool size the legacy store's thread hops ran on (min(32, cpus + 4)); writes still wait for their file's
# turn inside _execute_write with the legacy patience, so no call is refused for lack of a thread.
_EXECUTOR_WORKERS = min(32, (os.cpu_count() or 1) + 4)
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()


def _store_executor() -> ThreadPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=_EXECUTOR_WORKERS, thread_name_prefix="hermes-sqlite-store")
        return _executor


class SQLiteBackend:
    """Every store one assembly was built for; ``open(scope)`` binds a handle to one of them. See the module
    docstring."""

    def __init__(self, stores: Mapping[str, Path], *, read_only: bool = False,
                 read_patience_s: Optional[float] = None) -> None:
        """``stores`` maps each served ``hermes_home_key`` to its ``state.db`` path. ``read_patience_s`` is the lock
        wait each read-only holder opens its connection with; None keeps the holder's default."""
        self._stores = {key: Path(path) for key, path in stores.items()}
        self.read_only = read_only
        self._read_patience_s = read_patience_s
        self.executor = _store_executor()
        self._lock = threading.Lock()
        self._holders: dict[str, StoreHolder] = {}
        self._lent = False
        self._closed = False
        self.capabilities = dataclasses.replace(_BASE_CAPABILITIES, **_CAPABILITY_OVERRIDES)
        self.unfolded_capabilities = unfolded_capabilities()

    @classmethod
    def lending(cls, key: str, holder: StoreHolder) -> SQLiteBackend:
        """A backend serving the one store ``holder`` holds, under ``key``, through that holder alone: the caller keeps
        its reference, so the backend never acquires the store again and its close releases nothing."""
        backend = cls({key: Path(holder.db_path)})
        backend._holders[key] = holder
        backend._lent = True
        return backend

    def open(self, scope: ProfileScope) -> SQLiteStore:
        """A handle bound to ``scope``'s store; opening touches no file. A profile this backend holds no store for is
        refused with ``ValueError``."""
        path = self._stores.get(scope.hermes_home_key)
        if path is None:
            raise ValueError(f"this SQLite backend holds no store for {scope.profile_dir}")
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLite backend used after close")
        return SQLiteStore(self, scope, path)

    def _usable(self, holder: Optional[StoreHolder]) -> bool:
        # A registry sweep (profile delete) clears the ownership flag of the holders it closes.
        return holder is not None and (self.read_only or holder._shared_registry_owned)

    def _holder(self, key: str, *, reacquire: bool = True) -> StoreHolder:
        """The open holder for one served store, acquired on first touch and, unless ``reacquire`` is False, again after
        a registry sweep. The holder that replaces one the shutdown sweep (``close_all``) closed logs the reopen
        warning on its first read or write; after a profile sweep (``close_all_under``) the store opens again
        silently, as the gateway's re-serve of that profile does."""
        path = self._stores[key]
        replaced = False
        while True:
            with self._lock:
                if self._closed:
                    raise RuntimeError("SQLite backend used after close")
                current = self._holders.get(key)
                if self._lent or self._usable(current) or (current is not None and not reacquire):
                    break
                replaced = replaced or (current is not None and current._closed_by_close_all)
                self._holders.pop(key, None)
            with translated():
                fresh = (StoreHolder(path, read_only=True, read_patience_s=self._read_patience_s) if self.read_only
                         else _registry.acquire(path))
            with self._lock:
                current = self._holders.get(key)
                publish = not self._closed and not self._usable(current)
                if publish:
                    self._holders[key] = fresh
            if publish:
                current = fresh
                break
            _registry.release_or_close(fresh)
        if replaced:
            current._reopen_notice = True
        return current

    def _release_holders(self) -> None:
        with self._lock:
            if self._lent:
                return
            holders = list(self._holders.values())
            self._holders.clear()
        for holder in holders:
            _registry.release_or_close(holder)

    def _admin_file(self, target: ProfileScope | Path) -> Path:
        if not isinstance(target, ProfileScope):
            return Path(target)
        path = self._stores.get(target.hermes_home_key)
        if path is None:
            raise ValueError(f"this SQLite backend holds no store for {target.profile_dir}")
        return path

    def open_store_admin(self, target: ProfileScope | Path, *, read_only: bool = True) -> SQLiteStoreAdmin:
        """A ``SQLiteStoreAdmin`` bound to ``target``: the ``state.db`` this backend holds for a profile, or one
        database file with no live store behind it. Opening touches no file; a profile this backend holds no store
        for is refused with ``ValueError``."""
        from .admin import SQLiteStoreAdmin

        return SQLiteStoreAdmin(self._admin_file(target), read_only=read_only)

    def open_profile_repair(self, scope: ProfileScope, *, read_only: bool) -> SQLiteProfileRepairAdmin:
        """A ``SQLiteProfileRepairAdmin`` bound to the ``state.db`` this backend holds for ``scope``; as
        :meth:`open_store_admin`."""
        from .admin import SQLiteProfileRepairAdmin

        return SQLiteProfileRepairAdmin(self._admin_file(scope), read_only=read_only)

    def reopen(self) -> None:
        """Drop every held reference; the next touch acquires the store again."""
        self._release_holders()

    def close(self) -> None:
        """Release every holder this backend acquired and refuse further use."""
        with self._lock:
            self._closed = True
        self._release_holders()


class SQLiteStore:
    """One store handle bound to one profile: its repositories, capability facts and awaitable view ``aio``."""

    def __init__(self, backend: SQLiteBackend, scope: ProfileScope, store_path: Path) -> None:
        self.scope = scope
        self.store_path = store_path
        self.read_only = backend.read_only
        self.capabilities = backend.capabilities
        self.executor = backend.executor
        self._backend = backend
        self.aio = AwaitableStore(self)
        for attribute in CAPABILITY_ATTRIBUTES:
            factory = _REPOSITORY_FACTORIES.get(attribute)
            setattr(self, attribute, factory(self) if factory is not None else _UnfoldedCapability(attribute))

    def holder(self, *, reacquire: bool = True) -> StoreHolder:
        """The open holder for this profile's store, acquired on first touch. ``reacquire=False`` keeps the holder a
        registry sweep closed, so a read through it with ``reopen=False`` refuses instead of opening the store again."""
        return self._backend._holder(self.scope.hermes_home_key, reacquire=reacquire)

    def ready(self) -> None:
        """Open the store now, so an unopenable store raises its typed error here instead of on the first verb."""
        self.holder()

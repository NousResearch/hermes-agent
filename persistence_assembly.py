"""The one door from application code to persistence.

``profile_scope`` resolves the profile a caller works on; ``open_persistence(scope)`` returns a handle on that profile's
process-shared store; ``open_persistence_reader`` returns an owned read-only handle over one target that never creates,
migrates or repairs a store. A handle's repositories act on its profile only and no verb takes a scope. Opening a
handle reads no configuration and touches no file: the provider is read and the backend assembled on the handle's first
use. Every handle releases its reference on ``close``; the last release of a profile's shared backend releases its
store. ``storage_health``, ``store_exists`` and ``store_location`` answer for a profile without opening its store; ``live_stores`` lends a handle on
every store the process holds open. ``open_store_admin``
and ``open_profile_repair`` open the physical admin capability, or answer ``AdminUnsupported`` on a backend without it.
Only this module names a backend package: callers branch on ``capabilities``, never on a provider name. A provider
other than the built-in ``sqlite`` is a ``PersistenceProvider`` plugin (``plugins/persistence_providers``), handed the
stores and each profile's ``persistence`` config section.
"""

from __future__ import annotations

import inspect
import logging
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from persistence import ProfileScope, StorageHealth
from persistence.core.admin import PHYSICAL_ADMIN, AdminUnsupported, ProfileRepairAdmin, StoreAdmin
from persistence.core.backend import PersistenceStore

__all__ = [
    "Persistence",
    "close_all_stores",
    "close_stores_under",
    "live_stores",
    "open_persistence",
    "open_persistence_reader",
    "open_profile_repair",
    "open_store_admin",
    "profile_scope",
    "storage_health",
    "store_exists",
    "store_location",
    "store_owner_name",
]

_KNOWN_PROVIDERS = ("sqlite",)

# The names a handle forwards to its bound store: the storage interface's attributes and the eager open.
_FORWARDED = frozenset(inspect.get_annotations(PersistenceStore)) | {"ready"}

_cache_lock = threading.Lock()
# hermes_home_key -> [shared backend (None until a handle's first use), open handles]
_shared: dict[str, list[Any]] = {}


class Persistence:
    """A caller's handle on one profile's store. ``scope`` and ``read_only`` (True for a reader's handle) are known at
    open; ``capabilities``, ``aio``, ``ready`` and every repository attribute are the bound store's, bound on the
    first access to any of them. ``close`` releases this handle's reference once; any of them read after it raises.
    A name ``PersistenceStore`` does not declare, other than ``ready``, raises ``AttributeError``, open or closed.
    A repository, ``aio`` view or bound method taken before ``close`` belongs to the backend, not to this handle, and
    must not outlive it: the check runs on attribute lookup only. Once the backend's last handle closes, such a view
    refuses too."""

    def __init__(self, scope: ProfileScope, *, read_only: bool, bind: Callable[[], Any], release: Callable[[], None]) -> None:
        self.scope = scope
        self.read_only = read_only
        self._bind = bind
        self._release: Optional[Callable[[], None]] = release
        self._store: Any = None
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        if name not in _FORWARDED:
            raise AttributeError(name)
        return getattr(self._bound_store(), name)

    def _bound_store(self) -> Any:
        """The bound store, bound now if this is the first use; raises after ``close``."""
        with self._lock:
            if self._release is None:
                raise RuntimeError("persistence handle used after close")
            if self._store is None:
                self._store = self._bind()
            return self._store

    def serves_same_store(self, other: Persistence) -> bool:
        """``PersistenceStore.serves_same_store`` between the stores the two handles are bound to."""
        return self._bound_store().serves_same_store(other._bound_store())

    def close(self) -> None:
        with self._lock:
            release, self._release = self._release, None
        if release is not None:
            release()

    def __enter__(self) -> Persistence:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def profile_scope(profile: Optional[str] = None, *, home: Optional[Path] = None) -> ProfileScope:
    """The scope a target names: an explicit ``home`` first, then a named profile under the default root, then the
    current home. Only a named profile's own home carries its name; any other target is its own root, labelled
    ``default``."""
    from hermes_constants import get_default_hermes_root, get_hermes_home, named_profile_home, profile_name_for_home

    if home is not None:
        target = Path(home)
    elif profile is not None and profile != "default":
        target = get_default_hermes_root() / "profiles" / profile
    elif profile == "default":
        target = get_default_hermes_root()
    else:
        target = get_hermes_home()
    canonical = target.expanduser().resolve(strict=False)
    name = profile_name_for_home(canonical)
    if name is not None and name != "default" and (named_profile_home(canonical) or canonical) == canonical:
        return ProfileScope(profile_dir=canonical, name=name)
    return ProfileScope(profile_dir=canonical, name="default")


def store_owner_name(profile_dir: Path) -> Optional[str]:
    """The profile owning the store in ``profile_dir``: ``default`` for the default root, the profile's name for a
    folder directly under ``<root>/profiles`` with a valid profile name, else None. Never raises: a failure logs at
    debug and answers None."""
    try:
        from hermes_constants import get_default_hermes_root
        root = get_default_hermes_root().resolve()
        folder = Path(profile_dir).resolve()
        if folder == root:
            return "default"
        if folder.parent == root / "profiles" and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", folder.name):
            return folder.name
    except Exception:  # health: allow BLE001 -- an underivable owner is None, never a fabricated one
        logging.getLogger("hermes_state").debug("own-profile derivation failed", exc_info=True)
    return None


#: A profile home's provider pin: one line naming the provider its store lives in.
PROVIDER_PIN = ".persistence-provider"


def pinned_provider(home: Path) -> Optional[str]:
    """The provider ``home`` is pinned to, or None without a pin."""
    try:
        name = (Path(home) / PROVIDER_PIN).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        from persistence import StoreUnavailable

        raise StoreUnavailable(f"the persistence provider pin of {home} cannot be read: {exc}") from exc
    return name or None


def pin_provider(home: Path, provider: str) -> None:
    """Pin ``home`` to ``provider``: from now on its store is never looked for anywhere else."""
    target = Path(home) / PROVIDER_PIN
    temporary = target.with_name(PROVIDER_PIN + ".tmp")
    temporary.write_text(provider + "\n", encoding="utf-8")
    temporary.replace(target)


def _provider_for_home(home: Path) -> str:
    """The configured ``persistence.provider`` for ``home``: the process config for the current home, the target's own
    ``config.yaml`` for another one. A home pinned to a provider (``PROVIDER_PIN``) uses it when the config names none
    or cannot be read, and refuses a config naming another; an unpinned home whose config cannot be read refuses rather
    than guess. An absent section or file of an unpinned home means ``sqlite``; a name that is neither built in nor an
    installed provider plugin raises ``ValueError``."""
    from persistence import StoreUnavailable

    section, readable = _read_persistence_section(home)
    pinned = pinned_provider(home)
    configured = str(section["provider"]) if readable and section.get("provider") else None
    if not readable and pinned is None:
        raise StoreUnavailable(
            f"the config of {home} cannot be read, so its session store cannot be chosen; fix config.yaml "
            f"(or pin the provider in {Path(home) / PROVIDER_PIN})")
    if configured is not None and pinned is not None and configured != pinned:
        raise StoreUnavailable(
            f"{home} is pinned to the {pinned!r} persistence provider but its config names {configured!r}; "
            f"change persistence.provider back, or move the store and update {Path(home) / PROVIDER_PIN}")
    provider = configured or pinned or "sqlite"
    if provider not in _KNOWN_PROVIDERS and _plugin_provider(provider) is None:
        raise ValueError(
            f"unknown persistence provider {provider!r} for {home} (known: {', '.join(_KNOWN_PROVIDERS)}, or a "
            f"persistence provider plugin installed under plugins/{provider})"
        )
    return provider


def _plugin_provider(name: str) -> Any:
    """The ``PersistenceProvider`` plugin registered as ``name``, or None."""
    from plugins.persistence_providers import load_persistence_provider

    return load_persistence_provider(name)


def _persistence_section(home: Path) -> dict[str, Any]:
    """``home``'s ``persistence`` config section, read as ``_provider_for_home`` reads it; empty when absent."""
    return _read_persistence_section(home)[0]


def _read_persistence_section(home: Path) -> tuple[dict[str, Any], bool]:
    """``(section, readable)``: ``home``'s ``persistence`` section (empty when absent) and whether its config was read
    as written (False for a fallback a reader served in place of an unreadable file)."""
    from hermes_constants import get_hermes_home, hermes_home_key

    section: Any = None
    readable = True
    if hermes_home_key(home) == hermes_home_key(get_hermes_home()):
        from hermes_cli.config import load_config_readonly
        from hermes_cli.config_read_errors import FailedConfigRead

        loaded = load_config_readonly()
        readable = not isinstance(loaded, FailedConfigRead)
        section = loaded.get("persistence")
    else:
        config_path = home / "config.yaml"
        if config_path.exists():
            from utils import fast_safe_load

            try:
                with open(config_path, encoding="utf-8") as fh:
                    loaded = fast_safe_load(fh)
            except Exception:  # health: allow BLE001 -- the caller decides what an unreadable config means
                loaded, readable = None, False
            if isinstance(loaded, dict):
                section = loaded.get("persistence")
    return (dict(section) if isinstance(section, dict) else {}), readable


def _assemble(provider: str, stores: Mapping[str, Path], *, read_only: bool = False,
              read_patience_s: Optional[float] = None, check_drift: bool = False) -> Any:
    """The one provider branch."""
    if provider == "sqlite":
        from persistence.sqlite import SQLiteBackend

        return SQLiteBackend(stores, read_only=read_only, read_patience_s=read_patience_s, check_drift=check_drift)
    plugin = _plugin_provider(provider)
    if plugin is not None:
        settings = {key: _persistence_section(Path(path).parent) for key, path in stores.items()}
        return plugin.assemble(stores, settings, read_only=read_only, read_patience_s=read_patience_s,
                               check_drift=check_drift)
    raise ValueError(f"unknown persistence provider {provider!r}")


def _state_db(scope: ProfileScope) -> Path:
    return Path(scope.hermes_home_key) / "state.db"


def _bind_shared(scope: ProfileScope, entry: list[Any]) -> Any:
    with _cache_lock:
        if entry[0] is None:
            entry[0] = _assemble(_provider_for_home(scope.profile_dir), {scope.hermes_home_key: _state_db(scope)})
        backend = entry[0]
    return backend.open(scope)


def _release_shared(key: str, entry: list[Any]) -> None:
    with _cache_lock:
        entry[1] -= 1
        if entry[1] > 0 or _shared.get(key) is not entry:
            return
        del _shared[key]
    if entry[0] is not None:
        entry[0].close()


def open_persistence(scope: ProfileScope, *, eager: bool = False) -> Persistence:
    """A handle on ``scope``'s process-shared store.

    Every handle on one profile shares one backend, whose store is the registry's single writer for that file. The
    first use of a handle reads the configured provider and assembles the profile's backend when no handle did yet,
    raising ``ValueError`` for an unknown provider. With ``eager`` that use and the store's open happen here, so an
    unopenable store raises its typed error at acquisition."""
    key = scope.hermes_home_key
    with _cache_lock:
        entry = _shared.setdefault(key, [None, 0])
        entry[1] += 1
    persistence = Persistence(scope, read_only=False, bind=lambda: _bind_shared(scope, entry),
                              release=lambda: _release_shared(key, entry))
    if eager:
        try:
            persistence.ready()
        except BaseException:
            persistence.close()
            raise
    return persistence


@contextmanager
def open_persistence_reader(target: str | Path, *, patience_s: Optional[float] = None,
                            check_drift: bool = False) -> Iterator[Persistence]:
    """An owned, read-only handle over one target, closed on exit or by its ``close``.

    ``target`` is a profile name, a Hermes home directory, or a database file (a ``state.db`` path or any existing
    file). The store must exist: a reader never creates, migrates or repairs it. The handle's first use reads the
    configured provider and assembles its backend; closing the handle closes that backend. ``patience_s`` is how
    long the handle's open and its reads wait for a store another connection holds before raising
    ``StoreContention`` (a verb that sets its own wait reads with that one); None keeps the backend's read wait.
    With ``check_drift`` the handle's connection, when it opens (``ready()`` or the first repository read; fetching a
    repository such as ``reader.sessions`` opens nothing), also runs ``StoreAdmin.schema_drift_reason``'s probe and
    raises ``DeclarativeDrift`` with that reason, so a browse needs no second connection for it. Such a handle also
    reports every open failure to its caller only and leaves the process's last session-store init error alone."""
    path = target if isinstance(target, Path) and (target.name == "state.db" or target.is_file()) else None
    if isinstance(target, Path):
        scope = profile_scope(home=path.parent if path is not None else target)
    else:
        scope = profile_scope(target)
    store = path.expanduser().resolve(strict=False) if path is not None else _state_db(scope)
    backends: list[Any] = []

    def bind() -> Any:
        backend = _assemble(_provider_for_home(scope.profile_dir), {scope.hermes_home_key: store}, read_only=True,
                            read_patience_s=patience_s, check_drift=check_drift)
        backends.append(backend)
        return backend.open(scope)

    def release() -> None:
        for backend in backends:
            backend.close()

    with Persistence(scope, read_only=True, bind=bind, release=release) as handle:
        yield handle


def _admin_door(provider: str, stores: Mapping[str, Path], read_only: bool, open_admin: Callable[[Any], Any]) -> Any:
    """``provider``'s admin handle through ``open_admin``, or ``AdminUnsupported`` without the capability."""
    backend = _assemble(provider, stores, read_only=read_only)
    if not backend.capabilities.physical_admin:
        backend.close()
        return AdminUnsupported(PHYSICAL_ADMIN, provider)
    return open_admin(backend)


def open_store_admin(target: ProfileScope | Path, *, read_only: bool = True,
                     provider: Optional[str] = None) -> StoreAdmin | AdminUnsupported:
    """An admin handle bound to ``target``, a profile's store or one database file with no live store behind it;
    ``AdminUnsupported`` when the backend has no physical admin capability. A profile's store uses that profile's
    configured provider. A file target reads no configuration of the folder it lies in: it uses ``provider`` when the
    caller passes one, else the provider configured for the current home; an unknown provider raises ``ValueError``."""
    if isinstance(target, ProfileScope):
        scope, path = target, _state_db(target)
        provider = _provider_for_home(scope.profile_dir)
    else:
        from hermes_constants import get_hermes_home

        path = Path(target).expanduser().resolve(strict=False)
        scope = profile_scope(home=path.parent)
        if provider is None:
            provider = _provider_for_home(get_hermes_home())
    return _admin_door(provider, {scope.hermes_home_key: path}, read_only,
                       lambda backend: backend.open_store_admin(target, read_only=read_only))


def open_profile_repair(scope: ProfileScope, *, read_only: bool) -> ProfileRepairAdmin | AdminUnsupported:
    """The crossed-profile repair handle bound to ``scope``'s store; ``AdminUnsupported`` when the configured backend
    has no physical admin capability."""
    return _admin_door(_provider_for_home(scope.profile_dir), {scope.hermes_home_key: _state_db(scope)}, read_only,
                       lambda backend: backend.open_profile_repair(scope, read_only=read_only))


def store_location(scope: ProfileScope) -> Optional[Path]:
    """The file that holds ``scope``'s store under the configured provider, or ``None`` for a provider without one."""
    provider = _provider_for_home(scope.profile_dir)
    if provider != "sqlite":
        return _plugin_provider(provider).store_location(scope, _persistence_section(scope.profile_dir))
    return _state_db(scope)


def storage_health(scope: ProfileScope) -> StorageHealth:
    """The health latch and readiness probe of ``scope``'s store, answered without opening it."""
    provider = _provider_for_home(scope.profile_dir)
    if provider != "sqlite":
        return _plugin_provider(provider).storage_health(scope, _persistence_section(scope.profile_dir))
    from persistence.sqlite.services import SQLiteStorageHealth

    return SQLiteStorageHealth(_state_db(scope))


def store_exists(scope: ProfileScope) -> bool:
    """Whether ``scope`` has a store yet, answered by its readiness probe without opening, creating or migrating it."""
    return storage_health(scope).probe().detail != "not initialized"


def session_store_exists(home: Path) -> bool:
    """Whether the profile at ``home`` has a session store yet, under its configured provider: the ``state.db`` file
    for SQLite, the provider's readiness probe otherwise. Never opens, creates or migrates the store."""
    scope = profile_scope(home=Path(home))
    if _provider_for_home(scope.profile_dir) == "sqlite":
        return _state_db(scope).exists()
    return store_exists(scope)


def export_session_store(home: Path, destination: Path) -> bool:
    """Write the session store of the profile at ``home`` to ``destination`` when its provider keeps it outside the
    profile folder (a SQLite store is a file the backup copies itself); False when there is nothing to write."""
    scope = profile_scope(home=Path(home))
    provider = _provider_for_home(scope.profile_dir)
    if provider == "sqlite":
        return False
    return bool(_plugin_provider(provider).export_store(scope, _persistence_section(scope.profile_dir), destination))


def import_session_store(home: Path, source: Path) -> None:
    """Load a store :func:`export_session_store` wrote into the profile at ``home`` through its configured provider."""
    scope = profile_scope(home=Path(home))
    provider = _provider_for_home(scope.profile_dir)
    if provider == "sqlite":
        raise ValueError(f"{home} keeps its session store in state.db; a provider export cannot be imported into it")
    _plugin_provider(provider).import_store(scope, _persistence_section(scope.profile_dir), source)


def _profile_home_of(path: Path) -> Optional[Path]:
    """The profile home a store file lies in: its nearest folder holding a provider pin or a ``config.yaml``."""
    for folder in path.parents:
        if (folder / PROVIDER_PIN).exists() or (folder / "config.yaml").exists():
            return folder
    return None


def compat_store_name(home: Path, path: Path) -> str:
    """The store name a file store at ``path`` of the profile at ``home`` has under a provider: its path in the profile
    folder without the extension, as a lowercase identifier (``projects.db`` -> ``projects``, ``cron/executions.db`` ->
    ``cron_executions``)."""
    relative = path.relative_to(home).with_suffix("")
    return re.sub(r"[^a-z0-9]+", "_", relative.as_posix().lower()).strip("_") or "store"


def _compat_target(path: Path) -> Optional[tuple[Any, Optional[ProfileScope], Mapping[str, Any], str]]:
    """The provider, profile scope, settings and store name serving the file store at ``path``, or None when it stays
    a SQLite file (its profile uses SQLite, it lies in no profile home, or ``persistence.file_stores`` leaves it out).

    With ``HERMES_SQLITE_COMPAT_TEST_DSN`` set (test suites only), every file store is served by the Postgres provider
    from a schema named after the file (scope None), so a store's own tests run against Postgres."""
    import os

    path = Path(path).expanduser().resolve(strict=False)
    test_dsn = os.environ.get("HERMES_SQLITE_COMPAT_TEST_DSN")
    if test_dsn:
        import hashlib

        from plugins.persistence_providers import _load_provider_from_dir

        plugin_dir = os.environ.get("HERMES_PG_PLUGIN_DIR")
        provider = _load_provider_from_dir(Path(plugin_dir)) if plugin_dir else _plugin_provider("postgres")
        prefix = os.environ.get("HERMES_SQLITE_COMPAT_TEST_PREFIX", "hermes_compat_test_")
        return provider, None, {"dsn": test_dsn}, prefix + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    home = _profile_home_of(path)
    if home is None:
        return None
    scope = profile_scope(home=home)
    if pinned_provider(scope.profile_dir) is None:
        # A home never pinned to a provider keeps its files; an unreadable config does not stop them opening
        # (only the session store refuses then, see _provider_for_home).
        from persistence import StoreUnavailable
        try:
            provider = _provider_for_home(scope.profile_dir)
        except StoreUnavailable:
            return None
    else:
        provider = _provider_for_home(scope.profile_dir)
    if provider == "sqlite":
        return None
    section = _persistence_section(scope.profile_dir)
    store = compat_store_name(scope.profile_dir, path)
    # Each file store moves to the provider only once its data has: ``persistence.file_stores`` names the stores
    # served from there (``true`` for all of them); any other stays the SQLite file it is.
    chosen = section.get("file_stores")
    if not (chosen is True or (isinstance(chosen, (list, tuple)) and store in chosen)):
        return None
    return _plugin_provider(provider), scope, section, store


def open_compat_store(path: Path, *, row_factory: Any = None) -> Any:
    """A ``sqlite3``-compatible connection serving the file store at ``path`` from its profile's persistence provider,
    or None when the store stays a SQLite file."""
    target = _compat_target(path)
    if target is None:
        return None
    provider, scope, settings, store = target
    conn = provider.open_compat_store(scope, settings, store, row_factory=row_factory)
    if conn is not None and hasattr(conn, "filename"):
        conn.filename = str(Path(path).expanduser().resolve(strict=False))
    return conn


def file_store_exists(path: Path) -> bool:
    """Whether the file store at ``path`` exists yet: its file for a SQLite store, any table in the provider's store
    for one :func:`open_compat_store` serves. Callers that skip work while a store is absent ask this, not the file."""
    target = _compat_target(path)
    if target is None:
        return Path(path).exists()
    provider, scope, settings, store = target
    return provider.compat_store_exists(scope, settings, store)


def move_file_store(path: Path, new_path: Path) -> bool:
    """Carry the provider-served file store at ``path`` to ``new_path`` before its folder moves there (a SQLite
    file moves with its folder); False when ``path`` is a SQLite file or holds nothing."""
    target, destination = _compat_target(path), _compat_target(new_path)
    if target is None:
        return False
    if destination is None or type(destination[0]) is not type(target[0]) or destination[1] != target[1]:
        raise ValueError(f"{new_path} is not served by the same provider and profile as {path}")
    provider, scope, settings, store = target
    return provider.move_compat_store(scope, settings, store, destination[3])


def drop_file_store(path: Path) -> bool:
    """Delete the provider-served file store at ``path`` as its folder is deleted (a SQLite file goes with its
    folder); False when ``path`` is a SQLite file or holds nothing."""
    target = _compat_target(path)
    if target is None:
        return False
    provider, scope, settings, store = target
    return provider.drop_compat_store(scope, settings, store)


def store_fingerprint(home: Path) -> Optional[tuple]:
    """A value that changes when the session store at ``home`` changes, for caches keyed on it: the ``state.db`` and
    WAL file stats for SQLite, the provider's change token otherwise; None while there is no store."""
    scope = profile_scope(home=Path(home))
    provider = _provider_for_home(scope.profile_dir)
    if provider == "sqlite":
        stats = []
        for path in (_state_db(scope), _state_db(scope).with_name("state.db-wal")):
            try:
                found = path.stat()
            except OSError:
                stats.append(None)
            else:
                stats.append((found.st_size, found.st_mtime_ns, found.st_ino))
        return None if stats[0] is None else tuple(stats)
    token = _plugin_provider(provider).change_token(scope, _persistence_section(scope.profile_dir))
    return None if token is None else (provider, token)


@contextmanager
def live_stores() -> Iterator[list[Persistence]]:
    """A handle on every store this process holds open, for in-process maintenance such as the deferred search-index
    retry. Each store stays open for the block even when its last owner releases it meanwhile; a handle acts on the
    store as it is held and never opens it again. The handles close when the block exits. Only the built-in SQLite
    stores are lent; a provider plugin runs its own maintenance."""
    from persistence.sqlite import SQLiteBackend, _registry

    with _registry.borrow_live_shared_session_dbs() as holders:
        handles = []
        for holder in holders:
            scope = profile_scope(home=Path(holder.db_path).parent)
            backend = SQLiteBackend.lending(scope.hermes_home_key, holder)
            handles.append(Persistence(scope, read_only=False, bind=lambda backend=backend, scope=scope: backend.open(scope),
                                       release=backend.close))
        try:
            yield handles
        finally:
            for handle in handles:
                handle.close()


def close_stores_under(directory: str | Path) -> int:
    """Force-close every store this process holds under ``directory`` (profile delete); returns how many closed. A
    backend that held one acquires it again on its next touch."""
    from persistence.sqlite import _registry
    from plugins.persistence_providers import loaded_providers

    return _registry.close_all_under(directory) + sum(
        plugin.close_stores_under(Path(directory)) for plugin in loaded_providers())


def close_all_stores() -> int:
    """Close every store this process holds, regardless of references (gateway shutdown); returns how many closed."""
    from persistence.sqlite import _registry
    from plugins.persistence_providers import loaded_providers

    return _registry.close_all() + sum(plugin.close_all_stores() for plugin in loaded_providers())

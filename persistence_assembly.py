"""The one door from application code to persistence.

``profile_scope`` resolves the profile a caller works on; ``open_persistence(scope)`` returns a handle on that profile's
process-shared store; ``open_persistence_reader`` returns an owned read-only handle over one target that never creates,
migrates or repairs a store. A handle's repositories act on its profile only and no verb takes a scope. Opening a
handle reads no configuration and touches no file: the provider is read and the backend assembled on the handle's first
use. Every handle releases its reference on ``close``; the last release of a profile's shared backend releases its
store. ``storage_health``, ``store_exists`` and ``store_location`` answer for a profile without opening its store; ``live_stores`` lends a handle on
every store the process holds open. ``open_store_admin``
and ``open_profile_repair`` open the physical admin capability, or answer ``AdminUnsupported`` on a backend without it.
Only this module names a backend package: callers branch on ``capabilities``, never on a provider name.

Hosted mode (``bound_hosted_scope()`` is not None) binds every store to the bound scope's ``(tenant, agent)``:
``profile_scope()`` answers a scope pinned to it (``ProfileScope.pin``), a handle opens and binds only a scope carrying
exactly that pin, and no door reads or touches a local ``state.db`` for a hosted activity. Outside hosted mode no pinned
scope opens. All pinned handles of a process share one assembled backend per provider, which serves every pin: one
connection pool for all tenants instead of one per pin. Only ``close_all_stores`` closes it.
"""

from __future__ import annotations

import inspect
import logging
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from persistence import ProfileScope, ScopePinRefused, StorageHealth, StorePin
from persistence.core.admin import PHYSICAL_ADMIN, AdminUnsupported, ProfileRepairAdmin, StoreAdmin
from persistence.core.backend import PersistenceStore

__all__ = [
    "Persistence",
    "bound_hosted_scope",
    "close_all_stores",
    "close_stores_under",
    "live_stores",
    "open_persistence",
    "open_persistence_reader",
    "open_profile_repair",
    "open_store_admin",
    "profile_scope",
    "require_hosted_persistence",
    "storage_health",
    "store_exists",
    "store_location",
    "store_owner_name",
]

_KNOWN_PROVIDERS = ("sqlite",)

# The names a handle forwards to its bound store: the storage interface's attributes and the eager open.
_FORWARDED = frozenset(inspect.get_annotations(PersistenceStore)) | {"ready"}

_cache_lock = threading.Lock()
# store_key -> [shared backend (None until a handle's first use), open handles]
_shared: dict[str, list[Any]] = {}
# provider -> the one backend serving every pinned scope of this process (hosted mode)
_pinned_backends: dict[str, Any] = {}


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
        """The bound store, bound now if this is the first use; raises after ``close``.

        The first use refuses (``ScopePinRefused``) unless the scope's pin is exactly the bound hosted scope's (none
        outside hosted mode), as ``open_persistence`` did: a handle opened under one scope may first be used under
        another. A later use refuses while a hosted scope other than the handle's own is bound; with no hosted scope
        bound (a worker thread that did not inherit the activity's context) the bound store still serves only its pin."""
        with self._lock:
            if self._release is None:
                raise RuntimeError("persistence handle used after close")
            if self._store is None:
                _require_bound_pin(self.scope)
                self._store = self._bind()
            else:
                _require_bound_pin(self.scope, unbound_ok=True)
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


def bound_hosted_scope() -> Any:
    """The bound ``hermes_scope.Scope`` when it is hosted (its tenant is not the local one), else None. Hosted mode, for
    persistence, is exactly this: a hosted scope is bound. Nothing about a local install changes while it is None."""
    from hermes_scope import current_scope

    scope = current_scope()
    return scope if scope is not None and not scope.is_local else None


def _bound_pin() -> Optional[StorePin]:
    """The pin of the bound hosted scope, or None outside hosted mode."""
    hosted = bound_hosted_scope()
    return None if hosted is None else StorePin(hosted.tenant, hosted.agent)


def _require_bound_pin(scope: ProfileScope, *, unbound_ok: bool = False) -> None:
    """Refuse ``scope`` unless its pin is exactly the bound hosted scope's: in hosted mode an unpinned (local) scope or
    another pin, outside it any pinned scope. ``unbound_ok`` lets a pinned scope pass while no scope is bound."""
    expected = _bound_pin()
    if scope.pin == expected or (unbound_ok and expected is None):
        return
    if expected is None:
        raise ScopePinRefused("a pinned store opens only while its hosted scope is bound")
    raise ScopePinRefused("a hosted activity may use only the store pinned to its bound scope")


def _refuse_hosted(what: str) -> None:
    """Refuse a door that names a local profile home, file or directory while a hosted scope is bound."""
    if bound_hosted_scope() is not None:
        raise ScopePinRefused(f"a hosted activity cannot {what}")


def profile_scope(profile: Optional[str] = None, *, home: Optional[Path] = None) -> ProfileScope:
    """The scope a target names: an explicit ``home`` first, then a named profile under the default root, then the
    current home. Only a named profile's own home carries its name; any other target is its own root, labelled
    ``default``.

    In hosted mode the scope is pinned to the bound scope's ``(tenant, agent)``, labelled ``hosted`` and built in the
    process's current home, which selects no storage; naming a profile or a home there raises ``ScopePinRefused``."""
    from hermes_constants import get_default_hermes_root, get_hermes_home, named_profile_home, profile_name_for_home

    pin = _bound_pin()
    if pin is not None:
        if profile is not None or home is not None:
            raise ScopePinRefused("a hosted activity cannot name a profile or a profile home")
        return ProfileScope(profile_dir=get_hermes_home(), name="hosted", pin=pin)
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


def _provider_for_home(home: Path) -> str:
    """The configured ``persistence.provider`` for ``home``: the process config for the current home, the target's own
    ``config.yaml`` for another one. An absent section or file means ``sqlite``."""
    from hermes_constants import get_hermes_home, hermes_home_key

    section: Any = None
    if hermes_home_key(home) == hermes_home_key(get_hermes_home()):
        from hermes_cli.config import load_config_readonly

        section = load_config_readonly().get("persistence")
    else:
        config_path = home / "config.yaml"
        if config_path.exists():
            from utils import fast_safe_load

            try:
                with open(config_path, encoding="utf-8") as fh:
                    loaded = fast_safe_load(fh)
            except Exception:  # health: allow BLE001 -- an unreadable foreign config means the default provider
                loaded = None
            if isinstance(loaded, dict):
                section = loaded.get("persistence")
    provider = "sqlite"
    if isinstance(section, dict) and section.get("provider"):
        provider = str(section["provider"])
    if provider not in _KNOWN_PROVIDERS:
        raise ValueError(
            f"unknown persistence provider {provider!r} for {home} (known: {', '.join(_KNOWN_PROVIDERS)})"
        )
    return provider


def _assemble(provider: str, stores: Mapping[str, Path], *, read_only: bool = False,
              read_patience_s: Optional[float] = None) -> Any:
    """The one provider branch."""
    if provider == "sqlite":
        from persistence.sqlite import SQLiteBackend

        return SQLiteBackend(stores, read_only=read_only, read_patience_s=read_patience_s)
    raise ValueError(f"unknown persistence provider {provider!r}")


def _state_db(scope: ProfileScope) -> Path:
    return Path(scope.hermes_home_key) / "state.db"


def _pinned_backend(scope: ProfileScope) -> Any:
    """The process's one backend for pinned scopes under the configured provider, assembled over no local store: it
    serves pins, and a backend that cannot refuses the pin at ``open``. Caller holds ``_cache_lock``."""
    provider = _provider_for_home(scope.profile_dir)
    backend = _pinned_backends.get(provider)
    if backend is None:
        backend = _pinned_backends[provider] = _assemble(provider, {})
    return backend


def _bind_shared(scope: ProfileScope, entry: list[Any]) -> Any:
    with _cache_lock:
        if entry[0] is None:
            entry[0] = (_pinned_backend(scope) if scope.pin is not None else
                        _assemble(_provider_for_home(scope.profile_dir), {scope.hermes_home_key: _state_db(scope)}))
        backend = entry[0]
    return backend.open(scope)


def _release_shared(key: str, entry: list[Any]) -> None:
    with _cache_lock:
        entry[1] -= 1
        if entry[1] > 0 or _shared.get(key) is not entry:
            return
        del _shared[key]
        shared_by_pins = any(entry[0] is backend for backend in _pinned_backends.values())
    if entry[0] is not None and not shared_by_pins:
        entry[0].close()


def open_persistence(scope: ProfileScope, *, eager: bool = False) -> Persistence:
    """A handle on ``scope``'s process-shared store.

    Every handle on one profile shares one backend, whose store is the registry's single writer for that file. The
    first use of a handle reads the configured provider and assembles the profile's backend when no handle did yet,
    raising ``ValueError`` for an unknown provider. With ``eager`` that use and the store's open happen here, so an
    unopenable store raises its typed error at acquisition. A scope whose pin is not exactly the bound hosted scope's
    (an unpinned scope in hosted mode, any pinned one outside it) raises ``ScopePinRefused`` here and again at the
    handle's first use; handles share a backend only when their ``store_key`` is the same."""
    _require_bound_pin(scope)
    key = scope.store_key
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
def open_persistence_reader(target: str | Path, *, patience_s: Optional[float] = None) -> Iterator[Persistence]:
    """An owned, read-only handle over one target, closed on exit or by its ``close``.

    ``target`` is a profile name, a Hermes home directory, or a database file (a ``state.db`` path or any existing
    file). The store must exist: a reader never creates, migrates or repairs it. The handle's first use reads the
    configured provider and assembles its backend; closing the handle closes that backend. ``patience_s`` is how
    long the handle's open and its reads wait for a store another connection holds before raising
    ``StoreContention`` (a verb that sets its own wait reads with that one); None keeps the backend's read wait.
    A hosted activity names no target: it raises ``ScopePinRefused`` and reads its own store through its handle."""
    _refuse_hosted("open a reader on a profile, home or file")
    path = target if isinstance(target, Path) and (target.name == "state.db" or target.is_file()) else None
    if isinstance(target, Path):
        scope = profile_scope(home=path.parent if path is not None else target)
    else:
        scope = profile_scope(target)
    store = path.expanduser().resolve(strict=False) if path is not None else _state_db(scope)
    backends: list[Any] = []

    def bind() -> Any:
        backend = _assemble(_provider_for_home(scope.profile_dir), {scope.hermes_home_key: store}, read_only=True,
                            read_patience_s=patience_s)
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


def _refuse_pinned_admin(target: Any) -> None:
    _refuse_hosted("open physical admin on a local store")
    if isinstance(target, ProfileScope) and target.pin is not None:
        raise ScopePinRefused("physical admin works on local stores; a pinned scope has none")


def open_store_admin(target: ProfileScope | Path, *, read_only: bool = True,
                     provider: Optional[str] = None) -> StoreAdmin | AdminUnsupported:
    """An admin handle bound to ``target``, a profile's store or one database file with no live store behind it;
    ``AdminUnsupported`` when the backend has no physical admin capability. A profile's store uses that profile's
    configured provider. A file target reads no configuration of the folder it lies in: it uses ``provider`` when the
    caller passes one, else the provider configured for the current home; an unknown provider raises ``ValueError``.
    Physical admin works on local files: a hosted activity or a pinned scope raises ``ScopePinRefused``."""
    _refuse_pinned_admin(target)
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
    has no physical admin capability. A hosted activity or a pinned scope raises ``ScopePinRefused``."""
    _refuse_pinned_admin(scope)
    return _admin_door(_provider_for_home(scope.profile_dir), {scope.hermes_home_key: _state_db(scope)}, read_only,
                       lambda backend: backend.open_profile_repair(scope, read_only=read_only))


def store_location(scope: ProfileScope) -> Optional[Path]:
    """The file that holds ``scope``'s store under the configured provider, or ``None`` for a provider without one.
    A pinned scope's store is no local file: the bound pin answers ``None``; any scope a hosted activity may not use
    (another pin, a local scope while hosted, a pin outside hosted mode) raises ``ScopePinRefused``."""
    _require_bound_pin(scope)
    if scope.pin is not None:
        return None
    return _state_db(scope)


def storage_health(scope: ProfileScope) -> StorageHealth:
    """The health latch and readiness probe of ``scope``'s store, answered without opening it. Both read a local
    ``state.db``: a pinned scope, or any scope while a hosted scope is bound, raises ``ScopePinRefused`` (a hosted
    store reports its health through its handle's typed errors)."""
    _refuse_hosted("probe a local store's health")
    if scope.pin is not None:
        raise ScopePinRefused("a pinned scope has no local store to probe")
    from persistence.sqlite.services import SQLiteStorageHealth

    return SQLiteStorageHealth(_state_db(scope))


def store_exists(scope: ProfileScope) -> bool:
    """Whether ``scope`` has a store yet, answered by its readiness probe without opening, creating or migrating it;
    refuses as ``storage_health`` does."""
    return storage_health(scope).probe().detail != "not initialized"


@contextmanager
def live_stores() -> Iterator[list[Persistence]]:
    """A handle on every store this process holds open, for in-process maintenance such as the deferred search-index
    retry. Each store stays open for the block even when its last owner releases it meanwhile; a handle acts on the
    store as it is held and never opens it again. The handles close when the block exits. While a hosted scope is
    bound it lends none: every store it could lend is a local file, never the hosted activity's."""
    from persistence.sqlite import SQLiteBackend, _registry

    if bound_hosted_scope() is not None:
        yield []
        return
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
    backend that held one acquires it again on its next touch. A hosted activity names no directory: it raises
    ``ScopePinRefused``."""
    _refuse_hosted("close the stores under a directory")
    from persistence.sqlite import _registry

    return _registry.close_all_under(directory)


def close_all_stores() -> int:
    """Close every store this process holds, regardless of references (gateway shutdown); returns how many closed.
    The backends serving pinned scopes close too; a pinned handle's next first use assembles a fresh one."""
    from persistence.sqlite import _registry

    with _cache_lock:
        pinned = list(_pinned_backends.values())
        _pinned_backends.clear()
        for entry in _shared.values():
            if any(entry[0] is backend for backend in pinned):
                entry[0] = None
    for backend in pinned:
        backend.close()
    return _registry.close_all()


def require_hosted_persistence() -> None:
    """The hosted boot check: assemble the configured provider and return only when it declares ``scope_pinning`` and
    its ``assert_isolating`` passes; raise ``IsolationUnsupported`` otherwise (always for the built-in SQLite)."""
    raise NotImplementedError("hosted persistence boot check")  # scaffold: lands with provider discovery

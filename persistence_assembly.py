"""The one door from application code to persistence.

``profile_scope`` resolves the profile a caller works on; ``open_persistence(scope)`` returns a handle on that profile's
process-shared store; ``open_persistence_reader`` returns an owned read-only handle over one target that never creates,
migrates or repairs a store. A handle's repositories act on its profile only and no verb takes a scope. Opening a
handle reads no configuration and touches no file: the provider is read and the backend assembled on the handle's first
use. Every handle releases its reference on ``close``; the last release of a profile's shared backend releases its
store. ``storage_health`` and ``store_location`` answer for a profile without opening its store; ``live_stores`` lends a handle on
every store the process holds open. ``open_store_admin``
and ``open_profile_repair`` open the physical admin capability, or answer ``AdminUnsupported`` on a backend without it.
Only this module names a backend package: callers branch on ``capabilities``, never on a provider name.
"""

from __future__ import annotations

import logging
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

from persistence import ProfileScope, StorageHealth
from persistence.core.admin import PHYSICAL_ADMIN, AdminUnsupported, ProfileRepairAdmin, StoreAdmin

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
    "store_location",
    "store_owner_name",
]

_KNOWN_PROVIDERS = ("sqlite",)

_cache_lock = threading.Lock()
# hermes_home_key -> [shared backend (None until a handle's first use), open handles]
_shared: dict[str, list[Any]] = {}


class Persistence:
    """A caller's handle on one profile's store. ``scope`` and ``read_only`` (True for a reader's handle) are known at
    open; ``capabilities``, ``aio``, ``ready`` and every repository attribute are the bound store's, bound on the
    first access to any of them. ``close`` releases this handle's reference once; any of them read after it raises.
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
        if name.startswith("_"):
            raise AttributeError(name)
        with self._lock:
            if self._release is None:
                raise RuntimeError("persistence handle used after close")
            if self._store is None:
                self._store = self._bind()
            store = self._store
        return getattr(store, name)

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
def open_persistence_reader(target: str | Path, *, patience_s: Optional[float] = None) -> Iterator[Persistence]:
    """An owned, read-only handle over one target, closed on exit or by its ``close``.

    ``target`` is a profile name, a Hermes home directory, or a database file (a ``state.db`` path or any existing
    file). The store must exist: a reader never creates, migrates or repairs it. The handle's first use reads the
    configured provider and assembles its backend; closing the handle closes that backend. ``patience_s`` is how
    long the handle's open and its reads wait for a store another connection holds before raising
    ``StoreContention`` (a verb that sets its own wait reads with that one); None keeps the backend's read wait."""
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
    return _state_db(scope)


def storage_health(scope: ProfileScope) -> StorageHealth:
    """The health latch and readiness probe of ``scope``'s store, answered without opening it."""
    from persistence.sqlite.services import SQLiteStorageHealth

    return SQLiteStorageHealth(_state_db(scope))


@contextmanager
def live_stores() -> Iterator[list[Persistence]]:
    """A handle on every store this process holds open, for in-process maintenance such as the deferred search-index
    retry. Each store stays open for the block even when its last owner releases it meanwhile; a handle acts on the
    store as it is held and never opens it again. The handles close when the block exits."""
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

    return _registry.close_all_under(directory)


def close_all_stores() -> int:
    """Close every store this process holds, regardless of references (gateway shutdown); returns how many closed."""
    from persistence.sqlite import _registry

    return _registry.close_all()

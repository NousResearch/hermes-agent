"""Where the process-wide credential paths meet the secret store port.

The selected secret store (``HERMES_SECRET_STORE``, read from the environment on every call, never
from config) serves every read and write of a profile's ``.env``-style secrets and ``auth.json``
records that goes through:

- the profile ``.env`` read (:func:`read_profile_env`: the profile secret scope, ``load_env()``,
  every ``load_env_file(<home>/.env)`` reader), and the launch load of that file into
  ``os.environ`` (the ordinary dotenv loader, fenced by :func:`load_profile_dotenv`);
- ``save_env_value`` / ``remove_env_value`` (:func:`route_env_write`);
- the ``auth.json`` store's load and save (:func:`load_auth_store`, :func:`save_auth_store`) and the
  readers that parse the file themselves (:func:`read_auth_json`).

With the local store (the default) every function here is a pass-through to the code that ran
before it existed — the same file, read the same way — so a local install behaves exactly as it
did. With any other store (``files_backed`` false) the profile's local files keep only the store's
bootstrap entries (the names and records it needs to select, address and authenticate itself);
every other name and record is the store's, read from its view and written through it, and no
local entry outside the bootstrap set is ever read in its place.

A named store that is missing or unavailable stops the process (:class:`SecretStoreUnavailable`, a
``SystemExit`` like the config backend's): many readers fail open with ``except Exception``, and
reading the local files instead of the selected store is the silent failure this must not become.

``scripts/check_secret_file_access.py`` rejects new direct ``.env`` / ``auth.json`` access
outside this module and the local backend, with an allowlist for what legitimately stays local.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_secret_port import SECRET_STORE_ENV, SecretStoreProvider

ENV_FILENAME = ".env"
AUTH_FILENAME = "auth.json"
#: The ``auth.json`` record that carries the store's top-level metadata (``active_provider`` and
#: the like): everything but the per-provider states, the credential pool and the file's own
#: version stamp.
STORE_META_RECORD = "hermes:store"
#: Record-id prefix of one provider's credential-pool slice.
POOL_RECORD_PREFIX = "pool:"
_LOCAL_STORE_KEYS = ("version", "updated_at")
#: Names that only the launch decides (which home, which administrator scope), in any letter case:
#: never a profile secret, so never a store's.
LAUNCH_ONLY_ENV_NAMES = frozenset({"HERMES_HOME", "HERMES_MANAGED_DIR"})


class SecretStoreUnavailable(SystemExit):
    """The selected secret store cannot serve secrets. A ``SystemExit`` on purpose: serving the
    local files instead of the selected store is exactly the silent failure this must not be."""


_PROVIDERS: dict[str, SecretStoreProvider] = {}
_PROVIDERS_LOCK = threading.Lock()


def selected_store_name() -> str:
    """The selected store's name (``local`` when unset), from the environment on every call."""
    return os.environ.get(SECRET_STORE_ENV, "").strip().lower() or "local"


def _provider_named(name: str) -> SecretStoreProvider:
    provider = _PROVIDERS.get(name)
    if provider is not None:
        return provider
    from hermes_secret_port import resolve_secret_store
    with _PROVIDERS_LOCK:
        provider = _PROVIDERS.get(name)
        if provider is None:
            try:
                provider = resolve_secret_store(name)
            except LookupError as exc:
                raise SecretStoreUnavailable(
                    f"{SECRET_STORE_ENV}={name!r}: {exc}. Hermes does not start without its selected "
                    "secret store (it never falls back to the local .env / auth.json).") from None
            _PROVIDERS[name] = provider
    return provider


def get_secret_store_provider() -> SecretStoreProvider:
    """The provider ``HERMES_SECRET_STORE`` selects; :class:`SecretStoreUnavailable` when missing."""
    return _provider_named(selected_store_name())


def routed_provider() -> Optional[SecretStoreProvider]:
    """The selected provider when it keeps secrets somewhere other than today's files, else None
    (the local store: every caller runs today's file code)."""
    name = selected_store_name()
    if name == "local":
        return None
    provider = _provider_named(name)
    return None if provider.files_backed else provider


def reset_for_tests() -> None:
    """Forget resolved providers and published views (tests only)."""
    with _PROVIDERS_LOCK:
        _PROVIDERS.clear()
        _BOOTSTRAP_GENERATIONS.clear()
    with _PUBLISH_LOCK:
        _PUBLICATIONS.clear()
        _PUBLISHED_NAMES.clear()


# --- bootstrap and deployment ------------------------------------------------------------------

def bootstrap_env_names_for(kind: Optional[str]) -> frozenset[str]:
    """The env names the store *kind* (as the ``.env`` layers select it) needs before the first
    secret read, its selector included; just the selector for a kind this build lacks (selection
    refuses it afterwards)."""
    name = (kind or "").strip().lower() or "local"
    if name == "local":
        return frozenset({SECRET_STORE_ENV})
    try:
        return frozenset({SECRET_STORE_ENV}) | _provider_named(name).bootstrap_env_names()
    except SecretStoreUnavailable:
        return frozenset({SECRET_STORE_ENV})


def deployment_env_names() -> frozenset[str]:
    """The selected store's deployment env names (which store, where): never launch residue."""
    provider = routed_provider()
    if provider is None:
        return frozenset()
    return frozenset({SECRET_STORE_ENV}) | provider.deployment_env_names()


def supports_file_tooling() -> bool:
    """Whether tools that copy, back up or restore the ``.env`` / ``auth.json`` files may run."""
    provider = routed_provider()
    return provider is None or provider.supports_file_tooling()


def require_file_tooling(what: str) -> None:
    """Refuse *what* (a tool that copies the secret files) when the store keeps secrets elsewhere."""
    if not supports_file_tooling():
        raise SecretStoreUnavailable(
            f"{what} copies the local .env and auth.json, and the {selected_store_name()!r} secret "
            "store keeps this profile's secrets elsewhere (only its bootstrap entries are on disk). "
            "It is not available with this secret store.")


def _profile_home_of(path: Path, filename: str) -> Optional[Path]:
    """The profile home *path* is the ``filename`` of, or None (another file name, or the
    administrator-managed ``.env``, which is policy on disk and never a store's). Callers hand
    these functions profile files only; a store refuses a home that is not a profile's."""
    path = Path(path)
    if path.name != filename:
        return None
    home = path.parent
    if filename == ENV_FILENAME:
        try:
            from hermes_cli.managed_scope import get_managed_dir
            managed = get_managed_dir()
        except Exception:  # health: allow BLE001 -- no managed scope readable: nothing to exclude
            managed = None
        if managed is not None and Path(managed).resolve(strict=False) == home.resolve(strict=False):
            return None
    return home


# --- .env ----------------------------------------------------------------------------------------

def stays_local(provider: SecretStoreProvider, name: str) -> bool:
    """Whether env *name* stays in the profile's local ``.env`` under *provider* and is never sent
    to it: the store's bootstrap names, and every name Hermes treats as process-global (deployment
    settings, not profile secrets: ``agent.secret_scope._is_global_env``) or as a per-process
    credential a parent injects."""
    if name in provider.bootstrap_env_names() or name.upper() in LAUNCH_ONLY_ENV_NAMES:
        return True
    from agent.secret_scope import _is_global_env
    from hermes_cli.env_loader import _SPAWN_CREDENTIAL_KEYS
    return _is_global_env(name) or name in _SPAWN_CREDENTIAL_KEYS


def _routed_env_home(env_path: Path) -> tuple[Optional[SecretStoreProvider], Optional[Path]]:
    provider = routed_provider()
    if provider is None:
        return None, None
    home = _profile_home_of(env_path, ENV_FILENAME)
    return (provider, home) if home is not None else (None, None)

_BOOTSTRAP_GENERATIONS = {}
_PROFILE_SNAPSHOT = ContextVar("raw_profile_snapshot", default=None)


@contextmanager
def profile_snapshot(home, snapshot):
    token = _PROFILE_SNAPSHOT.set((Path(home), snapshot))
    try:
        yield
    finally:
        _PROFILE_SNAPSHOT.reset(token)


def bootstrap_generation(home):
    return _BOOTSTRAP_GENERATIONS.get(Path(home), 0)


def live_bootstrap_env(home, provider):
    """Invalidate cached ordinary builders after a local bootstrap credential write."""
    _BOOTSTRAP_GENERATIONS[Path(home)] = bootstrap_generation(home) + 1


def read_profile_env(env_path: Path) -> dict[str, str]:
    """A profile ``.env``'s assignments: ``load_env_file(env_path)`` with the local store; with any
    other store, the local file's bootstrap names plus the store's view of the profile."""
    from agent import secret_scope
    provider, home = _routed_env_home(env_path)
    if provider is None or home is None:
        return secret_scope.load_env_file(env_path)
    merged = {k: v for k, v in secret_scope.load_env_file(env_path).items() if stays_local(provider, k)}
    pinned = _PROFILE_SNAPSHOT.get()
    snapshot = pinned[1] if pinned is not None and pinned[0] == Path(home) else provider.current_snapshot(home)
    merged.update((k, v) for k, v in snapshot.env.items() if not stays_local(provider, k))
    return merged


def read_env_text(env_path: Path, *, errors: Optional[str] = None) -> str:
    """A profile ``.env``'s text, for the readers that tokenize it themselves:
    ``env_path.read_text(encoding="utf-8-sig", errors=errors)`` with the local store (raising what
    it raises); with any other store, the merged view as dotenv lines."""
    provider, home = _routed_env_home(env_path)
    if provider is None or home is None:
        return Path(env_path).read_text(encoding="utf-8-sig", errors=errors)
    from hermes_cli.config import _quote_env_value
    return "".join(f"{k}={_quote_env_value(v)}\n" for k, v in read_profile_env(env_path).items())


def routed_profile_env(env_path: Path) -> Optional[dict[str, str]]:
    """None with the local store (or a file that is not a profile ``.env``); with any other store,
    :func:`read_profile_env`: for readers that parse the file themselves only when it is the store's."""
    provider, home = _routed_env_home(env_path)
    return None if provider is None or home is None else read_profile_env(env_path)


def env_file_exists(env_path: Path) -> bool:
    """``env_path.exists()`` with the local store; with any other store, whether the profile has
    any secret at all (local bootstrap or the store's view)."""
    provider, home = _routed_env_home(env_path)
    if provider is None or home is None:
        return Path(env_path).exists()
    return True  # an authoritative empty profile still occupies the dotenv layer


_PUBLICATIONS: dict[str, tuple[int, dict[str, str]]] = {}
_PUBLISHED_NAMES: dict[str, set[str]] = {}


def published_env_names(home):
    """Store-owned launch names, including revoked ones: never a frozen-env fallback."""
    from hermes_constants import hermes_home_key
    with _PUBLISH_LOCK:
        return set(_PUBLISHED_NAMES.get(hermes_home_key(home), ()))

_PUBLISH_LOCK = threading.RLock()


def boot_secret_store(home: Path) -> None:
    """Boot without publishing: the ordinary dotenv loader owns publication."""
    provider = routed_provider()
    pinned = _PROFILE_SNAPSHOT.get()
    if provider is not None and (pinned is None or pinned[0] != Path(home)):
        provider.boot(Path(home))


def load_profile_dotenv(loader, **kwargs):
    """Serialize the ordinary loader against snapshot publication, not a second loader."""
    from hermes_constants import get_process_hermes_home, get_hermes_home, get_hermes_home_override
    from agent.secret_scope import is_multiplex_active
    from agent.safe_worker_policy import worker_config_snapshot
    home = Path(kwargs.get("hermes_home") or get_process_hermes_home())
    if worker_config_snapshot() is not None:
        return loader(**kwargs)
    launch = get_process_hermes_home().resolve()
    if (is_multiplex_active() and get_hermes_home_override() is not None
            and (home.resolve() != launch or get_hermes_home().resolve() != launch)):
        return loader(**kwargs)
    # Selection can be supplied by the local bootstrap file.
    from hermes_cli.config_backend import bootstrap_deployment
    bootstrap_deployment(home, kwargs.get("project_env"))
    provider = routed_provider()
    if provider is None:
        return loader(**kwargs)
    if _PROFILE_SNAPSHOT.get() is not None:
        return loader(**kwargs)
    provider.boot(home)
    return publish(provider.current_snapshot(home), home=home, loader=loader, **kwargs)


def publish(snapshot, *, home, consumer=None, loader=None, **kwargs):
    """Derive every consumer from this captured immutable snapshot, never a later read: a live
    scope's *consumer*, or the ordinary dotenv *loader* (called with *kwargs*) for os.environ."""
    with profile_snapshot(home, snapshot):
        if consumer is not None:
            return consumer(snapshot)
        from hermes_cli.env_loader import resolve_home_dotenv
        from hermes_constants import hermes_home_key
        provider = routed_provider()
        key = hermes_home_key(home)
        with _PUBLISH_LOCK:
            cell = _PUBLICATIONS.get(key)
            if cell is not None and snapshot.version < cell[0]:
                return None
            old = cell[1] if cell is not None else {}
            new = resolve_home_dotenv(Path(home))
            for name, value in old.items():
                if name not in new and os.environ.get(name) == value:
                    os.environ.pop(name, None)
            result = loader(**kwargs)
            values = {k: v for k, v in new.items() if not stays_local(provider, k)}
            _PUBLICATIONS[key] = (snapshot.version, values)
            _PUBLISHED_NAMES.setdefault(key, set()).update(values)
            return result


def route_env_write(env_path: Path, key: str, value: Optional[str]) -> Optional[bool]:
    """``save_env_value`` (*value* a string) / ``remove_env_value`` (*value* None) for *key* in the
    profile ``.env`` at *env_path*: None when the write is the file's (the local store, or a
    bootstrap name); otherwise the store's write — True when it set the name, or, for a removal,
    whether the view had it."""
    provider, home = _routed_env_home(env_path)
    if provider is None or home is None or stays_local(provider, key):
        return None
    if value is not None:
        provider.write_view(home, env_set={key: value})
        return True
    had = key in provider.env_view(home)
    if had:
        provider.write_view(home, env_unset=(key,))
    return had


# --- auth.json -------------------------------------------------------------------------------------

def auth_records(store: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """An ``auth.json`` store as records: each provider state under its provider id, each
    credential-pool slice under ``pool:<provider>`` (``{"entries": [...]}``), and the remaining
    top-level metadata under :data:`STORE_META_RECORD`. The file's own version stamp is not a
    record."""
    records: dict[str, dict[str, Any]] = {}
    providers = store.get("providers")
    for pid, state in (providers.items() if isinstance(providers, dict) else ()):
        if isinstance(state, dict):
            records[str(pid)] = state
    pool = store.get("credential_pool")
    for pid, entries in (pool.items() if isinstance(pool, dict) else ()):
        records[POOL_RECORD_PREFIX + str(pid)] = {"entries": entries}
    meta = {k: v for k, v in store.items() if k not in ("providers", "credential_pool", *_LOCAL_STORE_KEYS)}
    if meta:
        records[STORE_META_RECORD] = meta
    return records


def auth_store_from_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The inverse of :func:`auth_records` (no version stamp)."""
    store: dict[str, Any] = {"providers": {}}
    for rid in sorted(records):
        record = records[rid]
        if rid == STORE_META_RECORD:
            store.update((k, v) for k, v in record.items()
                         if k not in ("providers", "credential_pool", *_LOCAL_STORE_KEYS))
        elif rid.startswith(POOL_RECORD_PREFIX):
            store.setdefault("credential_pool", {})[rid[len(POOL_RECORD_PREFIX):]] = record.get("entries", [])
        else:
            store["providers"][rid] = dict(record)
    return store


_AUTH_BASES = threading.local()   # .bases: home key -> the records this thread's last load returned


def _auth_bases() -> dict[str, dict[str, Any]]:
    bases = getattr(_AUTH_BASES, "bases", None)
    if bases is None:
        bases = _AUTH_BASES.bases = {}
    return bases


def _routed_auth_home(auth_file: Path) -> tuple[Optional[SecretStoreProvider], Optional[Path]]:
    provider = routed_provider()
    if provider is None:
        return None, None
    home = _profile_home_of(auth_file, AUTH_FILENAME)
    return (provider, home) if home is not None else (None, None)


def routed_auth_loader(load):
    """Decorates ``hermes_cli.auth._load_auth_store``: resolves the default path (the active
    profile's ``auth.json``) exactly as the loader did, runs the loader on it, and hands its result
    to :func:`load_auth_store`."""
    import functools

    @functools.wraps(load)
    def loader(auth_file: Optional[Path] = None) -> dict[str, Any]:
        if not auth_file:
            import sys
            auth_file = sys.modules["hermes_cli.auth"]._auth_file_path()
        return load_auth_store(auth_file, load(auth_file))
    return loader


def load_auth_store(auth_file: Path, local_store: dict[str, Any]) -> dict[str, Any]:
    """The auth store ``_load_auth_store`` returns: *local_store* (the file as today's loader read
    it) unchanged with the local store; with any other store, the file's bootstrap records plus the
    store's view, every other local record ignored."""
    provider, home = _routed_auth_home(auth_file)
    if provider is None:
        return local_store
    import copy
    from hermes_constants import hermes_home_key
    keep = provider.bootstrap_auth_records()
    records = {rid: r for rid, r in provider.auth_view(home).items() if rid not in keep}
    _auth_bases()[hermes_home_key(home)] = copy.deepcopy(records)
    records.update((rid, r) for rid, r in auth_records(local_store).items() if rid in keep)
    store = auth_store_from_records(copy.deepcopy(records))
    for k in _LOCAL_STORE_KEYS:
        if k in local_store:
            store[k] = local_store[k]
    return store


def save_auth_store(auth_file: Path, auth_store: dict[str, Any]) -> Optional[dict[str, Any]]:
    """For ``_save_auth_store``: None with the local store (save the whole store to the file as
    today). With any other store, every changed non-bootstrap record is written through the store
    (one write), and the store that remains for the local file — its bootstrap records only — is
    returned for the caller to save. Changes are judged against what this thread's last load of
    the store returned, so a record another writer changed since, which this save never touched,
    is not written back over theirs."""
    provider, home = _routed_auth_home(auth_file)
    if provider is None:
        return None
    from hermes_constants import hermes_home_key
    keep = provider.bootstrap_auth_records()
    records = auth_records(auth_store)
    remote = {rid: r for rid, r in records.items() if rid not in keep}
    base = _auth_bases().get(hermes_home_key(home))
    if base is None:
        base = provider.auth_view(home)
    changes: dict[str, Any] = {rid: r for rid, r in remote.items() if base.get(rid) != r}
    changes.update((rid, None) for rid in base if rid not in remote)
    if changes:
        provider.write_view(home, auth=changes)
        import copy
        _auth_bases()[hermes_home_key(home)] = copy.deepcopy(remote)
    local = auth_store_from_records({rid: r for rid, r in records.items() if rid in keep})
    for k in _LOCAL_STORE_KEYS:
        if k in auth_store:
            local[k] = auth_store[k]
    return local


def read_auth_json(auth_file: Path) -> Any:
    """A profile ``auth.json`` parsed, for the readers that parse it themselves:
    ``json.loads(auth_file.read_text(encoding="utf-8-sig"))`` with the local store (raising what
    it raises); with any other store, the merged store ``_load_auth_store`` returns."""
    provider, _home = _routed_auth_home(auth_file)
    if provider is None:
        import json
        return json.loads(Path(auth_file).read_text(encoding="utf-8-sig"))
    from hermes_cli.auth import _load_auth_store
    return _load_auth_store(Path(auth_file))


def auth_view_version(auth_file: Path) -> Optional[str]:
    """For caches of a parsed ``auth.json``: None with the local store (the file's own metadata
    is the cache key, as today); with any other store, the store's version of the profile's view
    (``view_version``), which changes whenever a record does where the file never moves."""
    provider, home = _routed_auth_home(auth_file)
    return None if provider is None else provider.view_version(home)


def auth_json_exists(auth_file: Path) -> bool:
    """``auth_file.exists()`` with the local store; with any other store, True (the store always
    has a view: an empty one reads as a store with no records)."""
    provider, _home = _routed_auth_home(auth_file)
    return Path(auth_file).exists() if provider is None else True

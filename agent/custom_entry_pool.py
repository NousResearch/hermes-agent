"""Credential ownership for ``providers:`` entries whose pool slug is also a built-in's pool id.

``providers.deepseek`` pools under the slug ``deepseek``, which ``load_pool`` also seeds from the
built-in DeepSeek provider's env vars and auth files. Those rows are the built-in's accounts. Every
acquisition of a pool for the custom entry's runtime (main, frozen and auxiliary resolution, a
URL-matched bare ``custom`` route, restoring the primary after a fallback, a delegated child's pool)
takes the entry-scoped view from here, so the built-in's account never stands in for the entry's
declared credential, and no pooled key leaves the origin it was entered for (#130410). The
built-in provider itself still loads, and persists, the whole pool.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

_DECLARED_CREDENTIAL_KEYS = ("api_key", "key_env", "api_key_env", "key_cmd")
# The ``_primary_runtime`` field holding the owning entry's provenance (``runtime_entry_owner``).
SNAPSHOT_OWNER_KEY = "custom_entry"


def entry_declares_credential(entry: Any) -> bool:
    """Whether a custom entry names its own credential (``api_key`` / ``key_env`` / ``key_cmd``). A
    snapshot's provenance carries the verdict itself, since it never copies an inline secret."""
    if isinstance(entry, dict) and "declares_credential" in entry:
        return bool(entry["declares_credential"])
    return isinstance(entry, dict) and any(str(entry.get(k) or "").strip() for k in _DECLARED_CREDENTIAL_KEYS)


def row_owned_by_entry(row: Any, entry: Any) -> bool:
    """Ownership by provenance: ``hermes auth add`` (manual) rows, and rows seeded from the entry's
    own ``key_env``. Rows the built-in seeded from its registry env vars or auth files are foreign."""
    from agent.credential_pool import _is_manual_source
    source = str(getattr(row, "source", "") or "").strip()
    own_env = {f"env:{name}" for name in (str(entry.get(k) or "").strip() for k in ("key_env", "api_key_env")) if name}
    return _is_manual_source(source) or source in own_env


def entry_scoped_pool(pool: Any, pool_key: str, entry: Optional[dict[str, Any]], base_url: Optional[str] = None) -> Any:
    """The rows of ``pool`` the custom ``entry`` may send to ``base_url`` (default: the entry's own
    address). Every acquisition of a custom entry's pool goes through here.

    Ownership: a pool the entry shares with a same-named built-in yields only the entry's rows when
    the entry declares a credential. Origin binding: a row is kept only when it is one the entry
    re-seeds from its current config (``env:<its key_env>``, ``config:<its name>``,
    ``model_config``) or it recorded the same origin (scheme/host/port) as ``base_url``, so no key
    entered for one host, nor a built-in's seeded account, follows the entry to another host. A
    manual row that recorded no address is bound to the entry's current one first (see
    ``bind_unrecorded_rows``); any other row without one is dropped.

    The pool itself when every row is kept, else a view sharing its rows and token baselines
    (persisting it merges into the stored pool under its lock), None when nothing is left.
    """
    from agent.credential_pool import CUSTOM_POOL_PREFIX, CredentialPool
    if pool is None or not isinstance(entry, dict):
        return pool
    target = str(base_url or entry.get("base_url") or "").strip()
    bind_unrecorded_rows(pool, str(entry.get("base_url") or "").strip())
    whole = str(pool_key or "").startswith(CUSTOM_POOL_PREFIX) or not entry_declares_credential(entry)
    own = _own_seeded_sources(entry)
    rows = pool.entries()
    kept = [row for row in rows if (whole or row_owned_by_entry(row, entry)) and _row_serves(row, own, target)]
    if len(kept) == len(rows):
        return pool
    if not kept:
        return None
    view = CredentialPool(pool.provider, kept)
    view._persisted_token_pairs = pool._persisted_token_pairs
    return view


def _own_seeded_sources(entry: dict[str, Any]) -> set[str]:
    """Sources ``load_pool`` re-seeds from the entry's current config: they carry whatever key the
    entry holds now (an origin move strips it), never one recorded for an earlier address."""
    envs = {f"env:{name}" for name in (str(entry.get(k) or "").strip() for k in ("key_env", "api_key_env")) if name}
    return envs | {f"config:{str(entry.get('name') or '').strip()}", "model_config"}


def _row_serves(row: Any, own_sources: set[str], target: str) -> bool:
    from utils import same_base_url_origin
    if str(getattr(row, "source", "") or "").strip() in own_sources:
        return True
    url = str(getattr(row, "base_url", "") or "").strip()
    return bool(url and target) and same_base_url_origin(url, target)


def bind_unrecorded_rows(pool: Any, base_url: str) -> list[Any]:
    """Record ``base_url`` on the pool's manual rows that recorded no address (``hermes auth add``
    and dashboard rows written before writers recorded one): the key was entered while the entry
    pointed there. Persisted once when the active store owns the rows, so a later edit of the
    entry's address leaves them behind. Returns the rows it bound."""
    from dataclasses import replace

    from agent.credential_pool import CredentialPool, _is_manual_source, _profile_owns_pool_provider
    if not isinstance(pool, CredentialPool) or not base_url:
        return []
    unbound = [row for row in pool.entries()
               if _is_manual_source(str(row.source or "")) and not str(row.base_url or "").strip()]
    for row in unbound:
        pool._replace_entry(row, replace(row, base_url=base_url.rstrip("/")))
    if unbound and _profile_owns_pool_provider(pool.provider):
        pool._persist()
    return unbound


def custom_entry_for_endpoint(
    base_url: Optional[str], provider_name: Optional[str] = None, config: Optional[dict] = None,
) -> Optional[tuple[str, dict[str, Any]]]:
    """``(normalized_name, entry)`` of the configured custom entry that owns ``base_url``: by name
    first (a named entry's pool never serves another URL), else by URL. ``config`` is a frozen
    route's config, whose entry owns the endpoint even after a live edit."""
    from agent import credential_pool as cp
    if not base_url:
        return None
    url = cp._norm_url(base_url)
    if (named := named_custom_entry(provider_name, config)) is not None:
        entry_url = cp._norm_url(named[1].get("base_url"))
        return None if entry_url and entry_url != url else named
    return next(((n, e) for n, e in cp._iter_custom_providers(config) if url and cp._norm_url(e.get("base_url")) == url), None)


def named_custom_entry(provider_name: Optional[str], config: Optional[dict] = None) -> Optional[tuple[str, dict[str, Any]]]:
    """``(normalized_name, entry)`` of the configured custom entry ``provider_name`` names, else None."""
    from agent import credential_pool as cp
    aliases = cp._requested_custom_name_aliases(provider_name) if provider_name else set()
    return next(((n, e) for n, e in cp._iter_custom_providers(config) if aliases & cp._custom_entry_name_aliases(n, e)), None) if aliases else None


def pool_for_aux_provider(provider: Optional[str], pool_key: str, load: Callable[[str], Any]) -> Any:
    """``load(pool_key)`` for an auxiliary pool read (the key peek, credential recovery). A pinned
    ``custom:<name>`` normalizes to its entry's slug, which may also be a same-named built-in's pool,
    so it gets the entry-scoped view; a bare built-in name keeps its whole pool."""
    pool = load(pool_key)
    if pool is None or not str(provider or "").strip().lower().startswith("custom:"):
        return pool
    match = named_custom_entry(provider)
    return entry_scoped_pool(pool, pool_key, match[1] if match else None)


def _custom_identity(provider: Optional[str], requested_provider: Optional[str]) -> tuple[bool, Optional[str]]:
    """``(is_custom, custom:<name> or None)`` for a runtime's provider / requested_provider pair."""
    from agent.credential_pool import CUSTOM_POOL_PREFIX
    identities = [str(v or "").strip().lower() for v in (provider, requested_provider)]
    custom = any(v == "custom" or v.startswith(CUSTOM_POOL_PREFIX) for v in identities)
    return custom, next((v for v in identities if v.startswith(CUSTOM_POOL_PREFIX)), None)


def runtime_entry_owner(provider: Optional[str], requested_provider: Optional[str], base_url: Optional[str]) -> Optional[dict[str, Any]]:
    """Credential provenance of the entry that owns a custom runtime, for its ``_primary_runtime``
    snapshot: matched in the config the route runs on (the frozen scope, else the profile's), the way
    resolution matched it. Names only, never a secret. None for a built-in or an unowned endpoint."""
    custom, name = _custom_identity(provider, requested_provider)
    if not custom or (match := custom_entry_for_endpoint(base_url, name)) is None:
        return None
    norm_name, entry = match
    provenance = {k: entry.get(k) for k in ("provider_key", "base_url", "key_env", "api_key_env")}
    return {"name": norm_name, **provenance, "declares_credential": entry_declares_credential(entry)}


def load_runtime_pool(
    provider: Optional[str], base_url: Optional[str], requested_provider: Optional[str] = None,
    snapshot: Optional[dict[str, Any]] = None,
) -> Any:
    """The pool for a runtime snapshot's identity. A custom runtime (``custom`` or ``custom:<name>``)
    is the entry's, so it gets the entry-scoped view; a built-in identity loads its whole pool.
    Ownership and the pool key come from the runtime identity, never from the pool slug, which both
    share: a ``custom:<name>`` identity keys on its own entry, not the first entry on its URL.

    ``snapshot`` is the ``_primary_runtime`` being restored: the owner it recorded wins over live
    config, so an edit can't reassign the endpoint or drop the entry's declared credential. A named
    identity whose snapshot recorded no owner gets None and keeps the snapshot's own credential."""
    from agent.credential_pool import _pool_keys_for_custom_entry, load_pool, resolve_runtime_pool_key
    custom, name = _custom_identity(provider, requested_provider)
    if not custom:
        key = resolve_runtime_pool_key(provider, base_url)
        return load_pool(key) if key else None
    owner = (snapshot or {}).get(SNAPSHOT_OWNER_KEY)
    if snapshot is not None and name and not owner:
        return None
    match = (owner["name"], owner) if owner else custom_entry_for_endpoint(base_url, name)
    if name and match is None:
        return None  # a named identity off its entry's address has no pool to borrow (fail closed)
    key = (_pool_keys_for_custom_entry(*match) or [""])[0] if match and (name or owner) else resolve_runtime_pool_key(provider, base_url)
    return entry_scoped_pool(load_pool(key), key, match[1] if match else None, base_url) if key else None


def load_route_pool(provider: Optional[str], base_url: Optional[str], requested_provider: Optional[str] = None) -> Any:
    """The pool to attach when a live agent moves onto a route (fallback activation, ``/model``): a
    custom identity gets its entry's origin-bound view (``load_runtime_pool``), a built-in its own pool."""
    from agent.credential_pool import load_pool
    if _custom_identity(provider, requested_provider)[0]:
        return load_runtime_pool(provider, base_url, requested_provider)
    return load_pool(str(provider or "").strip().lower())

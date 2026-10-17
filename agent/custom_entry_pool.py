"""Credential ownership for ``providers:`` entries whose pool slug is also a built-in's pool id.

``providers.deepseek`` pools under the slug ``deepseek``, which ``load_pool`` also seeds from the
built-in DeepSeek provider's env vars and auth files. Those rows are the built-in's accounts. Every
acquisition of a pool for the custom entry's runtime (main, frozen and auxiliary resolution, a
URL-matched bare ``custom`` route, restoring the primary after a fallback, a delegated child's pool)
takes the entry-scoped view from here, so the built-in's account never stands in for the entry's
declared credential. The built-in provider itself still loads, and persists, the whole pool.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

_DECLARED_CREDENTIAL_KEYS = ("api_key", "key_env", "api_key_env", "key_cmd")


def entry_declares_credential(entry: Any) -> bool:
    """Whether a custom entry names its own credential (``api_key`` / ``key_env`` / ``key_cmd``)."""
    return isinstance(entry, dict) and any(str(entry.get(k) or "").strip() for k in _DECLARED_CREDENTIAL_KEYS)


def row_owned_by_entry(row: Any, entry: Any) -> bool:
    """Ownership by provenance: ``hermes auth add`` (manual) rows, and rows seeded from the entry's
    own ``key_env``. Rows the built-in seeded from its registry env vars or auth files are foreign."""
    from agent.credential_pool import _is_manual_source
    source = str(getattr(row, "source", "") or "").strip()
    own_env = {f"env:{name}" for name in (str(entry.get(k) or "").strip() for k in ("key_env", "api_key_env")) if name}
    return _is_manual_source(source) or source in own_env


def entry_scoped_pool(pool: Any, pool_key: str, entry: Optional[dict[str, Any]]) -> Any:
    """The rows of ``pool`` the custom ``entry`` owns: the pool itself when it owns them all, else a
    view of them, or None when it owns none. ``custom:<name>`` pools are seeded from the entry's own
    config and stay whole; so does any pool for an entry that declares no credential (as before).

    The view shares the loaded rows and token baselines, so persisting it merges into the stored
    pool under its lock and leaves the built-in's omitted rows on disk.
    """
    from agent.credential_pool import CUSTOM_POOL_PREFIX, CredentialPool
    if pool is None or str(pool_key or "").startswith(CUSTOM_POOL_PREFIX) or not entry_declares_credential(entry):
        return pool
    rows = pool.entries()
    owned = [row for row in rows if row_owned_by_entry(row, entry)]
    if len(owned) == len(rows):
        return pool
    if not owned:
        return None
    view = CredentialPool(pool.provider, owned)
    view._persisted_token_pairs = pool._persisted_token_pairs
    return view


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


def load_runtime_pool(provider: Optional[str], base_url: Optional[str], requested_provider: Optional[str] = None) -> Any:
    """The pool for a runtime snapshot's identity. A custom runtime (``custom`` or ``custom:<name>``)
    is the entry's, so it gets the entry-scoped view; a built-in identity loads its whole pool.
    Ownership and the pool key come from the runtime identity, never from the pool slug, which both
    share: a ``custom:<name>`` identity keys on its own entry, not the first entry on its URL."""
    from agent.credential_pool import CUSTOM_POOL_PREFIX, _pool_keys_for_custom_entry, load_pool, resolve_runtime_pool_key
    identities = [str(v or "").strip().lower() for v in (provider, requested_provider)]
    if not any(v == "custom" or v.startswith(CUSTOM_POOL_PREFIX) for v in identities):
        key = resolve_runtime_pool_key(provider, base_url)
        return load_pool(key) if key else None
    name = next((v for v in identities if v.startswith(CUSTOM_POOL_PREFIX)), None)
    match = custom_entry_for_endpoint(base_url, name)
    key = (_pool_keys_for_custom_entry(*match) or [""])[0] if name and match else resolve_runtime_pool_key(provider, base_url)
    return entry_scoped_pool(load_pool(key), key, match[1] if match else None) if key else None

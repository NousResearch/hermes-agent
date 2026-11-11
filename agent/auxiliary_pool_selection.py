"""Which pooled credential an auxiliary client is keyed on, and whether a cached one still holds.

Split from ``agent.auxiliary_client`` (facade size cap).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, NamedTuple, Optional

from agent.custom_entry_pool import pool_for_aux_provider

logger = logging.getLogger(__name__)

# Builders that send a pool entry themselves, so the pool strategy picks it: ONE ``select()`` per
# cache miss, recorded on the built client (``_stamp_client``) and checked on a hit without
# selecting (``_cached_credential_holds``). Their cache key digests every entry's key, after the build.
STRATEGY_SENT_PROVIDERS = frozenset({"nous", "openai-codex", "xai-oauth"})


class PoolSelection(NamedTuple):
    pool: Any   # None without a credentialed pool
    entry: Any  # None when every entry is filtered out for the model, or not peeked


def _aux_pool_selection(provider: str, model: Optional[str] = None, *, peek: bool = True) -> PoolSelection:
    """``(pool, entry)`` behind an auxiliary client, loaded under the canonical provider (``codex``
    is ``openai-codex``) and entry-scoped for a pinned ``custom:<name>``. ``entry`` is the
    non-mutating peek against ``model``'s cooldowns (skipped without ``peek``).
    """
    from agent.auxiliary_client import _load_pool_with_credentials, _normalize_aux_provider, _peek_pool_entry
    normalized = _normalize_aux_provider(provider)
    pool = pool_for_aux_provider(provider, normalized, lambda key: _load_pool_with_credentials(key, " (peek)"))
    return PoolSelection(pool, _peek_pool_entry(normalized, pool, model) if pool is not None and peek else None)


def _sent_selection(client: Any, pool: Any) -> Optional[PoolSelection]:
    """``(pool, entry)`` for the pool entry whose key ``client`` sends (any status), else None (a
    singleton credential, or a token the pool no longer holds)."""
    key = getattr(client, "api_key", None)
    entry_id = pool.entry_id_for_api_key(key) if pool is not None and isinstance(key, str) and key else None
    entry = next((e for e in pool.entries() if e.id == entry_id), None) if entry_id else None
    return PoolSelection(pool, entry) if entry is not None else None


def _stamp_client(client: Any, pool: Any, wire_model: Optional[str]) -> None:
    """Record on ``client`` the pool entry it sends (``""``: none, e.g. a singleton fallback) and
    the wire model it sends, for the check on a cache hit."""
    try:
        # Nothing available for the model: the client sends a singleton fallback, even when its token
        # equals a pooled entry's, so it records no entry (else every hit would rebuild it).
        sent = _sent_selection(client, pool) if pool is not None and pool.has_available(model=wire_model) else None
        client._hermes_aux_pool_entry = (getattr((sent or (None, None))[1], "id", ""), wire_model)
    except Exception as exc:  # an unstamped client never holds, so the next call rebuilds it
        logger.debug("Auxiliary client: could not record its pool entry: %s", exc, exc_info=True)


def _cached_credential_holds(client: Any, pool: Any) -> bool:
    """Whether the pool entry a cached client sends is still available for the model it sends
    (present, not exhausted, dead or cooling down), checked without selecting; a client sending no
    pool entry holds while none is available. Its token is in the cache key (``_pool_cache_hint``).
    """
    stamp = getattr(client, "_hermes_aux_pool_entry", None)
    if not isinstance(stamp, tuple) or len(stamp) != 2:
        return False
    entry_id, wire_model = stamp
    try:
        if not entry_id:
            return not pool.has_available(model=wire_model)
        with pool._lock:  # has_available()'s filter (the one peek() applies), matched by id
            available = pool._available_entries(model=wire_model)[0]
        # The filter can resync an entry from the auth store; a rotated token no longer holds.
        sent = getattr(client, "api_key", None)
        return any(e.id == entry_id and e.runtime_api_key == sent for e in available)
    except Exception as exc:
        logger.debug("Auxiliary client: could not check its cached pool entry: %s", exc, exc_info=True)
        return False


def _pool_credential_digest(pool: Any, entry: Any = None) -> str:
    """Secret-safe digest over the runtime key(s) backing the cache hint.

    When ``entry`` (the peeked selection) carries a key, only THAT key is digested: rotating a
    sibling account must not churn this entry's cached client. Otherwise every pool entry
    contributes, taken WITHOUT availability filters (``pool.entries()``, not ``peek()``): an entry
    benched by a per-model cooldown still counts, so a token rotated while the cooldown is active
    changes the digest too (#113022 point 5). Only the blake2b of the keys enters the cache key.
    """
    from agent.auxiliary_client import _pool_runtime_api_key
    keys = [k for k in (_pool_runtime_api_key(entry),) if k] if entry is not None else []
    if not keys:
        try:
            entries_fn = getattr(pool, "entries", None)
            entries = entries_fn() if callable(entries_fn) else []
        except Exception as exc:
            logger.debug("Auxiliary client: could not list pool entries for digest: %s", exc)
            return ""
        keys = sorted(k for k in (_pool_runtime_api_key(e) for e in entries or ()) if k)
    if not keys:
        return ""
    digest = hashlib.blake2b(digest_size=16)
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _pool_cache_hint(
    provider: str, *, main_runtime: Optional[dict[str, Any]] = None, model: Optional[str] = None,
    selection: Optional[PoolSelection] = None,
) -> str:
    """Return a cache discriminator that follows the pooled credential, not just its entry id.

    ``<provider>:<entry id>:<digest of that entry's key>`` when the selection names an entry,
    ``<provider>::<digest of every entry's key>`` when it names none but keyed entries exist: every
    entry is filtered out (cooldown), or a ``STRATEGY_SENT_PROVIDERS`` route, which is not peeked.
    ``selection`` is the build's ``_aux_pool_selection`` (taken here when absent; ``auto`` keys on
    the main provider's peek). A token rotated or refreshed by another process yields a new client
    instead of a 401 on the cached one (stale entries age out, #113022).
    """
    from agent.auxiliary_client import _normalize_aux_provider, _normalize_main_runtime, _read_main_provider
    normalized = _normalize_aux_provider(provider)
    peek = normalized not in STRATEGY_SENT_PROVIDERS
    if normalized == "auto":
        provider = _normalize_main_runtime(main_runtime).get("provider") or _read_main_provider()
        normalized, selection, peek = _normalize_aux_provider(provider), None, True
    if normalized in {"", "auto", "custom"}:
        return ""
    pool, entry = selection or _aux_pool_selection(provider, model, peek=peek)
    if pool is None:
        return ""
    digest = _pool_credential_digest(pool, entry)
    entry_id = str(getattr(entry, "id", "") or "").strip() if entry is not None else ""
    if not entry_id and not digest:
        return ""
    return f"{normalized}:{entry_id}:{digest}"

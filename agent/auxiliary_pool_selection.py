"""Which pooled credential an auxiliary client is keyed on.

Split from ``agent.auxiliary_client`` (facade size cap).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, NamedTuple, Optional

from agent.custom_entry_pool import pool_for_aux_provider

logger = logging.getLogger(__name__)


class PoolSelection(NamedTuple):
    pool: Any   # None without a credentialed pool
    entry: Any  # None when every entry is filtered out for the model


def _aux_pool_selection(provider: str, model: Optional[str] = None) -> PoolSelection:
    """``(pool, entry)`` behind an auxiliary client: ONE non-mutating peek that the cache key digests
    and the client build sends, so the two never name different credentials. Loaded under the
    canonical provider (``codex`` is ``openai-codex``), entry-scoped for a pinned ``custom:<name>``,
    peeked against ``model``'s cooldowns.
    """
    from agent.auxiliary_client import _load_pool_with_credentials, _normalize_aux_provider, _peek_pool_entry
    normalized = _normalize_aux_provider(provider)
    pool = pool_for_aux_provider(provider, normalized, lambda key: _load_pool_with_credentials(key, " (peek)"))
    return PoolSelection(pool, _peek_pool_entry(normalized, pool, model) if pool is not None else None)


def _take_pool_selection(selection: PoolSelection, provider: str, model: Optional[str]) -> Any:
    """The selection's peeked entry (the one the cache key digests), refreshed for sending. ``select()``
    still runs for the strategy's bookkeeping, as on the explicit-key routes; when it picks another
    entry the peeked one is reclaimed instead. With no peeked entry, ``select()``'s pick (usually None)."""
    from agent.auxiliary_client import _normalize_resolved_model
    pool, entry = selection
    wire_model = _normalize_resolved_model(model, provider) if model else None
    try:
        chosen = pool.select(model=wire_model)
        if entry is None or (chosen is not None and chosen.id == entry.id):
            return chosen
        return pool.reclaim(entry.id, model=wire_model)
    except Exception as exc:
        logger.debug("Auxiliary client: could not take the pool selection for %s: %s", provider, exc, exc_info=True)
        return None


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
    ``<provider>::<digest of every entry's key>`` when every entry is filtered out (cooldown)
    but keyed entries exist. ``selection`` is the build's ``_aux_pool_selection`` (taken here when
    absent; ``auto`` keys on the main provider's pool). A token rotated or refreshed by another
    process yields a new client instead of a 401 on the cached one (stale entries age out, #113022).
    """
    from agent.auxiliary_client import _normalize_aux_provider, _normalize_main_runtime, _read_main_provider
    normalized = _normalize_aux_provider(provider)
    if normalized == "auto":
        provider = _normalize_main_runtime(main_runtime).get("provider") or _read_main_provider()
        normalized, selection = _normalize_aux_provider(provider), None
    if normalized in {"", "auto", "custom"}:
        return ""
    pool, entry = selection or _aux_pool_selection(provider, model)
    if pool is None:
        return ""
    digest = _pool_credential_digest(pool, entry)
    entry_id = str(getattr(entry, "id", "") or "").strip() if entry is not None else ""
    if not entry_id and not digest:
        return ""
    return f"{normalized}:{entry_id}:{digest}"

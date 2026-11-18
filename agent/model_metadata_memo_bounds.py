"""Memory bounds for ``agent.model_metadata``'s per-credential in-process memos.

The memos are keyed by a credential fingerprint, and minted keys (Nous agent keys, OAuth access
tokens) rotate, so each rotation adds a key the process never reads again. Only entries no reader
can use are shed: expired ones, or the oldest past a count far above any real working set.
"""

import threading
import time
from typing import Any

from agent.cache_bounds import drop_expired, evict_oldest
from agent.codex_extended_context import catalog_extended_budgets

# Same bound as ``agent.codex_service_tiers._MAX_CATALOGS`` for the same token-fingerprint keys.
CODEX_CACHE_MAX = 128
# Every writer of a paired memo (value + stamp, context + max-context) holds this from sweep to publish:
# a sweep pops two tables, and a refresh landing between the pops would keep half an entry.
MEMO_LOCK = threading.Lock()


def remember_endpoint_models(memo_key: tuple[str, str], cache: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Publish a ``/models`` catalog, sweeping expired ones (the reader refetches those anyway)."""
    from agent import model_metadata as mm  # facade seam: tests swap the memo dicts there
    with MEMO_LOCK:
        now = time.time()
        drop_expired(mm._endpoint_model_metadata_cache, mm._endpoint_model_metadata_cache_time, mm._ENDPOINT_MODEL_CACHE_TTL, now)
        mm._endpoint_model_metadata_cache[memo_key] = cache
        mm._endpoint_model_metadata_cache_time[memo_key] = now
    return cache


def bound_codex_caches(cache_key: str) -> None:
    """Before storing ``cache_key``, evict the stalest Codex token entry from both context memos.
    Caller holds ``MEMO_LOCK`` through its publish."""
    from agent import model_metadata as mm  # facade seam: tests swap the memo dicts there
    evict_oldest(mm._codex_oauth_context_cache, cache_key, lambda entry: entry[1], CODEX_CACHE_MAX,
                 mm._codex_oauth_max_context_cache, mm._codex_oauth_extended_cache)


def remember_codex_context_catalog(access_token: str, base_url: str, entries: list) -> dict[str, int]:
    """Publish a successful, credential-bound discovery snapshot for the runtime and picker."""
    from agent import model_metadata as mm
    from agent.codex_service_tiers import remember_catalog
    cache_key = mm._codex_oauth_token_fingerprint(access_token, base_url)
    remember_catalog(cache_key, entries)
    result: dict[str, int] = {}
    max_result: dict[str, int] = {}
    for item in entries:
        slug, ctx, max_ctx = (item.get("slug"), item.get("context_window"), item.get("max_context_window")) if isinstance(item, dict) else (None, None, None)
        if isinstance(slug, str) and isinstance(ctx, int) and ctx > 0:
            result[slug.strip()] = ctx
            if isinstance(max_ctx, int) and max_ctx > 0:
                max_result[slug.strip()] = max_ctx
    if not result:
        return {}
    extended = catalog_extended_budgets(entries)
    with MEMO_LOCK:
        bound_codex_caches(cache_key)
        mm._codex_oauth_extended_cache[cache_key] = extended
        mm._codex_oauth_max_context_cache[cache_key] = max_result
        mm._codex_oauth_context_cache[cache_key] = (result, time.time())
    return result


def cached_codex_extended_budgets(access_token: str, base_url: str) -> dict[str, int]:
    """Return a fresh snapshot for this credential and endpoint without network I/O."""
    from agent import model_metadata as mm

    key = mm._codex_oauth_token_fingerprint(access_token, base_url)
    with MEMO_LOCK:
        entry = mm._codex_oauth_context_cache.get(key)
        if entry and entry[0] and time.time() - entry[1] < mm._CODEX_OAUTH_CONTEXT_CACHE_TTL:
            return mm._codex_oauth_extended_cache.get(key, {}).copy()
    return {}

"""Memory bounds for ``agent.model_metadata``'s per-credential in-process memos.

The memos are keyed by a credential fingerprint, and minted keys (Nous agent keys, OAuth access
tokens) rotate, so each rotation adds a key the process never reads again. Only entries no reader
can use are shed: expired ones, or the oldest past a count far above any real working set.
"""

import threading
import time
from typing import Any

from agent.cache_bounds import drop_expired, evict_oldest

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


def forget_route_context_length(model: str, base_url: str) -> None:
    """Forget the cached window of a route the user just switched to, including the ``/models``
    memo of its endpoint: a server restarted with a different window under the same model id is
    asked again instead of answering from a memo of the old one (#159799)."""
    from agent import model_metadata as mm  # facade seam: tests swap the memo dicts there
    if not model or not base_url:
        return
    mm._invalidate_cached_context_length(model, base_url)
    normalized = mm._normalize_base_url(base_url)
    with MEMO_LOCK:
        for memo_key in [k for k in mm._endpoint_model_metadata_cache if k[0] == normalized]:
            mm._endpoint_model_metadata_cache.pop(memo_key, None)
            mm._endpoint_model_metadata_cache_time.pop(memo_key, None)


def bound_codex_caches(cache_key: str) -> None:
    """Before storing ``cache_key``, evict the stalest Codex token entry from both context memos.
    Caller holds ``MEMO_LOCK`` through its publish."""
    from agent import model_metadata as mm  # facade seam: tests swap the memo dicts there
    evict_oldest(mm._codex_oauth_context_cache, cache_key, lambda entry: entry[1], CODEX_CACHE_MAX, mm._codex_oauth_max_context_cache)

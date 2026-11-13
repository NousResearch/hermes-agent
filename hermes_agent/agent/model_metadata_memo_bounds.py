"""Memory bounds for ``hermes_agent.agent.model_metadata``'s per-credential in-process memos.

The memos are keyed by a credential fingerprint, and minted keys (Nous agent keys, OAuth access
tokens) rotate, so each rotation adds a key the process never reads again. Only entries no reader
can use are shed: expired ones, or the oldest past a count far above any real working set.
"""

import threading
import time
from typing import Any

from hermes_agent.agent.cache_bounds import drop_expired, evict_oldest

# Same bound as ``hermes_agent.agent.codex_service_tiers._MAX_CATALOGS`` for the same token-fingerprint keys.
CODEX_CACHE_MAX = 128
# Every writer of a paired memo (value + stamp, context + max-context) holds this from sweep to publish:
# a sweep pops two tables, and a refresh landing between the pops would keep half an entry.
MEMO_LOCK = threading.Lock()


def remember_endpoint_models(memo_key: tuple[str, str], cache: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Publish a ``/models`` catalog, sweeping expired ones (the reader refetches those anyway)."""
    from hermes_agent.agent import model_metadata as mm  # facade seam: tests swap the memo dicts there
    with MEMO_LOCK:
        now = time.time()
        drop_expired(mm._endpoint_model_metadata_cache, mm._endpoint_model_metadata_cache_time, mm._ENDPOINT_MODEL_CACHE_TTL, now)
        mm._endpoint_model_metadata_cache[memo_key] = cache
        mm._endpoint_model_metadata_cache_time[memo_key] = now
    return cache


def bound_codex_caches(cache_key: str) -> None:
    """Before storing ``cache_key``, evict the stalest Codex token entry from both context memos.
    Caller holds ``MEMO_LOCK`` through its publish."""
    from hermes_agent.agent import model_metadata as mm  # facade seam: tests swap the memo dicts there
    evict_oldest(mm._codex_oauth_context_cache, cache_key, lambda entry: entry[1], CODEX_CACHE_MAX, mm._codex_oauth_max_context_cache)

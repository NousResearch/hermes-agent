"""The in-process ``/models`` metadata memo of ``agent.model_metadata``, per (endpoint, credential).

It is keyed per credential fingerprint because a governed endpoint answers each credential with its
org's catalog. Minted credentials rotate (Nous agent keys), so every rotation adds a key, and an entry
older than ``_ENDPOINT_MODEL_CACHE_TTL`` is never read again. Each write drops those, so the memo holds
only the credentials used within one TTL; without that, a gateway that had run for three days held
141 copies of the same catalog (about 70 MB).
"""
from __future__ import annotations

import time
from typing import Any


def endpoint_memo_key(normalized: str, api_key: object) -> tuple[str, str]:
    from agent.credential_persistence import fingerprint_secret_value
    # Callable (minted) keys are not fingerprinted here: doing so would mint on every cache hit.
    return normalized, (fingerprint_secret_value(api_key) or "") if isinstance(api_key, str) else ""


def remember_endpoint_models(memo_key: tuple[str, str], cache: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from agent.model_metadata import (
        _ENDPOINT_MODEL_CACHE_TTL, _endpoint_model_metadata_cache as memo, _endpoint_model_metadata_cache_time as stamps)
    now = time.time()
    for key in [key for key, at in stamps.items() if now - at >= _ENDPOINT_MODEL_CACHE_TTL]:
        memo.pop(key, None)
        stamps.pop(key, None)
    memo[memo_key] = cache
    stamps[memo_key] = now
    return cache

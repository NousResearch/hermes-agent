"""Point a post-rotation auxiliary retry at the rotated-to pool key.

Split from ``agent.auxiliary_client`` (facade size cap).
"""

from __future__ import annotations

from typing import Any, Optional


def runtime_with_rotated_pool_key(main_runtime: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """``main_runtime`` with its ``api_key`` swapped for the pool's current key after a rotation.

    Auto routes rebuild the retry client from the main runtime, whose key is the one the pool just
    benched, so the retry would resend the dead key and never reach the healthy entry. Swap only a
    key that is itself a member of the runtime provider's pool; any other session key (OAuth,
    callable token provider, a key from another pool) is returned unchanged.
    """
    from agent.auxiliary_client import (
        _load_pool_with_credentials,
        _normalize_aux_provider,
        _normalize_main_runtime,
        _pool_runtime_api_key,
    )

    runtime = _normalize_main_runtime(main_runtime)
    key = runtime.get("api_key")
    provider = _normalize_aux_provider(runtime.get("provider") or "")
    if not isinstance(key, str) or not key or provider in {"", "auto", "custom"}:
        return main_runtime
    pool = _load_pool_with_credentials(provider, " (retry)")
    if pool is None or key not in {_pool_runtime_api_key(e) for e in pool.entries()}:
        return main_runtime
    base_url = runtime.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        return main_runtime  # fail closed: without an endpoint we cannot tell whose key this is
    try:
        current = _pool_runtime_api_key(pool.peek(model=runtime.get("model") or None, endpoint_url=base_url))
    except Exception:  # health: allow BLE001 -- a broken pool must not break the retry; keep the runtime key
        return main_runtime
    if not current or current == key:
        return main_runtime
    return {**runtime, "api_key": current}

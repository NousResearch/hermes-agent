"""Point a post-rotation auxiliary retry at the rotated-to pool key.

Split from ``agent.auxiliary_client`` (facade size cap).
"""

from __future__ import annotations

from typing import Any, Optional


def adopt_rotated_entry(runtime: Optional[dict[str, Any]], failed_key: str, entry: Any, pool: Any) -> None:
    """Swap the ``api_key`` of ``runtime`` (IN PLACE) for the pool entry a rotation just chose.

    Auto routes rebuild the retry client from the main runtime, whose key may be the one the pool just
    benched, so the retry would resend the dead key and never reach the healthy entry. Swap only when the
    benched key IS the runtime's key and a member of ``pool``, and only for an entry that serves the
    runtime's ``base_url`` and is not cooled down for its model. Anything else (OAuth, callable token
    provider, a key from another pool or endpoint, no ``base_url``) leaves the runtime unchanged.
    """
    from agent.auxiliary_client import _pool_runtime_api_key
    from agent.credential_pool import credential_pool_entry_serves_endpoint
    from agent.credential_pool_model_cooldowns import model_cooldown_until

    if not isinstance(runtime, dict):
        return
    new = _pool_runtime_api_key(entry)
    base_url = runtime.get("base_url")
    if not (failed_key and new and new != failed_key and runtime.get("api_key") == failed_key
            and isinstance(base_url, str) and base_url  # fail closed: no endpoint, no way to tell whose key this is
            and credential_pool_entry_serves_endpoint(entry, base_url)
            and model_cooldown_until(entry, runtime.get("model") or None) is None
            and failed_key in {_pool_runtime_api_key(e) for e in pool.entries()}):
        return
    runtime["api_key"] = new

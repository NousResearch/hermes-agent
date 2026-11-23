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
    runtime's ``base_url`` and is not cooled down for its model; otherwise fall back to the pool's next
    guarded ``peek`` for that endpoint and model (same pool object, no reload). Anything else (OAuth, callable token
    provider, a key from another pool or endpoint, no ``base_url``) leaves the runtime unchanged.
    """
    from agent.auxiliary_client import _pool_runtime_api_key
    from agent.auxiliary_key_rotation import rotate_runtime_main_api_key
    from agent.credential_pool import credential_pool_entry_serves_endpoint
    from agent.credential_pool_model_cooldowns import model_cooldown_until

    if not isinstance(runtime, dict):
        return
    base_url, model = runtime.get("base_url"), runtime.get("model") or None
    if not (failed_key and runtime.get("api_key") == failed_key
            and isinstance(base_url, str) and base_url  # fail closed: no endpoint, no way to tell whose key this is
            and failed_key in {_pool_runtime_api_key(e) for e in pool.entries()}):
        return
    if not (entry is not None and credential_pool_entry_serves_endpoint(entry, base_url)
            and model_cooldown_until(entry, model) is None):
        try:  # the rotation picked a row for another endpoint/model: take this endpoint's next row instead
            entry = pool.peek(model=model, endpoint_url=base_url)
        except Exception:  # health: allow BLE001 -- a broken pool must not break the retry; keep the runtime key
            return
    new = _pool_runtime_api_key(entry) if entry is not None else ""
    if not new or new == failed_key:
        return
    runtime["api_key"] = new
    rotate_runtime_main_api_key(failed_key, new)  # the published runtime too, so later aux calls skip the dead key

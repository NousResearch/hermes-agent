"""Same-provider auxiliary recovery retains the credential selected for its retry."""

import logging
from typing import Any, Optional

from agent.credential_pool import PooledCredential

logger = logging.getLogger(__name__)


def recover_provider_pool(provider: str, exc: Exception, *, failed_api_key: str) -> Optional[PooledCredential]:
    """Recover a pool-owned failure and return the replacement, not just success.

    The caller binds its retry to this entry; re-resolving an ``auto`` route
    would otherwise replay the main session's stale credential snapshot.
    """
    from agent import auxiliary_client as aux

    normalized = aux._normalize_aux_provider(provider)
    try:
        pool = aux.load_pool(normalized)
    except Exception as load_exc:
        logger.debug("Auxiliary client: could not load pool for %s recovery: %s", normalized, load_exc, exc_info=True)
        return None
    if not pool or not pool.has_credentials():
        return None
    # A caller-pinned key outside this pool is a different credential authority.
    if not failed_api_key or not any(aux._pool_runtime_api_key(entry) == failed_api_key for entry in pool.entries()):
        return None
    status_code = getattr(exc, "status_code", None)

    def _rotate(fallback_status: int) -> Optional[PooledCredential]:
        error_context: dict[str, Any] = {"message": str(exc)}
        if status_code is not None:
            error_context["status_code"] = status_code
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else fallback_status,
            error_context=error_context, api_key_hint=failed_api_key,
        )
        if next_entry is not None:
            aux._evict_cached_clients(normalized)
        return next_entry

    if aux._is_auth_error(exc):
        refreshed = pool.try_refresh_matching(api_key_hint=failed_api_key)
        if refreshed is not None:
            aux._evict_cached_clients(normalized)
            return refreshed
        return _rotate(401)
    if aux._is_payment_error(exc):
        return _rotate(402)
    if aux._is_rate_limit_error(exc) and not aux._is_overloaded_error(exc):
        return _rotate(429)
    return None

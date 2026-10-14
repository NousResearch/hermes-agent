"""Provider and endpoint identity boundaries for credential pools."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent import credential_pool as _pool

logger = logging.getLogger(__name__)


def credential_pool_matches_provider(
    pool_or_provider: Any,
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    requested_provider: Optional[str] = None,
) -> bool:
    """Return whether a pool belongs to the requested runtime provider.

    Named custom endpoints may use three identities: the live agent can retain
    the configured name/provider key, newer runtime paths normalize it to
    ``custom``, and the pool may be keyed as the durable ``providers.<key>``
    slug or as legacy ``custom:<name>``. Accept those aliases only when the
    runtime endpoint belongs to the same configured custom provider. A bare
    ``custom`` agent routed through an unconfigured relayer can also identify
    a legacy pool by its exact ``custom:<name>`` requested_provider. That
    identity never overrides an endpoint mapped to a different custom pool.
    Empty identities fail closed. Legacy pool adapters without a ``provider``
    attribute remain compatible; production pools are scoped.
    """
    raw_pool_provider = getattr(pool_or_provider, "provider", None)
    if raw_pool_provider is None:
        if not isinstance(pool_or_provider, str):
            # Lightweight/unscoped pool adapters (old plugins, tests) may
            # expose only select()/has_credentials().
            return True
        raw_pool_provider = pool_or_provider
    pool_provider = str(raw_pool_provider or "").strip().lower()
    provider_norm = str(provider or "").strip().lower()
    if not pool_provider or not provider_norm:
        return False
    if not pool_provider.startswith(_pool.CUSTOM_POOL_PREFIX):
        if pool_provider == provider_norm:
            return True
        return _pool._keyed_custom_pool_matches(pool_provider, provider_norm, base_url)
    if provider_norm == "custom":
        try:
            matched_pool = _pool.get_custom_provider_pool_key(base_url or "")
            if str(matched_pool or "").strip().lower() == pool_provider:
                return True
            candidates = _pool.custom_provider_pool_key_candidates(base_url or "")
        except Exception:
            logger.debug("Failed to resolve custom provider pool identity", exc_info=True)
            return False
        candidate_keys = {str(key).strip().lower() for key in candidates}
        if pool_provider in candidate_keys:
            return True
        # Only an unconfigured relayer may use the explicit named identity.
        # A known different endpoint remains a mismatch, even if a fallback
        # retained the primary's requested_provider.
        requested_norm = requested_provider.strip().lower() if isinstance(requested_provider, str) else ""
        return (
            not matched_pool
            and not candidate_keys
            and bool(_pool._norm_url(base_url))
            and bool(pool_provider[len(_pool.CUSTOM_POOL_PREFIX):].strip())
            and requested_norm == pool_provider
        )

    runtime_url = _pool._norm_url(base_url)
    if not runtime_url:
        return False
    return _pool._legacy_custom_pool_matches(pool_provider, provider_norm, runtime_url)


def resolve_runtime_pool_key(
    provider: Optional[str],
    base_url: Optional[str],
    *,
    requested_provider: Optional[str] = None,
) -> str:
    """Resolve the credential-pool key for a runtime provider identity.

    Named custom runtimes retain their configured alias while their pool may
    be stored under the durable ``providers.<key>`` slug or legacy
    ``custom:<name>``. Return that scoped key only when the canonical
    provider/endpoint boundary accepts it; otherwise preserve the normalized
    runtime identity so callers fail closed.
    """
    provider_norm = str(provider or "").strip().lower()
    if not provider_norm:
        return ""

    def _accepts(candidate: str) -> bool:
        return credential_pool_matches_provider(
            candidate,
            provider_norm,
            base_url=base_url,
            requested_provider=requested_provider,
        )

    try:
        if provider_norm == "custom":
            candidate = _pool.get_custom_provider_pool_key(base_url)
            if candidate and _accepts(candidate):
                return str(candidate).strip().lower()
            requested_norm = str(requested_provider or "").strip().lower()
            if requested_norm.startswith(_pool.CUSTOM_POOL_PREFIX) and _accepts(requested_norm):
                return requested_norm
        else:
            # Named/exact custom runtimes are keyed by identity: search the
            # configured candidates by identity before endpoint so a sibling
            # sharing the URL cannot lend its pool.
            for normalized_name, entry in _pool._iter_custom_providers():
                for candidate in _pool._pool_keys_for_custom_entry(normalized_name, entry):
                    if _accepts(candidate):
                        return candidate
    except Exception:
        logger.debug("Failed to resolve runtime credential-pool key", exc_info=True)
    return provider_norm

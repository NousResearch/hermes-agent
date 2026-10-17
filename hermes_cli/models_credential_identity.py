"""Credential-identity digests for the provider-models catalog cache.

Extracted from ``hermes_cli.models`` (the facade is over its FILE_LINES ratchet): the
per-provider identity that keys catalog cache rows, plus the custom-endpoint fingerprint
that serves the same role for ``custom:<base_url>`` rows.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional


def _auth_store_provider_identity(provider: str) -> Optional[str]:
    """Digest of *provider*'s own credentials in the shared auth.json, for the catalog fingerprint.

    Covers the credential-pool entries and the ``providers.<id>`` state with the routinely rotating
    fields stripped — the same token/status field sets auth treats as generation-coupled, plus the
    pool-row runtime counters — so a token refresh, a pool counter bump or another provider's write
    leaves the digest unchanged, while a real credential switch (entry added/removed, a new static
    key, a different login identity) changes it. ``None`` when the store or its reader is
    unavailable; the caller then falls back to the whole-file mtime, which degrades to a spurious
    cache miss, never a stale hit."""
    try:
        from hermes_cli.auth import (
            _FLAT_OAUTH_TOKEN_KEYS,
            _POOL_STATUS_FIELDS,
            _POOL_TOKEN_GENERATION_FIELDS,
            _load_auth_store,
            _load_provider_state,
            read_credential_pool,
        )
    except Exception:  # health: allow BLE001 -- a missing auth surface must degrade to the mtime key, not kill catalog lookups
        return None
    volatile = (
        set(_POOL_STATUS_FIELDS)
        | set(_POOL_TOKEN_GENERATION_FIELDS)
        | set(_FLAT_OAUTH_TOKEN_KEYS)
        | {
            # Pool-row runtime state outside the auth field sets: the per-selection counter,
            # model-level cooldowns and the persisted exhaustion classification.
            "request_count",
            "model_cooldowns",
            "failure_reason",
            # Derived from / refreshed with the rotating secrets themselves.
            "anon_token",
            "secret_fingerprint",
            # Store-level write timestamp.
            "updated_at",
        }
    )

    def _stable(value):
        if isinstance(value, dict):
            return {k: _stable(v) for k, v in sorted(value.items()) if k not in volatile}
        if isinstance(value, list):
            return [_stable(v) for v in value]
        return value

    def _blob(value) -> str:
        return json.dumps(_stable(value), sort_keys=True, separators=(",", ":"), default=str)

    try:
        # Pool entries are an unordered set of credentials: digest them sorted so a row re-order
        # (merge, pool rewrite) does not invalidate the catalog; nested lists keep their order.
        pool = sorted(_blob(entry) for entry in read_credential_pool(provider))
        state = _blob(_load_provider_state(_load_auth_store(), provider))
        blob = f"pool={pool}|state={state}"
        return hashlib.blake2b(
            blob.encode("utf-8", errors="replace"), digest_size=8
        ).hexdigest()
    except Exception:  # health: allow BLE001 -- an unreadable store falls back to the caller's mtime key: a spurious cache miss, never a stale hit
        return None


def _custom_endpoint_fingerprint(
    api_key: Any, api_mode: Optional[str], headers: Optional[dict[str, str]]) -> str:
    """Custom endpoints have no ``PROVIDER_REGISTRY`` slug, so hash exactly what callers pass to
    ``fetch_api_models``: a rotated ``api_key``, changed ``api_mode`` or edited ``extra_headers``
    each bust the cache entry. blake2b for the same CodeQL rationale as
    ``_credential_fingerprint``."""
    from agent.command_token_source import CommandTokenSource

    identity = api_key.cache_identity if isinstance(api_key, CommandTokenSource) else api_key
    blob = "|".join((identity or "", api_mode or "", json.dumps(headers or {}, sort_keys=True)))
    return hashlib.blake2b(blob.encode("utf-8", errors="replace"), digest_size=8).hexdigest()

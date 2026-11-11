"""Credential renewal and rotation for an already resolved auxiliary provider pool."""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from agent import auxiliary_error_policy as aux_errors
from agent.credential_pool_model_cooldowns import pool_failure_reason

# --- Derived runtime credentials across auxiliary pool reloads ---
# Auxiliary reads reload the pool from disk, so a row's in-memory override would be lost and the
# durable grant sent instead. Overrides are kept here keyed by (profile home, provider, row id) and
# re-attached to a reloaded row only while its durable grant is unchanged (active_runtime_override).

_OVERRIDES: dict[tuple[str, str, str], Any] = {}
_OVERRIDES_LOCK = threading.Lock()


def clear_runtime_overrides() -> None:
    with _OVERRIDES_LOCK:
        _OVERRIDES.clear()


def _home_key() -> str:
    from hermes_constants import get_hermes_home, hermes_home_key
    return hermes_home_key(get_hermes_home())


def uses_runtime_credentials(pool: Any) -> bool:
    """True when the pool's provider derives runtime bearers (its rows must be prepared before dispatch)."""
    from agent.credential_pool_plugin import _overrides, _profile
    provider = getattr(pool, "provider", "")
    if not provider or provider == "nous" or not callable(getattr(pool, "prepare_runtime_entry", None)):
        return False
    profile = _profile(provider)
    return profile is not None and (_overrides(profile, "prepare_pool_entry")
                                    or _overrides(profile, "refresh_pool_runtime_credentials"))


def provider_derives_runtime_credentials(provider: str) -> bool:
    """True when ``provider``'s profile derives runtime bearers: its ambient key may be a durable grant."""
    from agent.credential_pool_plugin import _overrides, _profile
    profile = _profile(provider) if provider and provider != "nous" else None
    return profile is not None and (_overrides(profile, "prepare_pool_entry")
                                    or _overrides(profile, "refresh_pool_runtime_credentials"))


def stamp_entry(client: Any, entry: Any, api_key: str) -> Any:
    """Attribute ``client`` to the pool row whose runtime key it sends, so a 401 is scoped to that row."""
    entry_id = getattr(entry, "id", None)
    if (client is not None and isinstance(entry_id, str) and entry_id
            and str(getattr(entry, "runtime_api_key", "") or "") == api_key):
        client._hermes_credential_pool_entry_id = entry_id
    return client


def api_key_chain_credentials(provider_id: str, pconfig: Any, model: str,
                              resolve_ambient: Callable[[str], dict]) -> Optional[tuple[str, str, Any, str]]:
    """``(api_key, raw_base_url, pool_entry, via)`` for the auto api-key chain, or None to skip the provider.

    A pooled provider sends its selected (prepared) row's runtime key; without a pool, a provider that
    derives runtime bearers is skipped, since its ambient key may be the durable grant.
    """
    from agent import auxiliary_client as aux
    pool_present, entry = aux._select_pool_entry(provider_id, model)
    if pool_present:
        api_key = aux._pool_runtime_api_key(entry)
        raw_base_url = aux._pool_runtime_base_url(entry, pconfig.inference_base_url) or pconfig.inference_base_url
        return (api_key, raw_base_url, entry, " via pool") if api_key else None
    if provider_derives_runtime_credentials(provider_id):
        return None
    creds = resolve_ambient(provider_id)
    api_key = str(creds.get("api_key", "")).strip()
    raw_base_url = str(creds.get("base_url", "")).strip().rstrip("/") or pconfig.inference_base_url
    return (api_key, raw_base_url, None, "") if api_key else None


def rehydrate(pool: Any) -> Any:
    """Re-attach this profile's remembered overrides to a freshly loaded pool's rows."""
    from agent.credential_pool_plugin import _install, active_runtime_override
    if pool is None or not callable(getattr(pool, "entries", None)) or not uses_runtime_credentials(pool):
        return pool
    home = _home_key()
    with _OVERRIDES_LOCK:
        for row in pool.entries():
            key = (home, pool.provider, getattr(row, "id", ""))
            override = _OVERRIDES.get(key)
            if override is None or active_runtime_override(row) is not None:
                continue
            _install(row, override)
            if active_runtime_override(row) is None:  # grant rotated: the derived bearer is stale
                _install(row, None)
                _OVERRIDES.pop(key, None)
    return pool


def remember(pool: Any) -> None:
    """Record the pool's live overrides under the active profile."""
    from agent.credential_pool_plugin import active_runtime_override
    if pool is None or not callable(getattr(pool, "entries", None)):
        return
    home = _home_key()
    with _OVERRIDES_LOCK:
        for row in pool.entries():
            override = active_runtime_override(row)
            if override is not None and getattr(row, "id", ""):
                _OVERRIDES[(home, pool.provider, row.id)] = override


def live_loader(load: Callable[[str], Any]) -> Callable[[str], Any]:
    return lambda key: rehydrate(load(key))


def prepare_entry(pool: Any, entry: Any, *, source_base_url: Any = None) -> Any:
    """The row ready for dispatch (derived bearer installed), or None when it must not be adopted."""
    if entry is None or not uses_runtime_credentials(pool):
        return entry
    rehydrate(pool)  # select() may replace row objects, dropping an attached override
    prepared = pool.prepare_runtime_entry(entry, source_base_url=source_base_url)
    if prepared is not None:
        remember(pool)
    return prepared


def entry_for_explicit_key(pool: Any, api_key: str, *, source_base_url: Any = None) -> tuple[Any, Optional[str]]:
    """``(row, key_to_send)`` for an explicitly shared bearer.

    A unique derived-bearer match is attributed to its row only when that row may be adopted from
    ``source_base_url`` (exact allowlisted origin); otherwise nothing is sent. A durable grant
    backing a derived override is replaced by the prepared bearer (None if refused). Refresh-only
    profiles and legacy identity-string hooks may already store an inference key and keep it.
    """
    from agent.credential_pool_plugin import active_runtime_override, durable_runtime_key
    rows = pool.entries() if pool is not None else []
    derives = uses_runtime_credentials(pool)
    grants = [row for row in rows if durable_runtime_key(row) == api_key or row.access_token == api_key] if derives else []
    if not grants:
        matches = [row for row in rows if row.runtime_api_key == api_key]
        if not derives:  # legacy providers: unchanged attribution, key sent as given
            return (matches[0] if len(matches) == 1 else None), api_key
        if matches:
            # A derived bearer is sent only where its row may be adopted from the explicit endpoint.
            prepared = prepare_entry(pool, matches[0], source_base_url=source_base_url) if len(matches) == 1 else None
            if prepared is None or getattr(prepared, "runtime_api_key", None) != api_key:
                return None, None
            return prepared, api_key
        # Unattributed key for a provider with an allowlist: never to an origin outside it.
        from agent.credential_pool_plugin import _allowed_origins, _profile, exact_https_origin
        allowed = _allowed_origins(_profile(getattr(pool, "provider", "")))
        if source_base_url is not None and allowed and exact_https_origin(source_base_url) not in allowed:
            return None, None
        return None, api_key
    prepared = prepare_entry(pool, grants[0], source_base_url=source_base_url) if len(grants) == 1 else None
    key = str(getattr(prepared, "runtime_api_key", "") or "") if prepared is not None else ""
    # Refresh-only and legacy identity-string hooks already hold an inference key.
    # An installed derived override, however, must never replay its durable grant.
    if not key or (key == api_key and active_runtime_override(prepared) is not None):
        return None, None
    return prepared, key


def recover_loaded_pool(
    pool: Any, provider: str, exc: Exception, verdict: aux_errors.ClassifiedError, *,
    failed_api_key: str = "", failed_credential_id: str | None = None,
) -> bool:
    """Recover the dispatched credential, evicting cached clients only on success."""
    from agent import auxiliary_client as aux

    status_code = verdict.status_code
    gated = verdict.reason is aux.FailoverReason.model_entitlement

    def _rotate(fallback_status: int) -> bool:
        # Match the main loop's status, pool failure reason and model attribution.
        error_context: dict[str, Any] = {"message": str(exc)}
        if status_code is not None:
            error_context["status_code"] = status_code
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=fallback_status if status_code is None else status_code,
            error_context=error_context, api_key_hint=failed_api_key or None,
            failure_reason=pool_failure_reason(verdict.reason, verdict.error_context, verdict.billing_unverified),
            model=verdict.model or None,
            **({"credential_id": failed_credential_id} if failed_credential_id else {}),
        )
        if next_entry is None:
            return False
        aux._evict_cached_clients(provider)
        return True

    if aux_errors.auth_refresh(verdict):
        from agent.credential_auth_recovery import pool_runtime_refresh_profile
        if pool_runtime_refresh_profile(pool) is not None:
            from agent.credential_pool import CredentialPool
            from agent.credential_pool_plugin import _scoped_row
            if isinstance(pool, CredentialPool) and _scoped_row(pool, failed_credential_id, failed_api_key or None) is None:
                return False
            refreshed = pool.try_refresh_runtime_matching(
                credential_id=failed_credential_id, api_key_hint=failed_api_key or None, reason="unauthorized")
        else:
            refreshed = pool.try_refresh_current()
        if refreshed is not None:
            remember(pool)  # the retry reloads the pool; it must adopt the renewed bearer
            aux._evict_cached_clients(provider)
            return True
        return _rotate(401)
    if not verdict.should_rotate_credential:  # an overloaded server or upstream model throttle
        return False
    return _rotate(402 if aux_errors.benches_provider(verdict) else 400 if gated else 429)

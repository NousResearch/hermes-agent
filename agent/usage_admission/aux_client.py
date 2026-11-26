"""Usage-admission checks for auxiliary Codex clients (selection, cache key and client build agree)."""

from __future__ import annotations

from typing import Any, NamedTuple, Optional

from agent.usage_admission import policy


class CodexAdmission(NamedTuple):
    """Usage-admission verdict for an auxiliary client request (``codex_admission_for_client``)."""
    denied: bool = False
    evict: bool = False  # drop this provider's cached clients: every pool entry is over its ceiling
    pool_hint: Optional[str] = None  # cache-key discriminator naming the exact admitted entry
    api_key: str = ""  # the admitted credential, so the client is built with the key that was checked
    base_url: str = ""
    provider: str = ""  # "openai-codex" when an ``auto`` route resolved to a pool entry


def codex_admission_for_client(
    provider: str, runtime: dict[str, Any], api_key: Optional[str], base_url: Optional[str],
) -> CodexAdmission:
    """Apply ``credential_pool_usage_limits`` to an auxiliary Codex route.

    Selects the pool entry ONCE (admission, cache key and client build must agree on the account) and
    returns it; an explicit or inherited-explicit key is checked by its own fingerprint instead. Inert
    for other providers and when no limits are configured."""
    from agent import auxiliary_client as aux  # auxiliary_client imports this module

    admission_provider = aux._normalize_aux_provider(provider)
    if admission_provider == "auto":
        admission_provider = aux._normalize_aux_provider(runtime.get("provider") or aux._read_main_provider())
    if admission_provider != "openai-codex" or not policy.get_pool_usage_limits("openai-codex"):
        return CodexAdmission()
    runtime_api_key = runtime.get("api_key")
    inherited_explicit_key = (
        not api_key and isinstance(runtime_api_key, str) and bool(runtime_api_key.strip())
        and not runtime.get("credential_pool_entry_id")
    )
    if api_key or inherited_explicit_key:
        # Explicit credentials are keyed by their own fingerprint, not the pool's.
        key = api_key if api_key else str(runtime_api_key).strip()
        denied = policy.usage_admission_credential_denied("openai-codex", key)
        return CodexAdmission(denied=denied, pool_hint="", api_key=key if inherited_explicit_key else "",
                               base_url=(str(runtime.get("base_url") or "").strip() if inherited_explicit_key and not base_url else ""))
    pool_present, entry = aux._select_pool_entry("openai-codex")
    if pool_present and entry is None and policy.usage_admission_policy_denied("openai-codex"):
        # An explicit Codex route stops here; an ``auto`` route still owns a fallback ladder.
        return CodexAdmission(denied=aux._normalize_aux_provider(provider) != "auto", evict=True)
    if entry is None:
        return CodexAdmission()
    entry_id = str(getattr(entry, "id", "") or "").strip()
    key = aux._pool_runtime_api_key(entry)
    return CodexAdmission(
        pool_hint=f"openai-codex:{entry_id}" if entry_id else None, api_key=key,
        base_url=aux._pool_runtime_base_url(entry, aux._CODEX_AUX_BASE_URL) if key else "", provider="openai-codex",
    )


def recover_pool_after_auth_error(pool: Any, provider: str, failed_api_key: Optional[str]) -> Optional[bool]:
    """Force-refresh the pool entry behind a 401 and apply admission to the new token.

    ``True`` when a usable credential is ready (cached clients evicted), ``False`` when the refreshed
    credential is over its ceiling and no eligible sibling exists, ``None`` when the refresh produced
    nothing (the caller rotates as for any other dead credential)."""
    from agent import auxiliary_client as aux  # auxiliary_client imports this module

    refresh_matching = getattr(pool, "try_refresh_matching_with_admission_status", None)
    refresh_current = getattr(pool, "try_refresh_current_with_admission_status", None)
    outcome = None
    if failed_api_key and callable(refresh_matching):
        outcome = refresh_matching(api_key_hint=failed_api_key)
    elif callable(refresh_current):
        outcome = refresh_current()
    if isinstance(outcome, tuple) and len(outcome) == 2:
        refreshed, policy_denied_id = outcome
    else:
        # Backward compatibility for third-party/lightweight pool adapters without admission status.
        refreshed, policy_denied_id = pool.try_refresh_current(), None
    if refreshed is not None:
        aux._evict_cached_clients(provider)
        return True
    if policy_denied_id is None:
        return None
    # The forced refresh succeeded but the stable credential is over its configured admission ceiling.
    # Keep policy state separate from credential health and retry only if an eligible sibling exists.
    if pool.select() is None:
        return False
    aux._evict_cached_clients(provider)
    return True


def effective_aux_provider(cached: Any) -> str:
    """The provider an ``auto`` route's cached client resolved to (it is keyed under "auto")."""
    eff = str(getattr(cached, "_hermes_aux_effective_provider", "") or "")
    if not eff:
        eff = str(getattr(getattr(cached, "_real_client", None), "_hermes_aux_effective_provider", "") or "")
    return eff


def codex_credential_and_base(
    model: str, explicit_key: Optional[str], explicit_base_url: Optional[str],
) -> tuple[Optional[str], str]:
    """The Codex token and base URL: the caller's explicit credential when given, else the pool's pick."""
    from agent import auxiliary_client as aux  # auxiliary_client imports this module

    token = str(explicit_key or "").strip()
    if not token:
        return aux._resolve_codex_credential_and_base(model)
    return token, str(explicit_base_url or "").strip() or aux._codex_base_url_override() or aux._CODEX_AUX_BASE_URL


def drop_denied_cached_client(
    admission: CodexAdmission, provider: str, *, async_mode: bool, base_url: Optional[str],
    api_key: Optional[str], api_mode: Optional[str], main_runtime: Any, is_vision: bool,
    task: Optional[str], model: Optional[str],
) -> None:
    """Evict (and close) the cached client a denied admission verdict must not reuse."""
    from agent import auxiliary_client as aux  # auxiliary_client imports this module

    if admission.evict:
        aux._evict_cached_clients("openai-codex")
    cache_key = aux._client_cache_key(
        provider, async_mode=async_mode, base_url=admission.base_url or base_url,
        api_key=admission.api_key or api_key, api_mode=api_mode, main_runtime=main_runtime,
        is_vision=is_vision, task=task, model=model, pool_hint_override=admission.pool_hint,
    )
    with aux._client_cache_lock:
        denied_entry = aux._client_cache.pop(cache_key, None)
    if denied_entry is not None:
        aux._close_cached_client(denied_entry[0])

"""Application acquisition and scoped observation caches for model pricing.

OpenRouter-compatible ``/v1/models`` pricing fetch with a per-endpoint/per-credential cache,
Nous Portal sale chrome and org-policy filtering, and the Vercel AI Gateway / Novita / Fireworks /
DeepInfra pricing adapters. Pure billing interpretation and catalogue policy are model-owned.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Optional
from hermes_constants import hermes_home_key
from models.metadata.pricing import _pricing_entry, _per_token
from models.metadata.reasoning import _seed_reasoning_caps, configure_reasoning_metadata_sources


# Cache: maps model_id → {"prompt": str, "completion": str} per endpoint
_pricing_cache: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
# (profile key, provider) → endpoint cache key last fetched, so cached_only reads find the right entry.
_pricing_provider_cache_keys: dict[tuple[str, str], str] = {}

# A failed fetch caches its empty result too, so an unreachable endpoint isn't re-dialed on every
# call — but only until this deadline. Cached forever, one blip at startup would mean no live model
# discovery for the life of a process that runs for weeks (gateway, desktop backend), silently:
# every caller falls back to a curated list meanwhile.
_FAILED_CATALOG_TTL_SECONDS = 120.0

_pricing_cache_retry_after: dict[tuple[str, str], float] = {}


def _pricing_scope_key(cache_key: str) -> tuple[str, str]:
    return (hermes_home_key(), cache_key)


def _cached_catalog(cache_key: str) -> Optional[dict[str, dict[str, Any]]]:
    """The cached catalog for *cache_key*, or None to go fetch it."""
    cache_key = _pricing_scope_key(cache_key)
    cached = _pricing_cache.get(cache_key)
    if cached is None:
        return None
    retry_after = _pricing_cache_retry_after.get(cache_key)
    if retry_after is not None and time.monotonic() >= retry_after:
        _pricing_cache.pop(cache_key, None)
        _pricing_cache_retry_after.pop(cache_key, None)
        return None
    return cached


def _cache_catalog(
    cache_key: str,
    result: dict[str, dict[str, Any]],
    ttl_seconds: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    """Cache a catalog result, giving an empty one an expiry. *ttl_seconds* expires a non-empty
    result too — only for a catalog whose contents depend on server-side state the client cannot
    observe (an org's model policy can change while a long-lived process holds the entry)."""
    cache_key = _pricing_scope_key(cache_key)
    _pricing_cache[cache_key] = result
    if not result:
        _pricing_cache_retry_after[cache_key] = time.monotonic() + _FAILED_CATALOG_TTL_SECONDS
    elif ttl_seconds:
        _pricing_cache_retry_after[cache_key] = time.monotonic() + ttl_seconds
    else:
        _pricing_cache_retry_after.pop(cache_key, None)
    return result


# NUL cannot appear in a URL, so this cannot collide with a real base URL.
_PRICING_AUTH_KEY_PREFIX = "\x00auth:"


def _pricing_auth_fingerprint(api_key: str | None) -> str:
    """Cache-key suffix identifying the credential a catalog was read with: a governed endpoint
    answers each token with the catalog its org may reach, so two credentials cannot share an
    entry. blake2b for fingerprinting only (same rationale as ``_custom_endpoint_fingerprint``)."""
    if not api_key:
        return ""
    import hashlib

    digest = hashlib.blake2b(api_key.encode("utf-8", errors="replace"), digest_size=8)
    return _PRICING_AUTH_KEY_PREFIX + digest.hexdigest()


def peek_cached_pricing(base_url: str) -> dict[str, dict[str, Any]]:
    """Pricing already cached for *base_url* (with or without ``/v1``), or ``{}``; never fetches.
    Prefers an authenticated catalog, scanning newest first (callers hold no credential) and
    skipping expired entries so a rotated credential does not answer from its predecessor's."""
    root = _strip_v1((base_url or "").rstrip("/"))
    authed_prefix = root + _PRICING_AUTH_KEY_PREFIX
    for home, key in reversed(list(_pricing_cache)):
        if home == hermes_home_key() and key.startswith(authed_prefix):
            cached = _cached_catalog(key)
            if cached:
                return cached
    return _cached_catalog(root) or {}


def pricing_fetch_suppressed(base_url: str) -> bool:
    """A fetch for *base_url* failed recently and its empty result is still cached, so re-fetching now
    returns that ``{}`` without dialing (see ``_cache_catalog``). Lets a caller that would otherwise
    start a background refresh on a cold peek skip it for the rest of the failure window."""
    root = _strip_v1((base_url or "").rstrip("/"))
    now = time.monotonic()
    return any(
        _pricing_cache.get((home, key)) == {} and _pricing_cache_retry_after.get((home, key), 0.0) > now
        for home, key in list(_pricing_cache)
        if home == hermes_home_key() and (key == root or key.startswith(root + _PRICING_AUTH_KEY_PREFIX))
    )


def _strip_v1(url: str) -> str:
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


def _get_json(url: str, headers: dict[str, str], timeout: float, opener=None) -> Optional[dict]:
    """GET *url* as JSON via the origin's catalog opener (or *opener*); None on any failure."""
    from hermes_cli.models import _urlopen_model_catalog_request

    try:
        req = urllib.request.Request(url, headers=headers)
        with (opener or _urlopen_model_catalog_request)(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _catalog_items(payload: dict) -> list[dict]:
    return [item for item in payload.get("data", []) if isinstance(item, dict)]


def fetch_models_with_pricing(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api",
    timeout: float = 8.0,
    *,
    force_refresh: bool = False,
    include_sale_original: bool = False,
    cache_ttl_seconds: Optional[float] = None,
) -> dict[str, dict[str, Any]]:
    """Fetch ``/v1/models`` (any OpenRouter-compatible endpoint) → ``{model_id: {prompt, completion,
    ...}}``, cached per *base_url* and per credential so one caller's catalog never answers
    another's read. *include_sale_original* (Nous Portal only) copies the gateway's pre-discount
    ``pricing.original`` rates through as a nested ``original`` dict for sale chrome."""
    from hermes_cli.models import _HERMES_USER_AGENT
    url_root = (base_url or "").rstrip("/")
    cache_key = url_root + _pricing_auth_fingerprint(api_key)
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    url = url_root + "/v1/models"
    headers = {"Accept": "application/json", "User-Agent": _HERMES_USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = _get_json(url, headers, timeout)
    if payload is None:
        return _cache_catalog(cache_key, {})

    # Same document the reasoning-capability fetch would pull — mirror it so a later hot-path
    # lookup (and the next process) has an answer without its own round-trip.
    _seed_reasoning_caps(url, payload.get("data"))

    result: dict[str, dict[str, Any]] = {}
    for item in payload.get("data", []):
        mid, pricing = item.get("id"), item.get("pricing")
        if mid and isinstance(pricing, dict):
            entry = _pricing_entry(pricing)
            # Sale chrome is Nous Portal-only; never copy pricing.original for other catalogs.
            original = pricing.get("original") if include_sale_original else None
            if isinstance(original, dict):
                orig_entry = {key: str(original[key]) for key in ("prompt", "completion", "input_cache_read", "input_cache_write")
                              if original.get(key) not in (None, "")}
                if orig_entry.get("prompt") or orig_entry.get("completion"):
                    entry["original"] = orig_entry
            # Nous Portal-only: the gateway bills this row to a subscription the account holds, not to credits.
            if include_sale_original and item.get("billing_mode") == "subscription":
                entry["billing_mode"] = "subscription"
            result[mid] = entry

    return _cache_catalog(cache_key, result, cache_ttl_seconds)


def fetch_ai_gateway_pricing(timeout: float = 8.0, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Vercel AI Gateway /v1/models pricing, translating its ``input`` / ``output`` field names to
    the picker's ``prompt`` / ``completion`` (cache read/write names already match)."""
    from hermes_constants import AI_GATEWAY_BASE_URL

    cache_key = AI_GATEWAY_BASE_URL.rstrip("/")
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    payload = _get_json(f"{cache_key}/models", {"Accept": "application/json"}, timeout, opener=urllib.request.urlopen)
    if payload is None:
        return _cache_catalog(cache_key, {})

    result: dict[str, dict[str, str]] = {}
    for item in _catalog_items(payload):
        mid, pricing = item.get("id"), item.get("pricing")
        if mid and isinstance(pricing, dict):
            result[mid] = _pricing_entry(pricing, "input", "output")
    return _cache_catalog(cache_key, result)


def _resolve_openrouter_api_key() -> str:
    """Best-effort OpenRouter API key for pricing fetch."""
    return os.getenv("OPENROUTER_API_KEY", "").strip()


_DEFAULT_NOUS_INFERENCE_BASE = "https://inference-api.nousresearch.com"


def _resolve_nous_pricing_credentials() -> tuple[str, str]:
    """``(api_key, base_url)`` for Nous Portal pricing; base_url is the bare origin (no ``/v1``).
    Precedence mirrors runtime credential resolution: ``NOUS_INFERENCE_BASE_URL`` (staging /
    preview) → resolved credential ``base_url`` → production default. Without the override a
    staging profile's sale ``pricing.original`` would never reach the pickers."""
    try:
        from hermes_cli.auth import _nous_inference_env_override

        env_base = _nous_inference_env_override()
    except Exception:
        env_base = None
    api_key = creds_base = ""
    try:
        from hermes_cli.auth import resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
        if creds:
            api_key = creds.get("api_key", "") or ""
            creds_base = (creds.get("base_url", "") or "").strip()
    except Exception:
        pass
    base_url = (env_base or creds_base or _DEFAULT_NOUS_INFERENCE_BASE).rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return (api_key, base_url)


def _nous_reasoning_catalog_url() -> str:
    """Inject only the resolved endpoint name; credentials remain pricing-owned."""
    return f"{_resolve_nous_pricing_credentials()[1]}/v1/models"


configure_reasoning_metadata_sources(nous_url=_nous_reasoning_catalog_url)


# How long a Nous catalog stays trusted. Its contents depend on the org's policy, which an admin
# can change at any time and the client cannot observe, so a long-lived process must re-ask.
# Other providers' catalogs carry no such state and keep the default no-expiry caching.
_NOUS_CATALOG_TTL_SECONDS = 300.0


def _fetch_nous_pricing(api_key: str, base_url: str, *, force_refresh: bool) -> dict[str, dict[str, Any]]:
    """Shared by pricing and policy lookups so both read one cache entry."""
    return fetch_models_with_pricing(
        api_key=api_key,
        base_url=base_url,
        force_refresh=force_refresh,
        include_sale_original=True,  # Sale chrome (pricing.original) is Nous Portal-only.
        cache_ttl_seconds=_NOUS_CATALOG_TTL_SECONDS,
    )


def nous_policy_allowed_ids(*, force_refresh: bool = False) -> Optional[set[str]]:
    """The Nous model ids the caller's org may reach (keys of an authenticated ``GET /v1/models``,
    which omits policy-blocked rows), or ``None`` to not filter: no policy (or a token too old to
    say), an anonymous read (unfiltered catalog), or an empty read (a fetch failure, not an org
    that may reach nothing)."""
    try:
        from hermes_cli.nous_account import nous_policy_present

        if nous_policy_present() is not True:
            return None
    except Exception:
        return None

    api_key, base_url = _resolve_nous_pricing_credentials()
    if not api_key or not base_url:
        return None
    return set(_fetch_nous_pricing(api_key, base_url, force_refresh=force_refresh)) or None


# Past this size an allowed set reads as a whole catalog rather than an allowlist, and is not
# worth showing in place of an empty picker.


def _remember_provider_cache_key(provider: str, cache_key: str) -> None:
    _pricing_provider_cache_keys[(hermes_home_key(), provider)] = cache_key


def _fetch_openrouter_pricing(*, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    _remember_provider_cache_key("openrouter", _OPENROUTER_PRICING_BASE)
    return fetch_models_with_pricing(
        api_key=_resolve_openrouter_api_key(),
        base_url=_OPENROUTER_PRICING_BASE,
        force_refresh=force_refresh,
    )


def _fetch_ai_gateway_pricing_for_provider(*, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    _remember_provider_cache_key("ai-gateway", _ai_gateway_pricing_scope())
    return fetch_ai_gateway_pricing(force_refresh=force_refresh)


def _fetch_novita_pricing_for_provider(*, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    _remember_provider_cache_key("novita", _novita_pricing_scope())
    return _fetch_novita_pricing(force_refresh=force_refresh)


def _fetch_fireworks_pricing_for_provider(*, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    _remember_provider_cache_key("fireworks", _FIREWORKS_PRICING_KEY)
    return _fireworks_pricing_from_models_dev(force_refresh=force_refresh)


def _fetch_nous_pricing_for_provider(*, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    api_key, base_url = _resolve_nous_pricing_credentials()
    if not base_url:
        return {}
    _remember_provider_cache_key("nous", base_url.rstrip("/"))
    return _fetch_nous_pricing(api_key, base_url, force_refresh=force_refresh)


_OPENROUTER_PRICING_BASE = "https://openrouter.ai/api"
_FIREWORKS_PRICING_KEY = "models.dev/fireworks"


def _ai_gateway_pricing_scope() -> str:
    from hermes_constants import AI_GATEWAY_BASE_URL
    return AI_GATEWAY_BASE_URL.rstrip("/")


def _novita_pricing_scope() -> str:
    return (os.getenv("NOVITA_BASE_URL", "").strip() or "https://api.novita.ai/openai/v1").rstrip("/")


def get_cached_nous_inference_base_url() -> str:
    """The profile's persisted Nous endpoint (bare origin, no ``/v1``) without refreshing auth."""
    try:
        from hermes_cli.auth import (
            _load_auth_store, _load_provider_state, _optional_base_url, _validate_nous_inference_url_from_network,
        )

        state = _load_provider_state(_load_auth_store(), "nous") or {}
        url = _validate_nous_inference_url_from_network(_optional_base_url(state.get("inference_base_url"))) or ""
        return url.rstrip("/").removesuffix("/v1")
    except Exception:
        return ""


# Static endpoint identity per provider; dynamic ones (deepinfra, nous) are resolved in pricing_cache_scope.
_STATIC_PRICING_SCOPES = {
    "openrouter": lambda: _OPENROUTER_PRICING_BASE,
    "ai-gateway": _ai_gateway_pricing_scope,
    "novita": _novita_pricing_scope,
    "fireworks": lambda: _FIREWORKS_PRICING_KEY,
}


def pricing_cache_scope(provider: str, *, current_provider: str = "", current_base_url: str = "") -> str:
    """The current endpoint identity a provider's pricing cache is keyed on. Resolves local configuration
    only, never fetches: picker prewarm single-flight uses it so an endpoint rotation can start a new
    worker while the previous endpoint is still slow or unreachable."""
    from providers import normalize_provider

    normalized = normalize_provider(provider)
    static = _STATIC_PRICING_SCOPES.get(normalized)
    if static:
        return static()
    if normalized == "deepinfra":
        from application_deepinfra_catalog import deepinfra_base_url

        return deepinfra_base_url()
    if normalized == "nous":
        try:
            from hermes_cli.auth import _nous_inference_env_override

            env_base = _nous_inference_env_override()
        except Exception:
            env_base = None
        if env_base:
            return env_base.rstrip("/").removesuffix("/v1")
        if normalize_provider(current_provider) == "nous" and current_base_url:
            return current_base_url.rstrip("/").removesuffix("/v1")
        persisted_base = get_cached_nous_inference_base_url()
        if persisted_base:
            return persisted_base
        return _pricing_provider_cache_keys.get((hermes_home_key(), normalized), _DEFAULT_NOUS_INFERENCE_BASE)
    return ""


def _cached_only_pricing(normalized: str) -> dict[str, dict[str, str]]:
    """Process-resident pricing for *normalized* without any provider I/O."""
    if normalized == "deepinfra":
        return _fetch_deepinfra_pricing(cached_only=True)
    cache_key = _pricing_provider_cache_keys.get((hermes_home_key(), normalized))
    if cache_key is None and normalized in ("openrouter", "ai-gateway", "fireworks"):
        cache_key = _STATIC_PRICING_SCOPES[normalized]()
    return (_cached_catalog(cache_key) or {}) if cache_key else {}


def get_pricing_for_provider(
    provider: str, *, force_refresh: bool = False, cached_only: bool = False
) -> dict[str, dict[str, str]]:
    """Return live pricing for providers that support it (openrouter, nous, ai-gateway, novita,
    deepinfra, fireworks); ``{}`` for everything else. ``cached_only`` never starts provider I/O:
    normal picker opens use it so cold endpoints cannot hold the response path, while a background
    prewarm fills the same caches for later opens."""
    from providers import normalize_provider

    normalized = normalize_provider(provider)
    if cached_only:
        return _cached_only_pricing(normalized)
    fetcher = _PRICING_FETCHERS.get(normalized)
    return fetcher(force_refresh=force_refresh) if fetcher else {}


def _fireworks_pricing_from_models_dev(*, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """Fireworks picker pricing from the models.dev registry cache (``fetch_models_dev()`` keeps a
    shared in-memory + disk cache, 1h TTL) — a pure dict transform, no per-render network call."""
    cache_key = "models.dev/fireworks"
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    result: dict[str, dict[str, str]] = {}
    try:
        from agent.models_dev import _get_provider_models

        for mid, entry in (_get_provider_models("fireworks") or {}).items():
            cost = entry.get("cost") if isinstance(entry, dict) else None
            if not isinstance(cost, dict):
                continue
            inp, out = cost.get("input"), cost.get("output")
            if inp is None and out is None:
                continue
            row = {"prompt": _per_token(inp or 0), "completion": _per_token(out or 0)}
            if cost.get("cache_read"):
                row["input_cache_read"] = _per_token(cost["cache_read"])
            result[str(mid)] = row
    except Exception:
        result = {}

    return _cache_catalog(cache_key, result)


def _fetch_novita_pricing(timeout: float = 8.0, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """NovitaAI /v1/models pricing (per-million prices in units of 0.0001 USD → per-token strings),
    cached on the resolved base URL so menu renders don't re-hit the network."""
    from hermes_cli.models import _HERMES_USER_AGENT
    api_key = os.getenv("NOVITA_API_KEY", "").strip()
    if not api_key:
        return {}

    cache_key = (os.getenv("NOVITA_BASE_URL", "").strip() or "https://api.novita.ai/openai/v1").rstrip("/")
    if not force_refresh:
        cached = _cached_catalog(cache_key)
        if cached is not None:
            return cached

    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json", "User-Agent": _HERMES_USER_AGENT}
    payload = _get_json(cache_key + "/models", headers, timeout)
    if payload is None:
        return _cache_catalog(cache_key, {})

    result: dict[str, dict[str, str]] = {}
    for item in _catalog_items(payload):
        mid = item.get("id")
        inp, out = item.get("input_token_price_per_m"), item.get("output_token_price_per_m")
        if not mid or (inp is None and out is None):
            continue
        result[str(mid)] = {
            "prompt": str(float(inp or 0) / 10_000 / 1_000_000),
            "completion": str(float(out or 0) / 10_000 / 1_000_000),
        }

    return _cache_catalog(cache_key, result)


def _fetch_deepinfra_pricing(
    timeout: float = 5.0, *, force_refresh: bool = False, cached_only: bool = False
) -> dict[str, dict[str, str]]:
    """DeepInfra pricing projected from the same canonical tagged catalogue as every surface."""
    from application_deepinfra_catalog import models_by_tag

    items = models_by_tag("chat", timeout=timeout, force_refresh=force_refresh, cached_only=cached_only)
    result: dict[str, dict[str, str]] = {}
    for item in items or []:
        metadata = item.get("metadata") or {}
        pricing = metadata.get("pricing") if isinstance(metadata, dict) else None
        if not isinstance(pricing, dict):
            continue
        entry = {
            ours: _per_token(pricing[theirs])
            for theirs, ours in (("input_tokens", "prompt"), ("output_tokens", "completion"), ("cache_read_tokens", "input_cache_read"))
            if pricing.get(theirs) is not None
        }
        if entry:
            result[item["id"]] = entry
    return result


_PRICING_FETCHERS = {
    "openrouter": _fetch_openrouter_pricing,
    "ai-gateway": _fetch_ai_gateway_pricing_for_provider,
    "novita": _fetch_novita_pricing_for_provider,
    "deepinfra": _fetch_deepinfra_pricing,
    "fireworks": _fetch_fireworks_pricing_for_provider,
    "nous": _fetch_nous_pricing_for_provider,
}

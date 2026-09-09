"""Live model pricing.

OpenRouter-compatible ``/v1/models`` pricing fetch with a per-endpoint/per-credential cache,
Nous Portal sale chrome and org-policy filtering, and the Vercel AI Gateway / Novita / Fireworks /
DeepInfra pricing adapters. Split out of ``hermes_cli.models``; helpers still defined there are
looked up on ``hermes_cli.models`` at call time so ``patch("hermes_cli.models.<name>")`` mocks keep
intercepting.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Optional
from hermes_cli.models_reasoning_caps import _seed_reasoning_caps


# Cache: maps model_id → {"prompt": str, "completion": str} per endpoint
_pricing_cache: dict[str, dict[str, dict[str, str]]] = {}
# (profile key, provider) → endpoint cache key last fetched, so cached_only reads find the right entry.
_pricing_provider_cache_keys: dict[tuple[str, str], str] = {}

# A failed fetch caches its empty result too, so an unreachable endpoint isn't re-dialed on every
# call — but only until this deadline. Cached forever, one blip at startup would mean no live model
# discovery for the life of a process that runs for weeks (gateway, desktop backend), silently:
# every caller falls back to a curated list meanwhile.
_FAILED_CATALOG_TTL_SECONDS = 120.0

_pricing_cache_retry_after: dict[str, float] = {}


def _cached_catalog(cache_key: str) -> Optional[dict[str, dict[str, Any]]]:
    """The cached catalog for *cache_key*, or None to go fetch it."""
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
    for key in reversed(list(_pricing_cache)):
        if key.startswith(authed_prefix):
            cached = _cached_catalog(key)
            if cached:
                return cached
    return _cached_catalog(root) or {}


def _strip_v1(url: str) -> str:
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


def _format_price_per_mtok(per_token_str: str) -> str:
    """Per-token price string → $/Mtok string. Always 2 decimals so right-justified prices align;
    sub-cent prices (deep-discount cache-hit promos) widen precision until the value shows, keep
    one extra digit and trim trailing zeros instead of collapsing to "$0.00"."""
    try:
        val = float(per_token_str)
    except (TypeError, ValueError):
        return "?"
    if val == 0:
        return "free"
    per_m = val * 1_000_000
    text = f"{per_m:.2f}"
    if per_m < 0.01:
        prec = 3
        while prec < 12 and round(per_m, prec) == 0:
            prec += 1
        text = f"{per_m:.{min(prec + 1, 12)}f}".rstrip("0").rstrip(".")
    return f"${text}"


def _price_float(raw: Any, *, positive: bool) -> float | None:
    """*raw* as a finite float (> 0, or >= 0 when not *positive*); None when unset/invalid/NaN."""
    if raw in (None, ""):
        return None
    try:
        n = float(raw)
    except (TypeError, ValueError):
        return None
    if n != n or (n <= 0 if positive else n < 0):
        return None
    return n


def _sale_pct(current: Any, original: Any) -> int | None:
    """Percent discount when *current* is strictly below *original* (both positive finite)."""
    cur, orig = _price_float(current, positive=True), _price_float(original, positive=True)
    if cur is None or orig is None or cur >= orig:
        return None
    return int(round((1.0 - (cur / orig)) * 100))


def compute_sale_discount(prompt: str, completion: str, original: Any) -> tuple[int, str, str] | None:
    """Sale chrome from gateway ``pricing.original`` (Nous Portal only; callers gate on the provider
    and opted in via ``include_sale_original=True``): ``(discount_percent, was_prompt_raw,
    was_completion_raw)`` when ``original`` is a dict and the current prompt (fallback: completion)
    rate is strictly below the original. Free / $0 models get a flat 100% off, with "was" prices
    only when the gateway served an original (a natively-free stealth model gets bare "-100%")."""
    orig_dict = original if isinstance(original, dict) else {}
    was_prompt = orig_dict.get("prompt")
    was_completion = orig_dict.get("completion")
    was_prompt_str = str(was_prompt) if was_prompt not in (None, "") else ""
    was_completion_str = str(was_completion) if was_completion not in (None, "") else ""

    if _price_float(prompt, positive=False) == 0 and _price_float(completion, positive=False) in (0, None):
        return (100, was_prompt_str, was_completion_str)

    if not isinstance(original, dict) or (not was_prompt_str and not was_completion_str):
        return None

    pct = _sale_pct(prompt, was_prompt)
    if pct is not None:
        return (pct, was_prompt_str, was_completion_str) if pct >= 1 else None
    pct = _sale_pct(completion, was_completion)
    if pct is not None:
        return (pct, was_prompt_str, was_completion_str) if pct >= 1 else None
    return None


def _get_json(url: str, headers: dict[str, str], timeout: float, opener=None) -> Optional[dict]:
    """GET *url* as JSON via the origin's catalog opener (or *opener*); None on any failure."""
    from hermes_cli.models import _urlopen_model_catalog_request

    try:
        req = urllib.request.Request(url, headers=headers)
        with (opener or _urlopen_model_catalog_request)(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _pricing_entry(pricing: dict, prompt_key: str = "prompt", completion_key: str = "completion") -> dict[str, Any]:
    """Picker-shape ``{prompt, completion[, input_cache_read, input_cache_write]}`` from a catalog
    ``pricing`` block whose cache fields already use the hermes names."""
    entry: dict[str, Any] = {
        "prompt": str(pricing.get(prompt_key, "")),
        "completion": str(pricing.get(completion_key, "")),
    }
    for key in ("input_cache_read", "input_cache_write"):
        if pricing.get(key):
            entry[key] = str(pricing[key])
    return entry


def _per_token(per_mtok: Any) -> str:
    """$/MTok → the per-token price string the picker expects."""
    return str(float(per_mtok) / 1_000_000)


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
_NOUS_POLICY_APPEND_MAX = 64


def restrict_to_nous_policy(
    model_ids: list[str],
    allowed: Optional[set[str]],
    *,
    rescue_empty: bool = False,
) -> list[str]:
    """*model_ids* narrowed to *allowed*, preserving order. A ``:free`` sibling is kept when its
    base model is reachable (the gateway admits a row when any requestable id passes); over-listing
    costs a 403 from the authoritative gate, hiding a servable row is unrecoverable client-side.
    *rescue_empty*: an allowlist naming only models the curated manifest lacks would leave an empty
    picker — worse than no filter — so return the allowlist itself. Opt-in per list: an already-
    empty list (a paid tier's gated models) means "nothing to gate", not "nothing survived"."""
    if not allowed:
        return list(model_ids)
    kept = [mid for mid in model_ids if mid in allowed or mid.split(":", 1)[0] in allowed]
    if rescue_empty and not kept and len(allowed) <= _NOUS_POLICY_APPEND_MAX:
        return sorted(allowed)
    return kept


def _remember_provider_cache_key(provider: str, cache_key: str) -> None:
    from hermes_cli.models import _pricing_profile_key
    _pricing_provider_cache_keys[(_pricing_profile_key(), provider)] = cache_key


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
    from hermes_cli.models import _deepinfra_catalog_url, _pricing_profile_key, normalize_provider
    normalized = normalize_provider(provider)
    static = _STATIC_PRICING_SCOPES.get(normalized)
    if static:
        return static()
    if normalized == "deepinfra":
        return _deepinfra_catalog_url()[0]
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
        return _pricing_provider_cache_keys.get((_pricing_profile_key(), normalized), _DEFAULT_NOUS_INFERENCE_BASE)
    return ""


def _cached_only_pricing(normalized: str) -> dict[str, dict[str, str]]:
    """Process-resident pricing for *normalized* without any provider I/O."""
    from hermes_cli.models import _deepinfra_catalog_cache, _deepinfra_catalog_url, _pricing_profile_key
    if normalized == "deepinfra":
        cache_key, _url = _deepinfra_catalog_url()
        return _fetch_deepinfra_pricing() if cache_key in _deepinfra_catalog_cache else {}
    cache_key = _pricing_provider_cache_keys.get((_pricing_profile_key(), normalized))
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
    from hermes_cli.models import normalize_provider
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


def _fetch_deepinfra_pricing(timeout: float = 5.0, *, force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """DeepInfra chat-model pricing: ``input_tokens`` / ``output_tokens`` / ``cache_read_tokens`` in
    $/MTok → per-token ``prompt`` / ``completion`` / ``input_cache_read`` (cached by the by-tag
    helper)."""
    from hermes_cli.models import _fetch_deepinfra_models_by_tag
    items = _fetch_deepinfra_models_by_tag("chat", timeout=timeout, force_refresh=force_refresh)
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

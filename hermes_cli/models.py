"""Provider/model catalogs: discovery, caching, and identity helpers.

Origin module; cohesive clusters live in siblings and are re-imported here so
``hermes_cli.models.<name>`` stays the stable import/monkeypatch surface:
``models_catalog_static`` (curated tables, provider registry, aliases), ``models_reasoning_caps``,
``models_local`` (Ollama / LM Studio), ``models_pricing``, ``models_validate``.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import threading
import urllib.parse
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeGuard

from hermes_cli import __version__ as _HERMES_VERSION
from hermes_cli.urllib_security import open_credentialed_url
from hermes_cli.models_catalog_static import (
    CANONICAL_PROVIDERS,
    OPENROUTER_MODELS,
    PREFERRED_SILENT_DEFAULT_MODEL,
    VERCEL_AI_GATEWAY_MODELS,
    _AGGREGATOR_PROVIDERS,
    _AZURE_FOUNDRY_RESPONSES_PREFIXES,
    _BORROWED_MODEL_PROVIDERS,
    _COPILOT_MODEL_ALIASES,
    _KEYLESS_STABLE_CACHE_PROVIDERS,
    _LIVE_FIRST_PICKER_PROVIDERS,
    _MODELS_DEV_PREFERRED,
    _OPENAI_FAST_MODE_PREFIXES,
    _OPENROUTER_VARIANT_SUFFIXES,
    _PROVIDER_ALIASES,
    _PROVIDER_CATALOG_DELEGATES,
    _PROVIDER_LABELS,
    _PROVIDER_MODELS,
    _PROVIDER_RETIRED_ALIASES,
    _SILENT_DEFAULT_PROVIDERS,
    _xai_finalize_catalog)
from hermes_cli.models_reasoning_caps import (
    _OPENROUTER_CATALOG_URL,
    _seed_reasoning_caps)
from hermes_cli.models_local import (
    _OLLAMA_LOCAL_MODELS_CACHE,
    _OLLAMA_LOCAL_MODELS_CACHE_TTL,
    _OLLAMA_LOCAL_PROBE_FAILURE_CACHE,
    _OLLAMA_LOCAL_PROBE_REACHABLE,
    _get_ollama_base_url,
    _get_ollama_native_headers,
    _ollama_local_catalog,
    _ollama_probe_cache_key,
    _root_for_ollama_native_api,
    fetch_ollama_cloud_models)

logger = logging.getLogger(__name__)

# Identify ourselves so endpoints fronted by Cloudflare's Browser Integrity
# Check (error 1010) don't reject the default ``Python-urllib/*`` signature.
_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"

COPILOT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_MODELS_URL = f"{COPILOT_BASE_URL}/models"
COPILOT_EDITOR_VERSION = "vscode/1.104.1"
COPILOT_REASONING_EFFORTS_GPT5 = ["minimal", "low", "medium", "high"]
COPILOT_REASONING_EFFORTS_O_SERIES = ["low", "medium", "high"]

def _urlopen_model_catalog_request(req: urllib.request.Request, *, timeout: float, ssl_context=None):
    """Open catalog requests without forwarding headers across origins."""
    return open_credentialed_url(req, timeout=timeout, ssl_context=ssl_context)


def _get_json(
    url: str, *, timeout: float, headers: Optional[dict[str, str]] = None, opener=None, **open_kwargs: Any
) -> Any:
    """GET ``url`` and parse the JSON body. ``opener`` defaults to the catalog opener (resolved at
    call time so monkeypatching ``_urlopen_model_catalog_request`` still applies). Raises on failure."""
    req = urllib.request.Request(url, headers=headers or {})
    with (opener or _urlopen_model_catalog_request)(req, timeout=timeout, **open_kwargs) as resp:
        return json.loads(resp.read().decode())


def _read_json_cache(path: Path, *, errors=Exception) -> Optional[dict]:
    """Load a JSON-object cache file; None when missing, unreadable, or not a dict."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except errors:
        return None
    return data if isinstance(data, dict) else None


def _write_json_cache(path: Path, data: Any, **dump_kwargs: Any) -> None:
    """Atomically persist a cache file (creating parents). Raises on failure — callers decide
    whether a failed cache write is worth logging."""
    from utils import atomic_json_write

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(path, data, **dump_kwargs)


def _merge_unique(primary: list[str], secondary: list[str], key=lambda m: str(m).lower()) -> list[str]:
    """``primary`` verbatim, then ``secondary`` entries whose ``key`` is new (deduped as it goes)."""
    merged, seen = list(primary), {key(m) for m in primary}
    for m in secondary:
        k = key(m)
        if k not in seen:
            seen.add(k)
            merged.append(m)
    return merged


def _custom_provider_ssl_context(base_url: str):
    """``ssl.SSLContext`` honoring a custom provider's ``ssl_ca_cert`` / ``ssl_verify`` (mirrors the
    httpx TLS resolution), or None so the urllib ``/models`` probe keeps the default policy."""
    if not base_url:
        return None
    try:
        from hermes_cli.config import get_custom_provider_tls_settings

        tls = get_custom_provider_tls_settings(base_url)
        if not tls:
            return None
        import ssl

        if tls.get("ssl_verify") is False:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        ca = tls.get("ssl_ca_cert")
        if isinstance(ca, str) and ca and os.path.isfile(ca):
            return ssl.create_default_context(cafile=ca)
    except Exception:
        return None  # never break discovery on a TLS-config lookup
    return None


# Process-lifetime picker lists refreshed from the live catalogs (see fetch_*_models).
_openrouter_catalog_cache: list[tuple[str, str]] | None = None
_ai_gateway_catalog_cache: list[tuple[str, str]] | None = None


# ---------------------------------------------------------------------------
# Nous Portal free-model helpers — the Portal models endpoint is the source of truth for what is
# offered (free or paid); we surface it as-is, no local allowlist filtering.
# ---------------------------------------------------------------------------


def _zero_priced(pricing: Any, keys: tuple[str, str], default: str) -> bool:
    """True when both pricing fields parse to 0 (missing fields read as ``default``)."""
    if not isinstance(pricing, dict):
        return False
    try:
        return all(float(pricing.get(k, default)) == 0 for k in keys)
    except (TypeError, ValueError):
        return False


def _is_model_free(model_id: str, pricing: dict[str, dict[str, str]]) -> bool:
    """Return True if *model_id* has zero-cost prompt AND completion pricing."""
    return bool(pricing.get(model_id)) and _zero_priced(pricing.get(model_id), ("prompt", "completion"), "1")


def partition_nous_models_by_tier(
    model_ids: list[str], pricing: dict[str, dict[str, str]], free_tier: bool
) -> tuple[list[str], list[str]]:
    """Split Nous models into (selectable, unavailable): free-tier users may only select free models
    (paid ones are returned as unavailable, shown grayed out)."""
    if not free_tier or not pricing:  # no pricing → can't determine, show everything
        return (model_ids, [])
    selectable = [mid for mid in model_ids if _is_model_free(mid, pricing)]
    return (selectable, [mid for mid in model_ids if mid not in selectable])


def _union_with_portal_recommendations(
    tier_key: str, curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str,
    *, force_refresh: bool, synthesize_free_pricing: bool,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Append the Portal's ``<tier_key>`` recommendations missing from ``curated_ids``.

    Curated models show first, Portal-only picks follow. Failures (network, parse, missing field)
    silently return the inputs unchanged — never block the picker on a Portal-side hiccup.
    """
    try:
        payload = fetch_nous_recommended_models(portal_base_url, force_refresh=force_refresh)
    except Exception:
        payload = None
    block = payload.get(tier_key) if isinstance(payload, dict) else None
    entries = block if isinstance(block, list) else []
    portal_ids = [name for entry in entries if (name := _extract_model_name(entry))]
    if not portal_ids:
        return (list(curated_ids), dict(pricing))

    augmented_pricing = dict(pricing)
    if synthesize_free_pricing:
        for mid in portal_ids:
            augmented_pricing.setdefault(mid, {"prompt": "0", "completion": "0"})
    seen = set(curated_ids)
    return (list(curated_ids) + [mid for mid in portal_ids if mid not in seen], augmented_pricing)


def union_with_portal_free_recommendations(
    curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str = "", *,
    force_refresh: bool = False) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Curated list + pricing plus the Portal's ``freeRecommendedModels``; Portal-only free picks get a
    synthetic $0 pricing entry so tier partitioning sees them as free."""
    return _union_with_portal_recommendations(
        "freeRecommendedModels", curated_ids, pricing, portal_base_url,
        force_refresh=force_refresh, synthesize_free_pricing=True)


def union_with_portal_paid_recommendations(
    curated_ids: list[str], pricing: dict[str, dict[str, str]], portal_base_url: str = "", *,
    force_refresh: bool = False) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Curated list plus the Portal's ``paidRecommendedModels``; ``pricing`` is deliberately left untouched."""
    return _union_with_portal_recommendations(
        "paidRecommendedModels", curated_ids, pricing, portal_base_url,
        force_refresh=force_refresh, synthesize_free_pricing=False)


# Free-tier detection cache, per profile — short so an account upgrade shows within minutes.
_FREE_TIER_CACHE_TTL: int = 180  # seconds
_free_tier_cache: dict[str, tuple[bool, float]] = {}  # profile key -> (result, timestamp)


def _pricing_profile_key() -> str:
    """Stable profile identity for process-local pricing caches."""
    from hermes_constants import hermes_home_key

    return hermes_home_key()


def get_cached_nous_free_tier() -> Optional[bool]:
    """This profile's live cached entitlement, or ``None`` if unknown/expired."""
    cached = _free_tier_cache.get(_pricing_profile_key())
    if cached is None or time.monotonic() - cached[1] >= _FREE_TIER_CACHE_TTL:
        return None
    return cached[0]


def check_nous_free_tier(*, force_fresh: bool = False, cached_only: bool = False) -> bool:
    """True only when the Nous Portal user is KNOWN to be free-tier (unknown/error → False so this
    never blocks users). Cached ``_FREE_TIER_CACHE_TTL`` seconds so an upgrade shows within minutes.
    ``cached_only`` returns the live cached answer or the fail-open ``False`` without contacting Portal."""
    now = time.monotonic()
    profile_key = _pricing_profile_key()
    if not force_fresh:
        cached_result = get_cached_nous_free_tier()
        if cached_result is not None:
            return cached_result
    if cached_only:
        return False
    try:
        from hermes_cli.nous_account import get_nous_portal_account_info

        result = get_nous_portal_account_info(force_fresh=force_fresh).is_free_tier
    except Exception:
        result = False  # default to paid on error — don't block users
    _free_tier_cache[profile_key] = (result, now)
    return result


# ---------------------------------------------------------------------------
# Nous Portal recommended models — curated paid/free suggestions plus dedicated compaction (aux)
# and vision picks, TTL-cached per process. Fields read: {paid,free}RecommendedModels:
# [{modelName}], {paid,free}Recommended{Compaction,Vision}Model: {modelName} | null
# ---------------------------------------------------------------------------

NOUS_RECOMMENDED_MODELS_PATH = "/api/nous/recommended-models"
_NOUS_RECOMMENDED_CACHE_TTL: int = 600  # seconds (10 minutes)
# (result_dict, timestamp) keyed by portal_base_url so staging vs prod don't collide.
_nous_recommended_cache: dict[str, tuple[dict[str, Any], float]] = {}


def _nous_recommended_disk_path() -> "Path":
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "nous_recommended_cache.json"


def _read_nous_recommended_disk(base: str) -> dict[str, Any] | None:
    """Last-known-good payload for ``base`` from the per-base disk map
    ``{"<base>": {"data": {...}, "ts": <epoch>}}`` (staging and prod don't collide), or None."""
    blob = _read_json_cache(_nous_recommended_disk_path(), errors=(OSError, json.JSONDecodeError))
    entry = (blob or {}).get(base)
    data = entry.get("data") if isinstance(entry, dict) else None
    return data if isinstance(data, dict) and data else None


def _write_nous_recommended_disk(base: str, data: dict[str, Any]) -> None:
    """Merge ``data`` into the per-base disk map atomically; failures are debug-logged (the in-process
    cache still works)."""
    if not data:
        return
    path = _nous_recommended_disk_path()
    try:
        blob = _read_json_cache(path, errors=(OSError, json.JSONDecodeError)) or {}
        blob[base] = {"data": data, "ts": time.time()}
        _write_json_cache(path, blob, indent=2)
    except OSError as exc:
        logger.debug("nous recommended-models disk cache write failed: %s", exc)


def fetch_nous_recommended_models(
    portal_base_url: str = "", timeout: float = 5.0, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Fetch the Portal's public ``/api/nous/recommended-models`` payload (no auth).

    Cached per portal URL for ``_NOUS_RECOMMENDED_CACHE_TTL`` seconds in process (``force_refresh``
    bypasses); a successful fetch is also persisted as last-known-good on disk, which serves a live
    failure so a transient Portal hiccup doesn't drop the recommendations.
    """
    base = (portal_base_url or "https://portal.nousresearch.com").rstrip("/")
    now = time.monotonic()
    cached = _nous_recommended_cache.get(base)
    if not force_refresh and cached is not None and now - cached[1] < _NOUS_RECOMMENDED_CACHE_TTL:
        return cached[0]
    try:
        data = _get_json(
            f"{base}{NOUS_RECOMMENDED_MODELS_PATH}", timeout=timeout, headers={"Accept": "application/json"}
        )
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    if data:
        _write_nous_recommended_disk(base, data)
    else:
        data = _read_nous_recommended_disk(base) or data
    _nous_recommended_cache[base] = (data, now)
    return data


def _resolve_nous_portal_url() -> str:
    """Best-effort lookup of the Portal base URL the user is authed against."""
    try:
        from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL, get_provider_auth_state

        state = get_provider_auth_state("nous") or {}
        portal = str(state.get("portal_base_url") or "").strip()
        return (portal or str(DEFAULT_NOUS_PORTAL_URL)).rstrip("/")
    except Exception:
        return "https://portal.nousresearch.com"


def _extract_model_name(entry: Any) -> Optional[str]:
    """Pull the ``modelName`` field from a recommended-model entry, else None."""
    model_name = entry.get("modelName") if isinstance(entry, dict) else None
    return model_name.strip() if isinstance(model_name, str) and model_name.strip() else None


def get_nous_recommended_aux_model(
    *, vision: bool = False, free_tier: Optional[bool] = None, portal_base_url: str = "",
    force_refresh: bool = False) -> Optional[str]:
    """The Portal's recommended model for an auxiliary task: free tier → free pick only; paid tier →
    paid pick, falling back to the free one when the Portal returned ``null`` (staged rollouts)."""
    base = portal_base_url or _resolve_nous_portal_url()
    payload = fetch_nous_recommended_models(base, force_refresh=force_refresh)
    if not payload:
        return None
    if free_tier is None:
        try:
            free_tier = check_nous_free_tier()
        except Exception:
            free_tier = False  # assume paid on detection error — paid users see both fields anyway
    kind = "Vision" if vision else "Compaction"
    tiers = ("free",) if free_tier else ("paid", "free")
    return next((n for t in tiers if (n := _extract_model_name(payload.get(f"{t}Recommended{kind}Model")))), None)


def get_preferred_silent_default_model(provider: str = "openrouter") -> str:
    """Silent-default model id: the cached remote catalog's ``"default": true`` label (never hits the
    network — safe on hot paths), else :data:`PREFERRED_SILENT_DEFAULT_MODEL`."""
    try:
        from hermes_cli.model_catalog import get_default_model_from_cache
        labeled = get_default_model_from_cache(provider)
        if labeled:
            return labeled
    except Exception:
        pass
    return PREFERRED_SILENT_DEFAULT_MODEL


def pick_silent_default_model(model_ids: list[str], provider: str = "openrouter") -> str:
    """Catalog-labeled default when ``model_ids`` carries it, else the first entry, else "". Used by
    every surface that must choose a model without an interactive picker."""
    preferred = get_preferred_silent_default_model(provider)
    return preferred if preferred in model_ids else (model_ids[0] if model_ids else "")


def get_default_model_for_provider(provider: str) -> str:
    """Cost-safe default model for a provider, or "" — the NON-INTERACTIVE fallback when a provider
    is configured but no model was ever selected."""
    models = _PROVIDER_MODELS.get(provider, [])
    if provider in _SILENT_DEFAULT_PROVIDERS:
        preferred = get_preferred_silent_default_model(provider)
        # Trust the preferred default even without a static catalog (OpenRouter's picker list is
        # fetched live; its curated snapshot carries the default).
        if preferred and (preferred in models or not models):
            return preferred
    return models[0] if models else ""


def _openrouter_model_is_free(pricing: Any) -> bool:
    return _zero_priced(pricing, ("prompt", "completion"), "0")


def _openrouter_model_supports_tools(item: Any) -> bool:
    """True when ``supported_parameters`` advertises ``tools`` (hermes-agent is tool-calling-first).
    Permissive when the field is absent/malformed: some OpenRouter-compatible gateways (Nous Portal,
    private mirrors) don't populate it, and the picker must not silently empty for them.

    Ported from Kilo-Org/kilocode#9068.
    """
    params = item.get("supported_parameters") if isinstance(item, dict) else None
    return "tools" in params if isinstance(params, list) else True


# Reasoning-capability cache slots, one set per catalog (OpenRouter, Nous Portal). The logic
# lives in models_reasoning_caps and reads/writes these by name so tests can reset them here.
# ``*_cache``: model id → parsed caps for the process lifetime; ``*_failed_at``: monotonic time
# of the last failed fetch (60s re-fetch suppression); the flags are once-per-process guards.
_openrouter_reasoning_caps_cache: dict[str, Optional[dict[str, Any]]] | None = None
_openrouter_reasoning_caps_failed_at: float | None = None
_openrouter_caps_disk_checked = False
_openrouter_caps_warm_started = False
_nous_reasoning_caps_cache: dict[str, Optional[dict[str, Any]]] | None = None
_nous_reasoning_caps_failed_at: float | None = None
_nous_caps_disk_checked = False
_nous_caps_warm_started = False


from agent.reasoning_effort import clamp_effort as _clamp_effort


def clamp_reasoning_effort_to_supported(
    effort: Optional[str], supported_efforts: Optional[list[str]]) -> Optional[str]:
    """Thin wrapper over :func:`agent.reasoning_effort.clamp_effort`: keep a supported level verbatim,
    else the nearest WEAKER supported level (never silently escalate cost), else the weakest; unknown
    supported-sets and bespoke level names pass through unchanged."""
    return _clamp_effort(effort, supported_efforts)


def _fetch_live_catalog_index(url: str, timeout: float, opener) -> Optional[tuple[list, dict[str, dict[str, Any]]]]:
    """GET an OpenAI-style ``/models`` listing → ``(raw data array, {id: item})``, or None when the
    endpoint is unreachable or the payload has no ``data`` list."""
    try:
        payload = _get_json(url, timeout=timeout, headers={"Accept": "application/json"}, opener=opener)
    except Exception:
        return None
    live_items = payload.get("data", [])
    if not isinstance(live_items, list):
        return None
    live_by_id = {
        mid: item for item in live_items if isinstance(item, dict) and (mid := str(item.get("id") or "").strip())
    }
    return live_items, live_by_id


def fetch_openrouter_models(
    timeout: float = 8.0, *, force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return the curated OpenRouter picker list, refreshed from the live catalog when possible."""
    global _openrouter_catalog_cache

    if _openrouter_catalog_cache is not None and not force_refresh:
        return list(_openrouter_catalog_cache)

    # Remote catalog manifest first, in-repo snapshot when unreachable; the live /v1/models filter
    # (tool support, free pricing) is applied on top either way.
    try:
        from hermes_cli.model_catalog import get_curated_openrouter_models
        remote = get_curated_openrouter_models()
    except Exception:
        remote = None
    fallback = list(remote) if remote else list(OPENROUTER_MODELS)

    live = _fetch_live_catalog_index(_OPENROUTER_CATALOG_URL, timeout, _urlopen_model_catalog_request)
    if live is None:
        return list(_openrouter_catalog_cache or fallback)
    live_items, live_by_id = live

    # Free warm-up for the reasoning-capability cache: same payload the caps fetch would pull.
    global _openrouter_reasoning_caps_cache
    seeded = _seed_reasoning_caps(_OPENROUTER_CATALOG_URL, live_items)
    if _openrouter_reasoning_caps_cache is None and seeded is not None:
        _openrouter_reasoning_caps_cache = seeded

    curated: list[tuple[str, str]] = []
    silent_default = get_preferred_silent_default_model("openrouter")
    for preferred_id, _ in fallback:
        live_item = live_by_id.get(preferred_id)
        # Hide models without tool-calling support — selecting one fails at the first tool call.
        if live_item is None or not _openrouter_model_supports_tools(live_item):
            continue
        # Hide models that don't advertise tool-calling support — hermes-agent requires it and surfacing
        # them leads to immediate runtime failures when the user selects them. Ported from
        # Kilo-Org/kilocode#9068.
        if preferred_id == silent_default:
            desc = "default"  # keep the silent-default badge through the live refresh
        else:
            desc = "free" if _openrouter_model_is_free(live_item.get("pricing")) else ""
        curated.append((preferred_id, desc))

    if not curated:
        return list(_openrouter_catalog_cache or fallback)
    if not curated[0][1]:
        curated[0] = (curated[0][0], "recommended")
    _openrouter_catalog_cache = curated
    return list(curated)


def model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the OpenRouter model-id strings."""
    return [mid for mid, _ in fetch_openrouter_models(force_refresh=force_refresh)]


def get_curated_nous_model_ids() -> list[str]:
    """Curated Nous Portal model ids: the remote catalog manifest, else the in-repo
    ``_PROVIDER_MODELS["nous"]`` snapshot. Always a list."""
    try:
        from hermes_cli.model_catalog import get_curated_nous_models
        remote = get_curated_nous_models()
    except Exception:
        remote = None
    return list(remote or _PROVIDER_MODELS.get("nous", []))


def _ai_gateway_model_is_free(pricing: Any) -> bool:
    return _zero_priced(pricing, ("input", "output"), "0")


def fetch_ai_gateway_models(
    timeout: float = 8.0, *, force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return the curated AI Gateway picker list, refreshed from the live catalog when possible."""
    global _ai_gateway_catalog_cache

    if _ai_gateway_catalog_cache is not None and not force_refresh:
        return list(_ai_gateway_catalog_cache)

    from hermes_constants import AI_GATEWAY_BASE_URL

    fallback = list(VERCEL_AI_GATEWAY_MODELS)
    live = _fetch_live_catalog_index(f"{AI_GATEWAY_BASE_URL.rstrip('/')}/models", timeout, urllib.request.urlopen)
    if live is None:
        return list(_ai_gateway_catalog_cache or fallback)
    _, live_by_id = live

    curated = [
        (pid, "free" if _ai_gateway_model_is_free(live_by_id[pid].get("pricing")) else "")
        for pid, _ in fallback if pid in live_by_id]
    if not curated:
        return list(_ai_gateway_catalog_cache or fallback)

    # A free Moonshot model in the live catalog is auto-promoted to #1 as "recommended".
    free_moonshot = next(
        (mid for mid, item in live_by_id.items()
         if mid.startswith("moonshotai/") and _ai_gateway_model_is_free(item.get("pricing"))),
        None)
    if free_moonshot:
        curated = [(free_moonshot, "recommended")] + [(mid, desc) for mid, desc in curated if mid != free_moonshot]
    else:
        curated[0] = (curated[0][0], "recommended")
    _ai_gateway_catalog_cache = curated
    return list(curated)


def ai_gateway_model_ids(*, force_refresh: bool = False) -> list[str]:
    """Return just the AI Gateway model-id strings."""
    return [mid for mid, _ in fetch_ai_gateway_models(force_refresh=force_refresh)]


# ---------------------------------------------------------------------------
# Provider identity: ``provider:model`` parsing, auto-detection, labels
# ---------------------------------------------------------------------------

# All provider IDs and aliases valid on the left of the ``provider:model`` syntax.
_KNOWN_PROVIDER_NAMES: set[str] = set(_PROVIDER_LABELS) | set(_PROVIDER_ALIASES) | {"openrouter", "custom"}


_CONFIG_ERRORS = (ImportError, OSError, RuntimeError, TypeError, ValueError, AttributeError)


def _configured_custom_provider_ids() -> set[str]:
    """Return routable custom-provider IDs configured by the user."""
    ids = {"custom"}
    try:
        from hermes_cli.config import load_config
        from hermes_cli.providers import custom_provider_slug

        config = load_config()
        providers = config.get("providers", {})
        if isinstance(providers, dict):
            ids.update(custom_provider_slug(str(entry.get("name") or key), str(key))
                       for key, entry in providers.items() if isinstance(entry, dict))
        legacy = config.get("custom_providers", [])
        if isinstance(legacy, list):
            ids.update(
                custom_provider_slug(str(entry.get("name") or "")) for entry in legacy if isinstance(entry, dict))
    except _CONFIG_ERRORS:
        pass
    return ids


def _provider_has_credentials(pid: str) -> bool:
    try:
        from hermes_cli.auth import get_auth_status, has_usable_secret

        if pid == "custom":
            return bool((_get_custom_base_url() or "").strip())
        if pid == "openrouter":
            return has_usable_secret(os.getenv("OPENROUTER_API_KEY", ""))
        status = get_auth_status(pid)
        return bool(status.get("logged_in") or status.get("configured"))
    except Exception:
        return False


def list_available_providers() -> list[dict[str, str]]:
    """``{id, label, aliases, authenticated}`` for every provider usable with ``provider:model``,
    derived from :data:`CANONICAL_PROVIDERS` (shared with ``hermes model`` and ``/model``)."""
    aliases_for: dict[str, list[str]] = {}
    for alias, canonical in _PROVIDER_ALIASES.items():
        aliases_for.setdefault(canonical, []).append(alias)
    return [
        {
            "id": pid,
            "label": _PROVIDER_LABELS.get(pid, pid),
            "aliases": aliases_for.get(pid, []),
            "authenticated": _provider_has_credentials(pid)}
        for pid in [p.slug for p in CANONICAL_PROVIDERS] + ["custom"]]


def parse_model_input(raw: str, current_provider: str) -> tuple[str, str]:
    """Parse ``/model`` input into ``(provider, model)``. The colon is a provider delimiter only when
    the left side is a known provider/alias, so ``anthropic/claude-3.5-sonnet:beta`` stays a model."""
    stripped = raw.strip()
    colon = stripped.find(":")
    if colon > 0:
        provider_part = stripped[:colon].strip().lower()
        model_part = stripped[colon + 1:].strip()
        if provider_part and model_part and provider_part in _KNOWN_PROVIDER_NAMES:
            if provider_part == "custom":
                # Longest configured ``custom:<name>`` id that prefixes the input wins.
                lowered = stripped.lower()
                for custom_id in sorted(_configured_custom_provider_ids() - {"custom"}, key=len, reverse=True):
                    if lowered.startswith(f"{custom_id.lower()}:"):
                        return custom_id, stripped[len(custom_id) + 1 :].strip()
                # ``custom:local:qwen`` → ("custom:local", "qwen") for a configured named provider;
                # single-colon ``custom:qwen`` → ("custom", "qwen") as before.
                if ":" in model_part:
                    custom_name, actual_model = (part.strip() for part in model_part.split(":", 1))
                    if custom_name and actual_model:
                        if f"custom:{custom_name.lower()}" in _configured_custom_provider_ids():
                            return (f"custom:{custom_name.lower()}", actual_model)
                        return ("custom", model_part)
            return (normalize_provider(provider_part), model_part)
    return (current_provider, stripped)


def _get_custom_base_url() -> str:
    """The custom endpoint ``model.base_url`` from config.yaml."""
    return str(_get_model_config_dict().get("base_url", "")).strip()


def _get_provider_config_dict(provider: str) -> dict[str, Any]:
    """Return config.yaml providers.<provider>, or an empty dict."""
    key = str(provider or "").strip()
    if not key:
        return {}
    try:
        from hermes_cli.config import load_config
        providers_cfg = load_config().get("providers", {})
        if isinstance(providers_cfg, dict):
            entry = providers_cfg.get(key) or providers_cfg.get(key.lower())
            if isinstance(entry, dict):
                return entry
    except _CONFIG_ERRORS:
        pass
    return {}


def _get_model_config_dict() -> dict[str, Any]:
    """Return the main model config mapping, or an empty dict."""
    try:
        from hermes_cli.config import load_config
        model_cfg = load_config().get("model", {})
        if isinstance(model_cfg, dict):
            return model_cfg
    except Exception:
        pass
    return {}


def _base_url_looks_like_anthropic_messages(base_url: str) -> bool:
    normalized = str(base_url or "").strip().lower().rstrip("/")
    if not normalized:
        return False
    return urllib.parse.urlparse(normalized).path.rstrip("/").endswith(("/anthropic", "/anthropic/v1"))


def _anthropic_models_url(base_url: Optional[str] = None) -> str:
    endpoint = str(base_url or "https://api.anthropic.com").strip().rstrip("/")
    return endpoint + ("/models" if endpoint.endswith("/v1") else "/v1/models")


def curated_models_for_provider(
    provider: Optional[str],
    *,
    force_refresh: bool = False,
) -> list[tuple[str, str]]:
    """Return ``(model_id, description)`` tuples for a provider's model list.

    Tries to fetch the live model list from the provider's API first,
    falling back to the static ``_PROVIDER_MODELS`` catalog if the API
    is unreachable.
    """
    normalized = normalize_provider(provider)
    if normalized == "openrouter":
        return fetch_openrouter_models(force_refresh=force_refresh)

    # Try live API first (Codex, Nous, etc. all support /models)
    live = provider_model_ids(normalized)
    if live:
        return [(m, "") for m in live]

    # Fallback to static catalog
    models = _PROVIDER_MODELS.get(normalized, [])
    return [(m, "") for m in models]


def _provider_keys(provider: str) -> set[str]:
    key = (provider or "").strip().lower()
    normalized = normalize_provider(provider)
    return {k for k in (key, normalized) if k}


def _provider_catalog_names(provider: str) -> tuple[str, ...]:
    """Active picker models plus retired aliases recognized for detection; a catalog delegate
    (``_PROVIDER_CATALOG_DELEGATES``) answers with its delegate's catalog."""
    provider = _PROVIDER_CATALOG_DELEGATES.get(provider, provider)
    return tuple(_PROVIDER_MODELS.get(provider, [])) + _PROVIDER_RETIRED_ALIASES.get(provider, ())


def _model_in_provider_catalog(name_lower: str, providers: set[str]) -> bool:
    return any(
        name_lower == model.lower()
        for provider in providers
        for model in _provider_catalog_names(provider))


def _openrouter_variant_base(model_id: str) -> Optional[str]:
    """Base model id when ``model_id`` carries a recognized OpenRouter routing-variant suffix
    (``x-ai/grok-4:nitro`` → ``x-ai/grok-4``), else ``None``."""
    base, sep, suffix = (model_id or "").rpartition(":")
    return base if sep and base and suffix.lower() in _OPENROUTER_VARIANT_SUFFIXES else None


def _resolve_static_model_alias(
    name_lower: str, current_keys: set[str]) -> Optional[tuple[str, str]]:
    """Resolve short aliases (e.g. sonnet/opus) using static catalogs only."""
    try:
        from hermes_cli.model_switch import MODEL_ALIASES
    except Exception:
        return None

    identity = MODEL_ALIASES.get(name_lower)
    if identity is None:
        return None

    def _match(provider: str) -> Optional[str]:
        prefix = f"{identity.vendor}/{identity.family}" if provider in _AGGREGATOR_PROVIDERS else identity.family
        prefix = prefix.lower()
        models = _PROVIDER_MODELS.get(_PROVIDER_CATALOG_DELEGATES.get(provider, provider), [])
        return next((m for m in models if m.lower().startswith(prefix)), None)

    # Current provider first, then native vendors, then aggregators / borrow-list providers the user
    # is already on — so `sonnet` resolves to anthropic before any re-exposing provider.
    skip = current_keys | _AGGREGATOR_PROVIDERS | _BORROWED_MODEL_PROVIDERS
    candidates = [
        *current_keys, *(p for p in _PROVIDER_MODELS if p not in skip),
        *(p for p in _AGGREGATOR_PROVIDERS if p in current_keys),
        *(p for p in _BORROWED_MODEL_PROVIDERS if p in current_keys)]
    for provider in candidates:
        if matched := _match(provider):
            return provider, matched
    return None


def detect_static_provider_for_model(
    model_name: str, current_provider: str) -> Optional[tuple[str, str]]:
    """Auto-detect a provider from static catalogs only → ``(provider_id, model_name)`` (the name may
    be remapped by a static alias or a bare provider name), or ``None`` without a confident match."""
    name = (model_name or "").strip()
    if not name:
        return None

    name_lower = name.lower()
    current_keys = _provider_keys(current_provider)

    alias_match = _resolve_static_model_alias(name_lower, current_keys)
    if alias_match:
        return alias_match

    # Step 0: a bare provider name typed as the model (`/model nous`) is a provider switch to that
    # provider's default. Skip "custom" (no catalog) and "openrouter" (needs an explicit model).
    resolved_provider = _PROVIDER_ALIASES.get(name_lower, name_lower)
    if resolved_provider not in {"custom", "openrouter"}:
        default_models = _PROVIDER_MODELS.get(resolved_provider, [])
        if resolved_provider in _PROVIDER_LABELS and default_models and resolved_provider not in current_keys:
            # Cost-safe default, not ``default_models[0]``: metered aggregators list most-capable-first,
            # so [0] would silently escalate `/model nous` to the priciest flagship.
            return (resolved_provider, get_default_model_for_provider(resolved_provider) or default_models[0])

    # A model in the current provider's own catalog never suggests switching.
    if _model_in_provider_catalog(name_lower, current_keys):
        return None

    # Step 1: direct static-catalog match. Aggregators list other vendors' models — never
    # auto-switch TO them. A custom endpoint (custom / custom:*) is never auto-switched away
    # from: the user configured it deliberately and may serve the same model name there.
    if current_provider != "custom" and not current_provider.startswith("custom:"):
        for pid in _PROVIDER_MODELS:
            if pid in current_keys or pid in _AGGREGATOR_PROVIDERS or pid in _BORROWED_MODEL_PROVIDERS:
                continue
            if _model_in_provider_catalog(name_lower, {pid}):
                return (pid, name)

    # Borrow-list providers (re-expose other vendors' models) only after every native-vendor
    # catalog, and only when one is the current provider.
    for pid in _BORROWED_MODEL_PROVIDERS:
        if pid not in current_keys and _model_in_provider_catalog(name_lower, {pid}):
            return (pid, name)

    return None


def _configured_provider_ids() -> set[str]:
    """Provider ids (incl. ``custom:*``) from the user's ``providers:`` config block; empty when config
    is unreadable (callers fall through to built-in catalogs)."""
    try:
        from hermes_cli.config import load_config

        providers = (load_config() or {}).get("providers")
        if not isinstance(providers, dict):
            return set()
        return {key for pid in providers if (key := str(pid).strip().lower())}
    except Exception:
        return set()


def _resolve_provider_prefix(model_name: str) -> Optional[tuple[str, str]]:
    """Route an explicit ``vendor/model`` prefix (``nous/deepseek-v4-pro``, ``ollama/qwen3.5:4b``) to
    a provider the user defined in ``providers:`` (by raw name or alias) instead of the default.

    ``nous/deepseek-v4-pro`` or ``ollama/qwen3.5:4b`` should route to the named provider instead of falling
    back to the configured default (which silently sends non-default models to the wrong endpoint, #87189).
    """
    if "/" not in model_name:
        return None
    vendor, model = model_name.split("/", 1)
    vendor, model = vendor.strip().lower(), model.strip()
    if not vendor or not model:
        return None
    configured = _configured_provider_ids()
    # An explicitly named provider block (``ollama:``) wins over the alias table, which may
    # canonicalize the same name elsewhere (``ollama`` → ``custom``).
    for candidate in (vendor, _PROVIDER_ALIASES.get(vendor, vendor)):
        if candidate in configured:
            return (candidate, model)
    return None


def detect_provider_for_model(
    model_name: str, current_provider: str) -> Optional[tuple[str, str]]:
    """Auto-detect the best provider for a model name: static catalogs (bare provider name → its
    default; direct catalog match), then the OpenRouter catalog, then a configured ``vendor/`` prefix."""
    name = (model_name or "").strip()
    if not name:
        return None

    static_match = detect_static_provider_for_model(name, current_provider)
    if static_match:
        return static_match
    if _model_in_provider_catalog(name.lower(), _provider_keys(current_provider)):
        return None

    # OpenRouter catalog (exact slug, then bare model part).
    or_slug = _find_openrouter_slug(name)
    if or_slug:
        if current_provider != "openrouter" or or_slug != name:
            return ("openrouter", or_slug)
        return None  # already on openrouter with matching name

    # Explicit ``vendor/model`` prefix naming a configured provider — AFTER the OpenRouter lookup so
    # aggregator-native slugs (``deepseek/deepseek-chat``) keep their routing.
    return _resolve_provider_prefix(name)


def _find_openrouter_slug(model_name: str) -> Optional[str]:
    """Full OpenRouter slug for a bare or partial model name (exact slug first, then bare part)."""
    name_lower = model_name.strip().lower()
    if not name_lower:
        return None
    ids = model_ids()
    return (
        next((mid for mid in ids if name_lower == mid.lower()), None)
        or next((mid for mid in ids if "/" in mid and name_lower == mid.split("/", 1)[1].lower()), None)
    )


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to canonical ids. ``"auto"`` passes through — use
    ``hermes_cli.auth.resolve_provider()`` to resolve it from credentials."""
    normalized = (provider or "openrouter").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def provider_label(provider: Optional[str]) -> str:
    """Return a human-friendly label for a provider id or alias."""
    original = (provider or "openrouter").strip()
    normalized = original.lower()
    if normalized == "auto":
        return "Auto"
    normalized = normalize_provider(normalized)
    return _PROVIDER_LABELS.get(normalized, original or "OpenRouter")


def _is_openai_fast_model(model_id: Optional[str]) -> bool:
    """OpenAI flagship eligible for Priority Processing. Codex-series excluded — the Codex Responses
    API doesn't accept ``service_tier``."""
    base = _strip_vendor_prefix(str(model_id or "")).split(":")[0]
    return bool(base) and "codex" not in base and base.startswith(tuple(_OPENAI_FAST_MODE_PREFIXES))


def _strip_vendor_prefix(model_id: str) -> str:
    """Lowercase and strip a ``vendor/`` prefix (``anthropic/claude-opus-4-6`` → ``claude-opus-4-6``)."""
    raw = str(model_id or "").strip().lower()
    return raw.split("/", 1)[1] if "/" in raw else raw


def model_supports_fast_mode(model_id: Optional[str]) -> bool:
    """Return whether Hermes should expose the /fast toggle for this model."""
    from agent.model_metadata import is_grok_46_family

    return (
        _is_anthropic_fast_model(model_id)
        or _is_openai_fast_model(model_id)
        or is_grok_46_family(str(model_id or "")))


def _is_anthropic_fast_model(model_id: Optional[str]) -> bool:
    """Accepts the Anthropic Fast Mode ``speed`` param (Opus 4.8 / Opus 5 only) — deliberately NOT a
    general "fast model" check: Opus 4.7 hard-400s on it, and dedicated ``…-fast`` ids select fast
    inference via the model field and must not also get it."""
    base = _strip_vendor_prefix(str(model_id or "")).split(":")[0]
    if not base.startswith("claude-") or "-fast" in base:
        return False
    return any(v in base for v in ("opus-4-8", "opus-4.8", "opus-5"))


def _fast_mode_route_supported(
    model_id: Optional[str], provider: Optional[str], base_url: Optional[str]) -> bool:
    """Only the first-party endpoint that bills for fast mode may receive its params."""
    from urllib.parse import urlparse

    from agent.model_metadata import is_grok_46_family

    if _is_anthropic_fast_model(model_id):
        allowed = {"anthropic": "api.anthropic.com"}
    elif is_grok_46_family(str(model_id or "")):
        allowed = {"xai": "api.x.ai"}
    else:
        allowed = {"openai": "api.openai.com", "openai-codex": "chatgpt.com"}
    if provider and normalize_provider(provider) not in allowed:
        return False
    host = (urlparse(str(base_url or "")).hostname or "").lower()
    return not host or host in allowed.values()


def resolve_fast_mode_overrides(
    model_id: Optional[str], *, provider: Optional[str] = None, base_url: Optional[str] = None
) -> dict[str, Any] | None:
    """Fast/priority request_overrides — ``{"speed": "fast"}`` (Anthropic Fast Mode) or
    ``{"service_tier": "priority"}`` (OpenAI / xAI Priority Processing) — or None if unsupported.
    With ``provider``/``base_url`` the route is gated too (``_fast_mode_route_supported``) so proxies
    never see the params. Single fast-mode gate for ``/fast`` and ``agent.fast_mode`` windows."""
    if not model_supports_fast_mode(model_id):
        return None
    if (provider or base_url) and not _fast_mode_route_supported(model_id, provider, base_url):
        return None
    return {"speed": "fast"} if _is_anthropic_fast_model(model_id) else {"service_tier": "priority"}


def _first_exchangeable_copilot_token(raw_tokens) -> str:
    """Exchange stored GitHub tokens in order; the first that validates AND exchanges wins (every
    entry is tried so a later valid token survives an earlier malformed one)."""
    from hermes_cli.copilot_auth import exchange_copilot_token, validate_copilot_token

    for raw in raw_tokens:
        raw = str(raw or "").strip()
        if not raw or not validate_copilot_token(raw)[0]:
            continue
        try:
            api_token = exchange_copilot_token(raw)[0]  # (api_token, expires_at, base_url)
        except Exception:
            continue
        if api_token:
            return api_token
    return ""


def _copilot_cli_config_tokens() -> list[str]:
    """``copilotTokens`` from the GitHub Copilot CLI's own plaintext store (JSONC — strip
    ``//``-comment lines), written by ``copilot login`` on hosts without an OS keychain."""
    cli_config = os.path.expanduser("~/.copilot/config.json")
    if not os.path.isfile(cli_config):
        return []
    with open(cli_config, "r", encoding="utf-8", errors="ignore") as fh:
        raw_text = "\n".join(
            line for line in fh.read().splitlines() if not line.lstrip().startswith("//"))
    data = json.loads(raw_text) if raw_text.strip() else {}
    tokens = data.get("copilotTokens")
    return list(tokens.values()) if isinstance(tokens, dict) else []


def _resolve_copilot_catalog_api_key() -> str:
    """Best-effort GitHub token for the Copilot catalog: env vars / ``gh auth token`` via
    ``resolve_api_key_provider_credentials``, then ``auth.json`` ``credential_pool.copilot[]``, then
    ``~/.copilot/config.json`` ``copilotTokens`` (the ACP CLI's own store). Without the latter two,
    keyless users see the picker fall back to the stale curated list on a silent 401."""
    def _pool_token() -> str:
        from hermes_cli.auth import read_credential_pool

        return _first_exchangeable_copilot_token(
            entry.get("access_token") for entry in read_credential_pool("copilot") if isinstance(entry, dict))

    sources = (
        lambda: _api_key_credentials("copilot")[0],
        _pool_token,
        lambda: _first_exchangeable_copilot_token(_copilot_cli_config_tokens()),
    )
    for source in sources:
        try:
            token = source()
        except Exception:
            continue
        if token:
            return token
    return ""


def _model_dedup_key(model_id: str) -> str:
    """Case-insensitive dedup key folded through the picker-search alias table, so a bare live wire
    id and its curated public slug (Kimi ``k3`` / ``kimi-k3``) don't both survive a merge."""
    key = str(model_id).strip().lower()
    try:
        from hermes_cli.model_search import model_alias_canonical
        return model_alias_canonical(key)
    except Exception:
        return key


def _merge_with_models_dev(provider: str, curated: list[str]) -> list[str]:
    """models.dev entries first (their order), then curated-only extras, case-insensitively deduped
    while preserving curated casing. Curated unchanged when models.dev is unreachable/empty."""
    try:
        from agent.models_dev import list_agentic_models
        mdev = list_agentic_models(provider)
    except Exception:
        mdev = []
    if not mdev:
        return list(curated)
    return _merge_unique(_merge_unique([], mdev), curated)


def _openai_discovery_base_url(provider: str) -> str:
    """OpenAI endpoint for model discovery, mirroring runtime precedence so discovery probes the SAME
    endpoint inference uses: ``$OPENAI_BASE_URL`` → config ``model.base_url`` (when the configured
    provider matches) → the canonical default."""
    env_raw = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if env_raw:
        return env_raw
    try:
        model_cfg = _get_model_config_dict()
        cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
        same_provider = normalize_provider(provider) == normalize_provider(cfg_provider)
        if cfg_provider in ("openai", "openai-api") and same_provider:
            cfg_url = str(model_cfg.get("base_url") or "").strip().rstrip("/")
            if cfg_url:
                return cfg_url
    except Exception:
        pass
    return "https://api.openai.com/v1"


def _codex_catalog(normalized: str, force_refresh: bool) -> list[str]:
    from hermes_cli.codex_models import get_codex_model_ids

    # Live OAuth token so the picker matches what ChatGPT lists for this account; hardcoded
    # catalog without a token / when unreachable.
    try:
        from hermes_cli.auth import resolve_codex_runtime_credentials

        access_token = resolve_codex_runtime_credentials(refresh_if_expiring=True).get("api_key")
    except Exception:
        access_token = None
    return get_codex_model_ids(access_token=access_token)


def _copilot_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    try:
        live = _fetch_github_models(_resolve_copilot_catalog_api_key())
        if live:
            return live
    except Exception:
        pass
    return list(_PROVIDER_MODELS.get("copilot", [])) if normalized == "copilot-acp" else None


def _nous_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    try:
        from hermes_cli.auth import fetch_nous_models, resolve_nous_runtime_credentials

        creds = resolve_nous_runtime_credentials()
        if creds:
            live = fetch_nous_models(api_key=creds.get("api_key", ""), inference_base_url=creds.get("base_url", ""))
            if live:
                return live
    except Exception:
        pass
    # Live failed / no creds: the docs-hosted manifest — NOT the in-repo snapshot — so newly added
    # Portal models still surface without a Hermes release.
    return get_curated_nous_model_ids() or None


def _api_key_credentials(normalized: str) -> tuple[str, str]:
    """``(api_key, base_url)`` from ``resolve_api_key_provider_credentials``; empty strings on any miss."""
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials(normalized)
        return str(creds.get("api_key") or "").strip(), str(creds.get("base_url") or "").strip()
    except Exception:
        return "", ""


def _api_key_provider_live(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    """Live /v1/models for a simple api-key provider (stepfun, gmi); None on any miss."""
    api_key, base_url = _api_key_credentials(normalized)
    if not (api_key and base_url):
        return None
    try:
        return fetch_api_models(api_key, base_url) or None
    except Exception:
        return None


def _anthropic_catalog(normalized: str, force_refresh: bool) -> list[str]:
    model_cfg = _get_model_config_dict()
    cfg_base_url = cfg_api_key = ""
    if normalize_provider(str(model_cfg.get("provider", "") or "")) == "anthropic":
        cfg_base_url = str(model_cfg.get("base_url", "") or "").strip()
        cfg_api_key = str(model_cfg.get("api_key", "") or "").strip()
    live = _fetch_anthropic_models(base_url=cfg_base_url or None, api_key=cfg_api_key or None)
    curated = list(_PROVIDER_MODELS.get("anthropic", []))
    if not live:
        return curated
    # The live /v1/models dump lags newly-routed curated aliases (reachable before enumerated):
    # curated first, then live-only extras, so a fresh curated model never disappears.
    return live if cfg_base_url else _merge_unique(curated, live)


def _openai_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    base = _openai_discovery_base_url(normalized)
    # Custom OpenAI-compatible endpoints serve a small curated catalog — use it verbatim. Official
    # OpenAI hosts (canonical and data-residency regional) return 120+ embeddings/whisper/tts/…
    # entries, so intersect with the curated agentic catalog so ``/model`` matches ``hermes model``.
    # Model not in live /v1/models — check the curated catalog before rejecting. Providers may omit models
    # from their live listing that are still valid (stale cache, partial rollout, gated previews). Use the
    # pure-catalog helper (no extra live fetch) so we only accept models Hermes actually ships. (#46850)
    # Their /v1/models listing is access-scoped and authoritative — a model absent from it is one this key
    # CANNOT serve, so the curated soft-accept would manufacture a selection that 400s at first use. Custom
    # OpenAI-compatible proxies keep the fallback (incomplete listings are common there).
    from hermes_cli.providers import is_official_openai_host

    try:
        live = fetch_api_models(api_key, base)
    except Exception:
        live = None
    if not live:
        return None
    if not is_official_openai_host(base):
        return live
    live_lower = {m.lower() for m in live}
    curated = list(_PROVIDER_MODELS.get(normalized, []))
    # Curated order, only models the account has access to; an account serving none of them (rare)
    # falls back to curated so the picker still offers sane defaults.
    return [m for m in curated if m.lower() in live_lower] or curated or live


def _custom_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    base_url = _get_custom_base_url()
    if not base_url:
        return None
    model_cfg = _get_model_config_dict()
    # Try common API key env vars for custom endpoints.
    api_key = (
        str(model_cfg.get("api_key", "") or "").strip()
        or os.getenv("CUSTOM_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or os.getenv("OPENROUTER_API_KEY", ""))
    api_mode = "anthropic_messages" if _base_url_looks_like_anthropic_messages(base_url) else None
    return fetch_api_models(api_key, base_url, api_mode=api_mode) or None


def _bedrock_catalog(normalized: str, force_refresh: bool) -> Optional[list[str]]:
    # Live discovery keyed by the resolved AWS region so EU/AP users see eu.*/ap.* ids.
    try:
        from agent.bedrock_adapter import bedrock_model_ids_or_none

        return bedrock_model_ids_or_none()
    except Exception:
        return None


def _opencode_free_catalog(normalized: str, force_refresh: bool) -> list[str]:
    # Live keyless catalog filtered to the anonymous-servable `*-free` tier ourselves (models.dev's
    # cost.input==0 lags reality); the curated floor applies only when the live fetch fails/is empty.
    return _fetch_opencode_free_models(force_refresh=force_refresh) or list(_PROVIDER_MODELS.get(normalized, []))


# Per-provider catalog sources tried before the generic profile fetch. A fetcher returning None
# falls through to the profile/curated path; a list is returned as-is (even empty).
_PROVIDER_CATALOG_FETCHERS: dict[str, Any] = {
    "openrouter": lambda normalized, force_refresh: model_ids(force_refresh=force_refresh),
    "openai-codex": _codex_catalog,
    "copilot": _copilot_catalog,
    "copilot-acp": _copilot_catalog,
    "nous": _nous_catalog,
    "stepfun": _api_key_provider_live,
    "gmi": _api_key_provider_live,
    "anthropic": _anthropic_catalog,
    "ai-gateway": lambda normalized, force_refresh: _fetch_ai_gateway_models() or None,
    # DeepInfra's generic /models mixes chat, image, video, speech and embedding models; the tagged
    # catalog helper is the only safe source for the chat picker, including its empty/failure result.
    "deepinfra": lambda normalized, force_refresh: _fetch_deepinfra_models(force_refresh=force_refresh) or [],
    "ollama-cloud": lambda normalized, force_refresh: fetch_ollama_cloud_models(force_refresh=force_refresh) or None,
    "openai": _openai_catalog,
    "openai-api": _openai_catalog,
    "custom": _custom_catalog,
    "bedrock": _bedrock_catalog,
    "opencode-free": _opencode_free_catalog}


def _profile_live_catalog(normalized: str) -> Optional[list[str]]:
    """Generic live fetch for any provider registered in providers/ with ``auth_type="api_key"``.

    Live results are merged with the curated list so models the live endpoint omits still appear:
    curated-first by default so the newest curated models lead when the live API lags;
    ``_LIVE_FIRST_PICKER_PROVIDERS`` (OpenCode Zen/Go, authoritative live API) live-first so stale
    curated entries stop polluting the top. Plugin providers without a static entry use the
    profile's ``fallback_models`` as the curated list (Fireworks lists an image model first).
    """
    from providers import get_provider_profile

    profile = get_provider_profile(normalized)
    if not (profile and profile.auth_type == "api_key" and profile.base_url):
        return None
    api_key, base_url = _api_key_credentials(normalized)
    live = profile.fetch_models(api_key=api_key, base_url=base_url or profile.base_url or None) if api_key else None
    if not live:
        return list(profile.fallback_models) if profile.fallback_models else None
    curated = list(_PROVIDER_MODELS.get(normalized, [])) or list(profile.fallback_models or ())
    if not curated:
        return live
    primary, secondary = (live, curated) if normalized in _LIVE_FIRST_PICKER_PROVIDERS else (curated, live)
    return _merge_unique(primary, secondary, key=_model_dedup_key)


def provider_model_ids(provider: Optional[str], *, force_refresh: bool = False) -> list[str]:
    """Best known model catalog for a provider: per-provider live fetchers, then the generic profile
    fetch, then the static list (merged with models.dev for ``_MODELS_DEV_PREFERRED`` providers)."""
    requested = str(provider or "").strip().lower()
    if requested == "ollama":
        return _ollama_local_catalog(force_refresh)

    normalized = normalize_provider(provider)
    fetcher = _PROVIDER_CATALOG_FETCHERS.get(normalized)
    if fetcher is not None:
        models = fetcher(normalized, force_refresh)
        if models is not None:
            return models
    try:
        models = _profile_live_catalog(normalized)
    except Exception:
        models = None
    if models is not None:
        return models

    # Merge static curated list with live API results so models that the live endpoint omits (stale cache,
    # partial rollout) still appear in the picker. Single providers (kimi, zai) use curated-first (commit
    # 658ac1d86) to surface newest models even when live API lags (#46309). OpenCode Zen / Go are different:
    # their live API is the authoritative catalog, so they merge live-first — live entries lead and stale
    # curated entries no longer pollute the top of the picker. (#49129) Plugin providers with no static
    # _PROVIDER_MODELS entry fall back to the profile's curated fallback_models so their agentic picks lead
    # the picker instead of whatever the live catalog happens to return first (e.g. Fireworks lists an image
    # model, flux-*, ahead of its chat models).
    curated_static = list(_PROVIDER_MODELS.get(normalized, []))
    if normalized not in _MODELS_DEV_PREFERRED:
        return curated_static
    merged = _merge_with_models_dev(normalized, curated_static)
    return _xai_finalize_catalog(merged) if normalized in {"xai", "xai-oauth"} else merged


# ---------------------------------------------------------------------------
# Disk cache for provider_model_ids() — keeps /model picker fast (otherwise every open re-fetches
# every authed provider's /v1/models). One JSON file at $HERMES_HOME/provider_models_cache.json;
# entries keyed by credential fingerprint (rotate OPENAI_API_KEY → entry invalidates); 1h TTL;
# only NON-EMPTY results are cached so a transient failure is never pinned; any read/write error
# degrades silently to a live fetch.
# ---------------------------------------------------------------------------

_PROVIDER_MODELS_CACHE_TTL = 3600  # 1h
# Stale-while-revalidate window: an expired same-credentials entry is served IMMEDIATELY while a
# daemon thread refreshes the disk cache; beyond this bound the caller blocks on a live fetch.
# Catalogs change on release timescales, so hour-old data beats stalling every picker surface.
_PROVIDER_MODELS_STALE_SERVE_MAX = 7 * 24 * 3600  # 7d

# Cache keys with a background SWR refresh in flight — dedupes concurrent refreshes.
_swr_refresh_inflight: set = set()
_swr_refresh_lock = threading.Lock()


def _cache_entry(fp: str, models: list[str], at: Optional[float] = None) -> dict:
    """One provider row of the disk cache: credential fingerprint, write time, model ids."""
    return {"fp": fp, "at": time.time() if at is None else at, "models": list(models)}


def _ollama_native_probe_reachable() -> bool:
    """Whether the configured local Ollama root answered the native ``/api/tags`` probe (an empty
    catalog from a reachable server is authoritative; a failed probe is not)."""
    base_url = _get_ollama_base_url()
    headers = _get_ollama_native_headers(base_url) or None
    probe_key = _ollama_probe_cache_key(_root_for_ollama_native_api(base_url), headers)
    return _OLLAMA_LOCAL_PROBE_REACHABLE.get(probe_key) is True


def _spawn_swr_refresh(cache_key: str, refresh_fn=None) -> None:
    """Fire-and-forget daemon refresh of *cache_key*'s cache entry, at most one in flight per key.
    Failures are swallowed — the stale entry stays served until a later refresh succeeds.
    ``refresh_fn`` (no-args → fresh entry dict or None) lets ``custom:<base_url>`` keys from
    :func:`cached_fetch_api_models` reuse the same inflight-dedupe scaffolding."""
    with _swr_refresh_lock:
        if cache_key in _swr_refresh_inflight:
            return
        _swr_refresh_inflight.add(cache_key)

    def _default_refresh():
        live = provider_model_ids(cache_key, force_refresh=True)
        if live or (cache_key == "ollama" and _ollama_native_probe_reachable()):
            return _cache_entry(_credential_fingerprint(cache_key), live or [])
        return None

    def _refresh() -> None:
        try:
            entry = (refresh_fn or _default_refresh)()
            if entry:
                _store_cache_entry(cache_key, entry)
        except Exception:
            logger.debug("SWR refresh failed for %s", cache_key, exc_info=True)
        finally:
            with _swr_refresh_lock:
                _swr_refresh_inflight.discard(cache_key)

    threading.Thread(target=_refresh, daemon=True, name=f"model-cache-swr-{cache_key}").start()


def _provider_models_cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "provider_models_cache.json"


def _credential_fingerprint(provider: str) -> str:
    """Short hash of the credentials ``provider_model_ids(provider)`` would see right now: api-key /
    base-url env vars from ``PROVIDER_REGISTRY`` plus the mtimes of ``auth.json`` and external
    credential files (OAuth re-auth busts the cache without parsing every file shape)."""
    import hashlib

    # Keyless providers serve the catalog anonymously: nothing the user rotates should invalidate
    # the entry, so a stable fingerprint keeps the SWR cache alive and busts only on TTL expiry.
    if (provider or "").strip().lower() in _KEYLESS_STABLE_CACHE_PROVIDERS:
        return "keyless:" + (provider or "").strip().lower()

    parts: list[str] = []
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        pcfg = PROVIDER_REGISTRY.get(provider)
        if pcfg is not None:
            for ev in getattr(pcfg, "api_key_env_vars", ()) or ():
                parts.append(f"{ev}={os.environ.get(ev, '')}")
            bev = getattr(pcfg, "base_url_env_var", "") or ""
            if bev:
                parts.append(f"{bev}={os.environ.get(bev, '')}")
    except Exception:
        pass

    # config.yaml's model.base_url changes the endpoint discovery probes (data-residency hosts)
    # without touching any env var, so it must change the fingerprint too.
    if provider in ("openai", "openai-api"):
        try:
            parts.append(f"effective_base={_openai_discovery_base_url(provider)}")
        except Exception:
            pass

    if provider == "ollama":
        provider_cfg = _get_provider_config_dict("ollama")
        key_env = provider_cfg.get("key_env") or provider_cfg.get("api_key_env") or ""
        model_cfg = _get_model_config_dict()
        parts += [
            f"OLLAMA_HOST={os.environ.get('OLLAMA_HOST', '')}",
            "providers.ollama.base_url="
            f"{provider_cfg.get('base_url', '') or provider_cfg.get('api', '') or provider_cfg.get('url', '')}",
            f"providers.ollama.api_key={provider_cfg.get('api_key', '')}",
            f"providers.ollama.key_env={key_env}",
        ]
        if key_env:
            parts.append(f"{key_env}={os.environ.get(str(key_env), '')}")
        parts += [
            f"model.provider={model_cfg.get('provider', '')}|model.base_url={model_cfg.get('base_url', '')}",
            "providers.ollama.extra_headers="
            + json.dumps(provider_cfg.get("extra_headers", {}), sort_keys=True, default=str),
        ]

    def _mtime_part(label: str, path) -> None:
        try:
            parts.append(f"{label}@{os.stat(path).st_mtime_ns}")
        except FileNotFoundError:
            parts.append(f"{label}@missing")
        except Exception:
            pass

    try:
        from hermes_constants import get_hermes_home
        for rel in ("auth.json", "credentials.json"):
            _mtime_part(rel, get_hermes_home() / rel)
    except Exception:
        pass
    for rel in ("~/.codex/auth.json", "~/.claude/.credentials.json",
                "~/.config/github-copilot/hosts.json", "~/.minimax/credentials.json"):
        path = os.path.expanduser(rel)
        _mtime_part(path, path)

    blob = "|".join(parts).encode("utf-8", errors="replace")
    # blake2b, not sha256: fingerprint only (collisions = a harmless cache miss), and CodeQL's
    # weak-sensitive-data-hashing rule flags sha256 over env vars named *API_KEY*/*TOKEN*.
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def _load_provider_models_cache() -> dict:
    """Return the full cache dict, or {} on any error."""
    try:
        return _read_json_cache(_provider_models_cache_path()) or {}
    except Exception:
        return {}


_cache_write_lock = threading.Lock()


def _save_provider_models_cache(data: dict) -> None:
    """Persist the cache dict. Best-effort — silent on any error."""
    try:
        _write_json_cache(_provider_models_cache_path(), data, indent=None)
    except Exception:
        pass


def _store_cache_entry(cache_key: str, entry: dict, cache: Optional[dict] = None) -> None:
    """Write one row into the disk cache (reloading the latest state unless ``cache`` is given)."""
    if cache is None:
        cache = _load_provider_models_cache()
    cache[cache_key] = entry
    _save_provider_models_cache(cache)


def update_provider_cache_entry(provider: str, models: list[str]) -> None:
    """Thread-safe single-entry update for parallel prefetch workers: load-modify-save under a lock
    so concurrent fetches don't clobber each other's rows. Best-effort, silent on any error."""
    try:
        normalized = normalize_provider(provider) or (provider or "")
        if not normalized or not models:
            return
        fp = _credential_fingerprint(normalized)
        with _cache_write_lock:
            _store_cache_entry(normalized, _cache_entry(fp, models))
    except Exception:
        pass


def _normalized_cache_slug(provider: Optional[str]) -> str:
    """``ollama`` stays a raw slug (its alias would canonicalize to ``custom``); everything else normalizes."""
    requested = str(provider or "").strip().lower()
    return requested if requested == "ollama" else (normalize_provider(provider) or (provider or ""))


def cached_provider_model_ids(
    provider: Optional[str], *, force_refresh: bool = False,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL) -> list[str]:
    """Disk-cached :func:`provider_model_ids`: fresh cache hit, else live fetch persisting a non-empty
    result. Always returns a list."""
    normalized = _normalized_cache_slug(provider)
    if not normalized:
        return []
    is_ollama = normalized == "ollama"
    if is_ollama:
        ttl_seconds = min(ttl_seconds, _OLLAMA_LOCAL_MODELS_CACHE_TTL)

    cache = _load_provider_models_cache()
    fp = _credential_fingerprint(normalized)
    entry = cache.get(normalized)
    now = time.time()

    if not force_refresh and _cache_entry_valid(entry, fp, allow_empty=is_ollama):
        age = now - entry["at"]
        if age < ttl_seconds:
            return list(entry["models"])
        # Empty native catalogs are authoritative only for the short native TTL — never served
        # through the stale window. Non-empty stale rows are served immediately (SWR) so picker
        # opens never block on serial /v1/models round-trips.
        if entry["models"] and age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            _spawn_swr_refresh(normalized)
            return list(entry["models"])

    live = provider_model_ids(normalized, force_refresh=force_refresh)
    if live:
        _store_cache_entry(normalized, _cache_entry(fp, live, now), cache)
        return list(live)

    if is_ollama:
        if _ollama_native_probe_reachable():
            # A reachable empty native catalog is authoritative; do not resurrect a stale disk catalog.
            _store_cache_entry(normalized, _cache_entry(fp, [], now), cache)
            return []
        # A failed/non-native probe is not authoritative: keep a stale catalog rather than blanking
        # the picker during a transient outage.
        same_creds = isinstance(entry, dict) and entry.get("fp") == fp
        if same_creds and isinstance(entry.get("models"), list) and entry["models"]:
            return list(entry["models"])
        return []
    # Live returned nothing: a stale same-fingerprint entry beats an empty result.
    if _cache_entry_valid(entry, fp):
        return list(entry["models"])
    return []


def clear_provider_models_cache(provider: Optional[str] = None) -> None:
    """Drop one provider's cache entry, or wipe the whole cache (``provider=None``). Used by
    ``/model --refresh`` and ``hermes model --refresh``."""
    try:
        # Native Ollama tags are keyed by root URL, not provider slug — a targeted refresh can't
        # identify the root from the name alone, so clear this small in-process cache every time.
        _OLLAMA_LOCAL_MODELS_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_FAILURE_CACHE.clear()
        _OLLAMA_LOCAL_PROBE_REACHABLE.clear()
        if provider is None:
            path = _provider_models_cache_path()
            if path.exists():
                path.unlink()
            return
        cache = _load_provider_models_cache()
        normalized = _normalized_cache_slug(provider)
        if normalized in cache:
            del cache[normalized]
            _save_provider_models_cache(cache)
    except Exception:
        pass


def _resolve_anthropic_pool_catalog_credentials() -> tuple[str, str]:
    """Read-only API-key pool credential for model discovery (``resolve_anthropic_token()`` ignores
    ``api_key`` pool entries — its runtime contract is OAuth-oriented)."""
    try:
        from agent.credential_pool import AUTH_TYPE_API_KEY
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool("anthropic"):
            if not isinstance(entry, dict) or entry.get("auth_type") != AUTH_TYPE_API_KEY:
                continue
            token = str(entry.get("access_token") or "").strip()
            if token:
                return token, str(entry.get("base_url") or entry.get("inference_base_url") or "").strip()
    except Exception:
        pass
    return "", ""


def _fetch_anthropic_models(
    timeout: float = 5.0, *, base_url: Optional[str] = None, api_key: Optional[str] = None
) -> Optional[list[str]]:
    """Sorted model ids from the Anthropic /v1/models endpoint, or None. Credentials: explicit
    ``api_key``, else ``resolve_anthropic_token()`` (env / OAuth / Claude Code), else a read-only
    API-key credential_pool entry."""
    try:
        from agent.anthropic_credentials import resolve_anthropic_token, _is_oauth_token
    except ImportError:
        return None

    resolved_base_url = base_url
    token = (api_key or "").strip() or resolve_anthropic_token()
    if not token:
        # A pool credential and its endpoint are one security boundary — never pair the pool key
        # with a caller-provided endpoint.
        token, resolved_base_url = _resolve_anthropic_pool_catalog_credentials()
    if not token:
        return None

    headers: dict[str, str] = {"anthropic-version": "2023-06-01"}
    is_oauth = _is_oauth_token(token)
    if is_oauth:
        headers["Authorization"] = f"Bearer {token}"
        from agent.anthropic_adapter import _COMMON_BETAS, _OAUTH_ONLY_BETAS, _CONTEXT_1M_BETA
        headers["anthropic-beta"] = ",".join(_COMMON_BETAS + _OAUTH_ONLY_BETAS)
    else:
        headers["x-api-key"] = token

    url = _anthropic_models_url(resolved_base_url)
    try:
        try:
            data = _get_json(url, timeout=timeout, headers=headers)
        except urllib.error.HTTPError as http_err:
            # OAuth subscriptions that 400 the 1M context beta ("long context beta is not yet
            # available for this subscription"): retry once without it; re-raise anything else.
            if not (is_oauth and http_err.code == 400):
                raise
            try:
                body_text = http_err.read().decode(errors="ignore").lower()
            except Exception:
                body_text = ""
            if not ("long context beta" in body_text and "not yet available" in body_text):
                raise
            headers["anthropic-beta"] = ",".join(
                [b for b in _COMMON_BETAS if b != _CONTEXT_1M_BETA] + list(_OAUTH_ONLY_BETAS)
            )
            data = _get_json(url, timeout=timeout, headers=headers)
        models = [m["id"] for m in data.get("data", []) if m.get("id")]
        # opus, then sonnet, then haiku; alphabetical within tier.
        return sorted(models, key=lambda m: ("opus" not in m, "sonnet" not in m, "haiku" not in m, m))
    except Exception as e:
        logger.debug("Failed to fetch Anthropic models: %s", e)
        return None


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    data = payload.get("data", []) if isinstance(payload, dict) else payload
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def copilot_default_headers(*, is_agent_turn: bool = True) -> dict[str, str]:
    """Standard headers for Copilot API requests."""
    try:
        from hermes_cli.copilot_auth import copilot_request_headers
        return copilot_request_headers(is_agent_turn=is_agent_turn)
    except ImportError:
        return {
            "Editor-Version": COPILOT_EDITOR_VERSION,
            "User-Agent": "HermesAgent/1.0",
            "Openai-Intent": "conversation-edits",
            "x-initiator": "agent" if is_agent_turn else "user"}


_COPILOT_CHAT_ENDPOINTS = {"/chat/completions", "/responses", "/v1/messages"}


def _copilot_catalog_item_is_text_model(
    item: dict[str, Any], *, ignore_picker_flag: bool = False) -> bool:
    if not str(item.get("id") or "").strip():
        return False
    if not ignore_picker_flag and item.get("model_picker_enabled") is False:
        return False
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        model_type = str(capabilities.get("type") or "").strip().lower()
        if model_type and model_type != "chat":
            return False
    supported_endpoints = item.get("supported_endpoints")
    if isinstance(supported_endpoints, list):
        endpoints = {e for endpoint in supported_endpoints if (e := str(endpoint).strip())}
        if endpoints and not endpoints & _COPILOT_CHAT_ENDPOINTS:
            return False
    return True


def _copilot_text_models(items: list[dict[str, Any]], *, ignore_picker_flag: bool = False) -> list[dict[str, Any]]:
    """Chat-capable catalog rows, deduped by id, in catalog order."""
    models: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in items:
        model_id = str(item.get("id") or "").strip()
        if model_id in seen_ids:
            continue
        if not _copilot_catalog_item_is_text_model(item, ignore_picker_flag=ignore_picker_flag):
            continue
        seen_ids.add(model_id)
        models.append(item)
    return models


# Short-TTL cache of the filtered GitHub Copilot /models catalog (picker + context/normalize helpers
# share it). Keyed by the api_key of the successful fetch so a credential swap never serves the
# previous account's catalog; monotonic clock; lock-free (a race at worst duplicates one fetch).
_github_model_catalog_cache: Optional[list[dict[str, Any]]] = None
_github_model_catalog_cache_key: Optional[str] = None
_github_model_catalog_cache_time: float = 0.0
_GITHUB_MODEL_CATALOG_CACHE_TTL = 300  # 5 minutes


def fetch_github_model_catalog(
    api_key: Optional[str] = None, timeout: float = 5.0) -> Optional[list[dict[str, Any]]]:
    """Fetch the live GitHub Copilot model catalog for this account."""
    global _github_model_catalog_cache, _github_model_catalog_cache_key
    global _github_model_catalog_cache_time

    if (
        _github_model_catalog_cache is not None
        and _github_model_catalog_cache_key == api_key
        and (time.monotonic() - _github_model_catalog_cache_time) < _GITHUB_MODEL_CATALOG_CACHE_TTL
    ):
        return copy.deepcopy(_github_model_catalog_cache)  # deep: callers must not mutate cached dicts

    attempts: list[dict[str, str]] = []
    if api_key:
        attempts.append({**copilot_default_headers(), "Authorization": f"Bearer {api_key}"})
    attempts.append(copilot_default_headers())

    for headers in attempts:
        try:
            items = _payload_items(_get_json(COPILOT_MODELS_URL, timeout=timeout, headers=headers))
        except Exception:
            continue
        models = _copilot_text_models(items)
        if not models and items:
            # GitHub has been observed returning ``model_picker_enabled: false`` for EVERY model on
            # some accounts, which would strand the picker on the stale curated fallback. The flag
            # is a display hint, not an availability contract — retry without it (chat/endpoint
            # checks still apply).
            models = _copilot_text_models(items, ignore_picker_flag=True)
        if models:
            _github_model_catalog_cache = copy.deepcopy(models)
            _github_model_catalog_cache_key = api_key
            _github_model_catalog_cache_time = time.monotonic()
            return models
    return None


# ─── Copilot catalog context-window helpers ───

# Module-level cache: {model_id: max_prompt_tokens}
_copilot_context_cache: dict[str, int] = {}
_copilot_context_cache_time: float = 0.0
_COPILOT_CONTEXT_CACHE_TTL = 3600  # 1 hour


def get_copilot_model_context(model_id: str, api_key: Optional[str] = None) -> Optional[int]:
    """``max_prompt_tokens`` for a Copilot model from the live /models API (cached in-process 1h; a
    miss on a fresh cache does not re-fetch), or None."""
    global _copilot_context_cache, _copilot_context_cache_time

    if _copilot_context_cache and (time.time() - _copilot_context_cache_time < _COPILOT_CONTEXT_CACHE_TTL):
        return _copilot_context_cache.get(model_id)

    catalog = fetch_github_model_catalog(api_key=api_key)
    if not catalog:
        return None
    cache: dict[str, int] = {}
    for item in catalog:
        mid = str(item.get("id") or "").strip()
        max_prompt = ((item.get("capabilities") or {}).get("limits") or {}).get("max_prompt_tokens")
        if mid and isinstance(max_prompt, int) and max_prompt > 0:
            cache[mid] = max_prompt
    _copilot_context_cache = cache
    _copilot_context_cache_time = time.time()
    return cache.get(model_id)


def _is_github_models_base_url(base_url: Optional[str]) -> bool:
    return (base_url or "").strip().rstrip("/").lower().startswith(
        (COPILOT_BASE_URL, "https://models.github.ai/inference", "https://models.inference.ai.azure.com")
    )


def _fetch_github_models(api_key: Optional[str] = None, timeout: float = 5.0) -> Optional[list[str]]:
    catalog = fetch_github_model_catalog(api_key=api_key, timeout=timeout)
    return [item.get("id", "") for item in catalog if item.get("id")] if catalog else None


def _copilot_catalog_ids(
    catalog: Optional[list[dict[str, Any]]] = None, api_key: Optional[str] = None) -> set[str]:
    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)
    return {mid for item in (catalog or []) if (mid := str(item.get("id") or "").strip())}


def normalize_copilot_model_id(
    model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None) -> str:
    raw = str(model_id or "").strip()
    if not raw:
        return ""

    catalog_ids = _copilot_catalog_ids(catalog=catalog, api_key=api_key)
    alias = _COPILOT_MODEL_ALIASES.get(raw)
    if alias:
        return alias

    candidates = [raw]
    if "/" in raw:
        candidates.append(raw.split("/", 1)[1].strip())
    if raw.endswith(("-mini", "-nano", "-chat")):
        candidates.append(raw[:-5])

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in _COPILOT_MODEL_ALIASES:
            return _COPILOT_MODEL_ALIASES[candidate]
        if candidate in catalog_ids:
            return candidate

    if "/" in raw:
        return raw.split("/", 1)[1].strip()
    return raw


def _github_reasoning_efforts_for_model_id(model_id: str) -> list[str]:
    raw = (model_id or "").strip().lower()
    if raw.startswith(("openai/o1", "openai/o3", "openai/o4", "o1", "o3", "o4")):
        return list(COPILOT_REASONING_EFFORTS_O_SERIES)
    normalized = normalize_copilot_model_id(model_id).lower()
    if normalized.startswith("gpt-5"):
        return list(COPILOT_REASONING_EFFORTS_GPT5)
    return []


def _should_use_copilot_responses_api(model_id: str) -> bool:
    """opencode's ``shouldUseCopilotResponsesApi``: GPT-5+ uses the Responses API except
    ``gpt-5-mini``; non-GPT models (Claude, Gemini, ...) use Chat Completions."""
    match = re.match(r"^gpt-(\d+)", model_id)
    return bool(match) and int(match.group(1)) >= 5 and not model_id.startswith("gpt-5-mini")


def copilot_model_api_mode(
    model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None) -> str:
    """API mode for a Copilot model from the id pattern (opencode's approach). Copilot's Claude models
    go through its OpenAI-compatible chat endpoint, not the native Anthropic adapter: the catalog may
    advertise /v1/messages but the Copilot token/header scheme lives in the OpenAI client path."""
    if catalog is None and api_key:  # fetch once so normalize + endpoint check share it
        catalog = fetch_github_model_catalog(api_key=api_key)
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if normalized and _should_use_copilot_responses_api(normalized):
        return "codex_responses"
    return "chat_completions"


def azure_foundry_model_api_mode(model_name: Optional[str]) -> Optional[str]:
    """``"codex_responses"`` for families that only accept the Responses API on Azure Foundry (GPT-5.x
    incl. gpt-5-mini, codex, o1/o3/o4), else None. Any ``vendor/`` prefix is stripped first."""
    raw = str(model_name or "").strip().lower().rsplit("/", 1)[-1]
    return "codex_responses" if raw and raw.startswith(tuple(_AZURE_FOUNDRY_RESPONSES_PREFIXES)) else None


_OPENCODE_FAMILIES = ("opencode-free", "opencode-go", "opencode-zen")


def opencode_provider_family(provider_id: Optional[str]) -> Optional[str]:
    """Resolve a provider id (canonical or prefixed) to its OpenCode family, or None.

    Returns ``"opencode-zen"`` or ``"opencode-go"`` for the built-in providers AND for custom providers
    whose name extends a family slug (e.g. ``opencode-go-bridge`` pointing at
    ``https://opencode.ai/zen/go/v1``, issue #85589). Matching is case-insensitive. Custom family providers
    need the same per-model api_mode routing and /v1 base-url normalization as the built-ins — this
    predicate is the single owner of that family-membership question; do not re-implement it inline.
    """
    raw = str(provider_id or "").strip().lower()
    if not raw:
        return None
    canonical = normalize_provider(provider_id)
    if canonical in _OPENCODE_FAMILIES:
        return canonical
    return next((f for f in _OPENCODE_FAMILIES if raw.startswith(f)), None)


def normalize_opencode_model_id(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Normalize OpenCode config IDs to the bare model slug used in API requests."""
    family = opencode_provider_family(provider_id)
    current = str(model_id or "").strip()
    if not current or family is None:
        return current
    for prefix in (f"{provider_id or family}/", f"{family}/"):
        if current.lower().startswith(prefix.lower()):
            return current[len(prefix):]
    return current


# OpenCode Zen free-tier models (``*-free`` slugs plus unsuffixed ones like big-pickle) are
# served ANONYMOUSLY on the Zen relay: no Authorization header succeeds, while ANY unrecognized
# non-empty bearer — including our placeholder and OpenCode GO subscription keys — is 401'd (the
# Go relay doesn't serve the free tier at all).
OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER = "opencode-zen-free-keyless"
_OPENCODE_ZEN_FREE_BASE_URL = "https://opencode.ai/zen/v1"

# ``-free``-suffixed slugs that are KEYED (Go-subscription) models, NOT anonymous-servable —
# excluded from the keyless catalog despite the suffix (ox-alpha-free is Ox Alpha's Go twin).
_OPENCODE_FREE_KEYED_SUFFIX_MODELS = frozenset({"ox-alpha-free"})

# In-process memo for _fetch_opencode_free_models(): (fetched_at, ids-or-None). Validation and
# healing call provider_model_ids("opencode-free") several times per resolution; failures are
# memoized too so an unreachable relay doesn't stall every call for `timeout` seconds.
_opencode_free_live_memo: Optional[tuple[float, Optional[list[str]]]] = None
_OPENCODE_FREE_LIVE_MEMO_TTL = 300.0  # 5 min; SWR disk cache handles the rest


def opencode_zen_free_headers() -> dict:
    """Client default_headers for anonymous Zen free-tier requests. ``Authorization: ""`` overrides the
    OpenAI SDK's ``Bearer <api_key>`` so the placeholder never reaches the wire (the relay 401s any
    unknown bearer). Attribution headers mirror the opencode provider profile."""
    try:
        from hermes_cli import __version__ as _v
    except Exception:
        _v = "0"
    return {
        "Authorization": "",
        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
        "X-Title": "Hermes Agent",
        "User-Agent": f"HermesAgent/{_v}"}


def _fetch_opencode_free_models(
    timeout: float = 8.0, *, force_refresh: bool = False) -> Optional[list[str]]:
    """Live keyless OpenCode Free catalog from the Zen relay, filtered to the anonymous-servable
    ``*-free`` tier minus known keyed twins (Go ``ox-alpha-free`` is KEYED despite the suffix) — the
    same membership criterion ``opencode_zen_free_runtime`` routes on."""
    from hermes_cli.urllib_security import open_credentialed_url

    now = time.time()
    memo = _opencode_free_live_memo
    if not force_refresh and memo is not None and now - memo[0] < _OPENCODE_FREE_LIVE_MEMO_TTL:
        return list(memo[1]) if memo[1] else None

    req = urllib.request.Request(f"{_OPENCODE_ZEN_FREE_BASE_URL.rstrip('/')}/models")
    req.add_header("Accept", "application/json")
    for k, v in opencode_zen_free_headers().items():
        if k.lower() != "authorization":  # never send a bearer keylessly
            req.add_header(k, v)
    try:
        with open_credentialed_url(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        items = data if isinstance(data, list) else data.get("data", [])
    except Exception:
        _set_opencode_free_live_memo(None)
        return None
    live_free = [
        m["id"] for m in items
        if isinstance(m, dict) and isinstance(m.get("id"), str)
        and m["id"].lower().endswith("-free") and m["id"].lower() not in _OPENCODE_FREE_KEYED_SUFFIX_MODELS
    ]
    result = live_free or None
    _set_opencode_free_live_memo(result)
    return result


def _set_opencode_free_live_memo(ids: Optional[list[str]]) -> None:
    global _opencode_free_live_memo
    _opencode_free_live_memo = (time.time(), list(ids) if ids else None)


def _opencode_free_known_model_slugs() -> set[str]:
    """Lowercased keyless free-tier slugs known right now WITHOUT network I/O: static floor ∪ live
    memo ∪ SWR disk-cache entry. The ``opencode_zen_free_runtime`` healing path runs during model
    resolution and must never block on a fetch."""
    known = {m.lower() for m in _PROVIDER_MODELS.get("opencode-free", [])}
    memo = _opencode_free_live_memo
    if memo is not None and memo[1]:
        known.update(m.lower() for m in memo[1])
    try:
        entry = _load_provider_models_cache().get("opencode-free") or {}
        known.update(str(m).lower() for m in entry.get("models", []) or [])
    except Exception:
        pass
    return known


def opencode_zen_free_runtime(provider_id: Optional[str], model_id: Optional[str]) -> Optional[dict]:
    """Keyless runtime entry for an OpenCode Zen free-tier model, or None. Fires when ``provider_id``
    is ``opencode-free`` (EVERY model on it routes anonymously) or when any other OpenCode-family
    provider selected a model in the known keyless catalog (static floor ∪ cached live catalog —
    never a blocking fetch), healing a free-model pick made under Zen/Go whose keys the free tier
    rejects."""
    family = opencode_provider_family(provider_id)
    if family is None:
        return None
    normalized = normalize_opencode_model_id(provider_id, model_id)
    if family != "opencode-free" and normalized.strip().lower() not in _opencode_free_known_model_slugs():
        return None
    api_mode = opencode_model_api_mode("opencode-zen", normalized)
    base_url = normalize_opencode_base_url("opencode-zen", api_mode, _OPENCODE_ZEN_FREE_BASE_URL)
    return {
        "provider": family,
        "api_mode": api_mode,
        "base_url": base_url,
        "api_key": OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER,
        "default_headers": opencode_zen_free_headers(),
        "source": "opencode-zen-free-keyless"}


# Per-family (model-id prefix → api_mode) routing from OpenCode's published Zen/Go endpoint
# tables, checked in order. GPT/Codex/Grok and Muse Spark use /v1/responses (Muse Spark 503s on
# chat/completions); Claude (Zen) and MiniMax (Go) use /v1/messages, as do Qwen models on both
# relays; everything else falls through to /v1/chat/completions.
_OPENCODE_API_MODE_PREFIXES: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "opencode-go": (
        (("gpt-", "grok-", "muse-spark"), "codex_responses"),
        (("minimax-", "qwen"), "anthropic_messages")),
    "opencode-zen": (
        (("claude-",), "anthropic_messages"), (("gpt-", "grok-", "muse-spark"), "codex_responses"),
        (("qwen",), "anthropic_messages"))}


def opencode_model_api_mode(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Determine the API mode for an OpenCode Zen / Go model (see ``_OPENCODE_API_MODE_PREFIXES``)."""
    family = opencode_provider_family(provider_id)
    if family == "opencode-free":  # the free tier lives on the Zen relay → Zen's routing
        family = "opencode-zen"
    normalized = normalize_opencode_model_id(provider_id, model_id).lower()
    if normalized:
        for prefixes, mode in _OPENCODE_API_MODE_PREFIXES.get(family or "", ()):
            if normalized.startswith(prefixes):
                return mode
    return "chat_completions"


def normalize_opencode_base_url(
    provider_id: Optional[str], api_mode: Optional[str], base_url: Optional[str]) -> str:
    """Normalize an OpenCode Zen / Go base URL for the API mode. Must be SYMMETRIC: the anthropic-
    stripped URL gets persisted to ``model.base_url`` after switching into an anthropic-routed model,
    and chat/codex modes heal it by re-adding ``/v1`` — but only on opencode.ai hosts, so custom
    ``OPENCODE_*_BASE_URL`` proxies are left alone."""
    url = str(base_url or "").strip().rstrip("/")
    if not url or opencode_provider_family(provider_id) is None:
        return url
    if api_mode == "anthropic_messages":
        return re.sub(r"/v1$", "", url)
    if url.endswith("/v1"):
        return url
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        host = ""
    return url + "/v1" if host == "opencode.ai" or host.endswith(".opencode.ai") else url


def github_model_reasoning_efforts(
    model_id: Optional[str], *, catalog: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None) -> list[str]:
    """Return supported reasoning-effort levels for a Copilot-visible model."""
    normalized = normalize_copilot_model_id(model_id, catalog=catalog, api_key=api_key)
    if not normalized:
        return []

    if catalog is None and api_key:
        catalog = fetch_github_model_catalog(api_key=api_key)
    catalog_entry = next((item for item in catalog if item.get("id") == normalized), None) if catalog else None
    if catalog_entry is not None:
        capabilities = catalog_entry.get("capabilities")
        if isinstance(capabilities, dict):
            # Structured catalog: the advertised list is authoritative (empty when absent).
            supports = capabilities.get("supports")
            efforts = supports.get("reasoning_effort") if isinstance(supports, dict) else None
            if not isinstance(efforts, list):
                return []
            return list(dict.fromkeys(e for effort in efforts if (e := str(effort).strip().lower())))
        # Legacy list-shaped capabilities: only a "reasoning" tag unlocks the pattern defaults.
        if "reasoning" not in {str(c).strip().lower() for c in catalog_entry.get("capabilities", [])}:
            return []
    return _github_reasoning_efforts_for_model_id(str(model_id or normalized))


def _probe_result(
    models, probed_url, resolved_base_url, suggested_base_url=None, used_fallback=False
) -> dict[str, Any]:
    return {
        "models": models,
        "probed_url": probed_url,
        "resolved_base_url": resolved_base_url,
        "suggested_base_url": suggested_base_url,
        "used_fallback": used_fallback}


def probe_api_models(
    api_key: Optional[str], base_url: Optional[str], timeout: float = 5.0,
    api_mode: Optional[str] = None, request_headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Probe a ``/models`` endpoint with light URL heuristics (``base`` then ``base±/v1``).
    ``anthropic_messages`` mode sends ``x-api-key`` + ``anthropic-version`` instead of a bearer; the
    ``data[].id`` response shape is identical. ``models`` is None when no candidate answered."""
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        return _probe_result(None, None, "")
    if _is_github_models_base_url(normalized):
        models = _fetch_github_models(api_key=api_key, timeout=timeout)
        return _probe_result(models, COPILOT_MODELS_URL, COPILOT_BASE_URL)

    alternate_base = normalized[:-3].rstrip("/") if normalized.endswith("/v1") else normalized + "/v1"
    candidates: list[tuple[str, bool]] = [(normalized, False)]
    if alternate_base and alternate_base != normalized:
        candidates.append((alternate_base, True))

    tried: list[str] = []
    headers: dict[str, str] = {"User-Agent": _HERMES_USER_AGENT}
    if urllib.parse.urlparse(normalized).hostname == "generativelanguage.googleapis.com":
        headers["X-Goog-Api-Client"] = f"hermes-agent/{_HERMES_VERSION}"
    if api_key and api_mode == "anthropic_messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if normalized.startswith(COPILOT_BASE_URL):
        headers.update(copilot_default_headers())
    if isinstance(request_headers, dict):
        # Per-provider custom headers can contain secrets: merge last so endpoint config wins; never log.
        from hermes_cli.config import normalize_extra_headers

        headers.update(normalize_extra_headers(request_headers))

    # Only thread ssl_context when a per-provider TLS override applies; public endpoints keep the
    # original 2-arg call so existing call-seam mocks stay valid.
    _open_kwargs: dict[str, Any] = {}
    _ssl_context = _custom_provider_ssl_context(normalized)
    if _ssl_context is not None:
        _open_kwargs["ssl_context"] = _ssl_context
    for candidate_base, is_fallback in candidates:
        url = candidate_base.rstrip("/") + "/models"
        tried.append(url)
        try:
            data = _get_json(url, timeout=timeout, headers=headers, **_open_kwargs)
        except Exception:
            continue
        return _probe_result(
            [m.get("id", "") for m in data.get("data", [])], url, candidate_base.rstrip("/"),
            alternate_base if alternate_base != candidate_base else normalized, is_fallback)
    return _probe_result(
        None, tried[0] if tried else normalized.rstrip("/") + "/models", normalized,
        alternate_base if alternate_base != normalized else None)


# Legacy id-regex filter for items with no surface tag; unreachable (deletable) once every catalog
# entry carries an explicit ``chat``/``embed``/``image-gen``/``tts``/``stt`` tag.
_DEEPINFRA_EXCLUDE_RE = re.compile(
    r"(?i)(embed|rerank|whisper|stable-diffusion|flux|sdxl|"
    r"tts|bark|speech|image-gen|clip|vit-|dpt-)")

# Surface tags say *what kind of model* this is. Absent all of them, the tags array only carries
# capability tags (``reasoning``, ``vision``, …) and the chat surface falls back to id-regex inference.
_DEEPINFRA_SURFACE_TAGS: frozenset[str] = frozenset({
    "chat", "embed", "image-gen", "tts", "stt", "video-gen"})

_DEEPINFRA_DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
_DEEPINFRA_MODELS_QUERY = "filter=true&sort_by=hermes"

# Full tagged catalog keyed by base URL; every surface filter reads it so one round-trip serves all.
_deepinfra_catalog_cache: dict[str, list[dict]] = {}

# Negative cache (monotonic time of the last failed fetch per base URL) so an unreachable catalog
# doesn't make every surface helper eat the full timeout in turn. Short TTL so connectivity recovers.
_deepinfra_catalog_neg_cache: dict[str, float] = {}
_DEEPINFRA_CATALOG_NEG_TTL = 60.0  # seconds


def _deepinfra_catalog_url() -> tuple[str, str]:
    """Return ``(cache_key, full_url)`` for the DeepInfra catalog endpoint."""
    base = os.getenv("DEEPINFRA_BASE_URL", "").strip() or _DEEPINFRA_DEFAULT_BASE_URL
    cache_key = base.rstrip("/")
    return cache_key, f"{cache_key}/models?{_DEEPINFRA_MODELS_QUERY}"


def _fetch_deepinfra_catalog(
    *, timeout: float = 5.0, force_refresh: bool = False) -> Optional[list[dict]]:
    """Raw DeepInfra catalog list (chat, embed, image-gen, TTS, STT in one response), cached per base
    URL. A Bearer token is attached when available so user-scoped catalogs (private fine-tunes) show."""
    cache_key, url = _deepinfra_catalog_url()
    if not force_refresh:
        if cache_key in _deepinfra_catalog_cache:
            return _deepinfra_catalog_cache[cache_key]
        last_fail = _deepinfra_catalog_neg_cache.get(cache_key)
        if last_fail is not None and (time.monotonic() - last_fail) < _DEEPINFRA_CATALOG_NEG_TTL:
            return None

    headers: dict[str, str] = {"User-Agent": _HERMES_USER_AGENT}
    api_key = os.getenv("DEEPINFRA_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        payload = _get_json(url, timeout=timeout, headers=headers)
    except Exception:
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        _deepinfra_catalog_neg_cache[cache_key] = time.monotonic()
        return None
    _deepinfra_catalog_cache[cache_key] = data
    _deepinfra_catalog_neg_cache.pop(cache_key, None)
    return data


def _fetch_deepinfra_models_by_tag(
    tag: str, *, timeout: float = 5.0, force_refresh: bool = False) -> Optional[list[dict]]:
    """DeepInfra ``{"id", "metadata"}`` items whose ``metadata.tags`` includes *tag*. Items with no
    surface tag fall through to the legacy id-regex exclusion (chat surface only — embed/image-gen/
    tts/stt cannot be inferred from an id). ``None`` on network failure."""
    data = _fetch_deepinfra_catalog(timeout=timeout, force_refresh=force_refresh)
    if data is None:
        return None
    matched: list[dict] = []
    for item in data:
        mid = item.get("id")
        raw_metadata = item.get("metadata")
        if not mid or raw_metadata is None:  # metadata None = listed-but-not-served stub
            continue
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_tags = metadata.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        if any(t in _DEEPINFRA_SURFACE_TAGS for t in tags):
            hit = tag in tags
        else:
            hit = tag == "chat" and not _DEEPINFRA_EXCLUDE_RE.search(mid)
        if hit:
            matched.append({"id": mid, "metadata": metadata})
    return matched


def _fetch_deepinfra_models(
    timeout: float = 5.0, *, force_refresh: bool = False) -> Optional[list[str]]:
    """DeepInfra chat-model ids (string-list contract for :func:`provider_model_ids`); ``None`` on
    network failure or when no chat-tagged id exists."""
    items = _fetch_deepinfra_models_by_tag("chat", timeout=timeout, force_refresh=force_refresh)
    return ([item["id"] for item in items] or None) if items is not None else None


def deepinfra_model_ids(tag: str, *, force_refresh: bool = False) -> list[str]:
    """Return DeepInfra model ids carrying surface *tag* (``[]`` on failure)."""
    items = _fetch_deepinfra_models_by_tag(tag, force_refresh=force_refresh)
    return [item["id"] for item in items] if items else []


def deepinfra_base_url(section: Optional[dict] = None) -> str:
    """DeepInfra base URL: config-section ``base_url`` → ``DEEPINFRA_BASE_URL`` env → default; stripped."""
    candidate = section.get("base_url") if isinstance(section, dict) else None
    value = candidate or os.getenv("DEEPINFRA_BASE_URL") or _DEEPINFRA_DEFAULT_BASE_URL
    return str(value).strip().rstrip("/")


def _fetch_ai_gateway_models(timeout: float = 5.0) -> Optional[list[str]]:
    """Fetch available language models with tool-use from AI Gateway."""
    api_key = os.getenv("AI_GATEWAY_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = os.getenv("AI_GATEWAY_BASE_URL", "").strip()
    if not base_url:
        from hermes_constants import AI_GATEWAY_BASE_URL
        base_url = AI_GATEWAY_BASE_URL

    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": _HERMES_USER_AGENT}
    try:
        url = base_url.rstrip("/") + "/models"
        data = _get_json(url, timeout=timeout, headers=headers, opener=urllib.request.urlopen)
        return [
            m["id"] for m in data.get("data", [])
            if m.get("id") and m.get("type") == "language" and "tool-use" in (m.get("tags") or [])]
    except Exception:
        return None


def fetch_api_models(
    api_key: Optional[str], base_url: Optional[str], timeout: float = 5.0,
    api_mode: Optional[str] = None, headers: Optional[dict[str, str]] = None,
) -> Optional[list[str]]:
    """Fetch the list of available model IDs from the provider's ``/models`` endpoint."""
    result = probe_api_models(api_key, base_url, timeout=timeout, api_mode=api_mode, request_headers=headers)
    return result.get("models")


def _custom_endpoint_fingerprint(
    api_key: Optional[str], api_mode: Optional[str], headers: Optional[dict[str, str]]) -> str:
    """Custom endpoints have no ``PROVIDER_REGISTRY`` slug, so hash exactly what callers pass to
    :func:`fetch_api_models`: a rotated ``api_key``, changed ``api_mode`` or edited ``extra_headers``
    each bust the cache entry. blake2b for the same CodeQL rationale as ``_credential_fingerprint``."""
    import hashlib

    blob = "|".join((api_key or "", api_mode or "", json.dumps(headers or {}, sort_keys=True)))
    return hashlib.blake2b(blob.encode("utf-8", errors="replace"), digest_size=8).hexdigest()


def _cache_entry_valid(
    entry: Any, fp: str, *, allow_empty: bool = False) -> "TypeGuard[dict[str, Any]]":
    """Well-formed cache row for fingerprint *fp*. Requires a numeric ``at`` so corrupt disk state
    degrades to a cache miss instead of raising; empty model lists are valid only when the caller
    opts into an authoritative empty catalog."""
    return (
        isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and (allow_empty or bool(entry["models"]))
        and isinstance(entry.get("at"), (int, float))
        and not isinstance(entry.get("at"), bool))


def cached_fetch_api_models(
    api_key: Optional[str], base_url: Optional[str], *, timeout: float = 5.0,
    api_mode: Optional[str] = None, headers: Optional[dict[str, str]] = None,
    force_refresh: bool = False, cache_only: bool = False,
    ttl_seconds: int = _PROVIDER_MODELS_CACHE_TTL) -> Optional[list[str]]:
    """Disk-cached :func:`fetch_api_models` for custom endpoints. ``cache_only`` callers (GUI picker
    opens that must not block on a stopped local endpoint) still get a warm catalog instead of
    collapsing to the config-declared subset."""
    def _live():
        return fetch_api_models(api_key, base_url, timeout=timeout, api_mode=api_mode, headers=headers)

    normalized_url = str(base_url or "").strip().rstrip("/").lower()
    if not normalized_url:  # nothing to key the cache on
        return None if cache_only else _live()

    cache_key = f"custom:{normalized_url}"
    fp = _custom_endpoint_fingerprint(api_key, api_mode, headers)
    cache = _load_provider_models_cache()
    entry = cache.get(cache_key)
    now = time.time()
    valid = not force_refresh and _cache_entry_valid(entry, fp)

    if cache_only:
        # Same trust window as the SWR tier below, minus the revalidation.
        return list(entry["models"]) if valid and now - entry["at"] < _PROVIDER_MODELS_STALE_SERVE_MAX else None

    if valid:
        age = now - entry["at"]
        if age < ttl_seconds:
            return list(entry["models"])
        if age < _PROVIDER_MODELS_STALE_SERVE_MAX:
            # Stale-while-revalidate: serve now, refresh off-thread for the next open.
            def _refresh_custom():
                live = _live()
                return _cache_entry(fp, live) if live else None

            _spawn_swr_refresh(cache_key, _refresh_custom)
            return list(entry["models"])

    live = _live()
    if live:
        _store_cache_entry(cache_key, _cache_entry(fp, live, now), cache)
        return list(live)
    # Live returned nothing (offline, timeout, auth hiccup): a stale same-fingerprint entry beats it.
    if _cache_entry_valid(entry, fp):
        return list(entry["models"])
    return live


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import NamedTuple  # noqa: F401,E402
from difflib import get_close_matches  # noqa: F401,E402
import http.client  # noqa: F401,E402

def is_nous_free_tier(account_info: dict[str, Any]) -> bool:
    """Return True if the account info indicates a free (unpaid) tier.

    Prefer the Portal's explicit ``paid_service_access.allowed`` entitlement
    decision.  Legacy payloads fall back to ``subscription.monthly_charge == 0``.
    Returns False when both signals are missing or unparseable.
    """
    paid_access = account_info.get("paid_service_access")
    if isinstance(paid_access, dict):
        allowed = paid_access.get("allowed")
        if isinstance(allowed, bool):
            return not allowed
        paid = paid_access.get("paid_access")
        if isinstance(paid, bool):
            return not paid

    sub = account_info.get("subscription")
    if not isinstance(sub, dict):
        return False
    charge = sub.get("monthly_charge")
    if charge is None:
        return False
    try:
        return float(charge) == 0
    except (TypeError, ValueError):
        return False

_OPENCODE_KEYLESS_EXTRA_SLUGS = frozenset({"big-pickle"})

def is_opencode_zen_free_model(model_id: Optional[str]) -> bool:
    """True when ``model_id`` is an OpenCode Zen free-tier slug.

    Matches the ``*-free`` suffix plus the known unsuffixed free slugs
    (``big-pickle``). Tolerates provider-prefixed ids
    (``opencode-zen/x-preview-f-free``). The Go catalog serves no free
    models (verified 2026-08-21), so this identifies the Zen free tier
    across the OpenCode family.
    """
    bare = str(model_id or "").strip().rsplit("/", 1)[-1].lower()
    if not bare:
        return False
    return bare.endswith("-free") or bare in _OPENCODE_KEYLESS_EXTRA_SLUGS


_PLUGIN_COMPAT_LAZY = {
    'LMStudioLoadResult': ('hermes_cli.models_local', 'LMStudioLoadResult'),
    'PROVIDER_GROUPS': ('hermes_cli.models_catalog_static', 'PROVIDER_GROUPS'),
    'ProviderEntry': ('hermes_cli.models_catalog_static', 'ProviderEntry'),
    'atomic_json_write': ('utils', 'atomic_json_write'),
    'base_url_host_matches': ('utils', 'base_url_host_matches'),
    'compute_sale_discount': ('hermes_cli.models_pricing', 'compute_sale_discount'),
    'ensure_lmstudio_model_loaded': ('hermes_cli.models_local', 'ensure_lmstudio_model_loaded'),
    'fetch_ai_gateway_pricing': ('hermes_cli.models_pricing', 'fetch_ai_gateway_pricing'),
    'fetch_lmstudio_models': ('hermes_cli.models_local', 'fetch_lmstudio_models'),
    'fetch_models_with_pricing': ('hermes_cli.models_pricing', 'fetch_models_with_pricing'),
    'fetch_ollama_local_models': ('hermes_cli.models_local', 'fetch_ollama_local_models'),
    'get_cached_nous_inference_base_url': ('hermes_cli.models_pricing', 'get_cached_nous_inference_base_url'),
    'get_pricing_for_provider': ('hermes_cli.models_pricing', 'get_pricing_for_provider'),
    'group_providers': ('hermes_cli.models_catalog_static', 'group_providers'),
    'lmstudio_model_reasoning_options': ('hermes_cli.models_local', 'lmstudio_model_reasoning_options'),
    'nous_catalog_url': ('hermes_cli.models_reasoning_caps', 'nous_catalog_url'),
    'nous_model_reasoning_capabilities': ('hermes_cli.models_reasoning_caps', 'nous_model_reasoning_capabilities'),
    'nous_policy_allowed_ids': ('hermes_cli.models_pricing', 'nous_policy_allowed_ids'),
    'ollama_model_supports_thinking': ('hermes_cli.models_local', 'ollama_model_supports_thinking'),
    'openrouter_model_reasoning_capabilities': ('hermes_cli.models_reasoning_caps', 'openrouter_model_reasoning_capabilities'),
    'parse_openrouter_reasoning_capabilities': ('hermes_cli.models_reasoning_caps', 'parse_openrouter_reasoning_capabilities'),
    'peek_cached_pricing': ('hermes_cli.models_pricing', 'peek_cached_pricing'),
    'pricing_cache_scope': ('hermes_cli.models_pricing', 'pricing_cache_scope'),
    'probe_lmstudio_models': ('hermes_cli.models_local', 'probe_lmstudio_models'),
    'probe_ollama_local_models': ('hermes_cli.models_local', 'probe_ollama_local_models'),
    'provider_group_for_slug': ('hermes_cli.models_catalog_static', 'provider_group_for_slug'),
    'refresh_reasoning_caps_async': ('hermes_cli.models_reasoning_caps', 'refresh_reasoning_caps_async'),
    'restrict_to_nous_policy': ('hermes_cli.models_pricing', 'restrict_to_nous_policy'),
    'should_use_ollama_native_catalog': ('hermes_cli.models_local', 'should_use_ollama_native_catalog'),
    'url_origin': ('hermes_cli.urllib_security', 'url_origin'),
    'validate_requested_model': ('hermes_cli.models_validate', 'validate_requested_model'),
    'warm_nous_reasoning_caps_async': ('hermes_cli.models_reasoning_caps', 'warm_nous_reasoning_caps_async'),
    'warm_openrouter_reasoning_caps_async': ('hermes_cli.models_reasoning_caps', 'warm_openrouter_reasoning_caps_async'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

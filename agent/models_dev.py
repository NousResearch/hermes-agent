"""Models.dev registry integration — primary database for providers and models.

Resolution: in-memory cache (fresh, or stale served while one background daemon
thread refreshes) → disk cache (~/.hermes/models_dev_cache.json, any age) →
network only when no cache exists. Failed refreshes back off 5 min process-wide.
Refreshes use ETag conditional GET when a servable registry is held. Hot paths
pass ``allow_network=False`` and never do I/O. A corrupt/empty disk cache is
quarantined, never served as ``{}``. ``models_dev.url`` in config.yaml = mirror."""

import contextlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_json_write, atomic_write_text

from hermes_constants import openrouter_variant_base
from models import ModelRef
from models.metadata import (
    ModelInfo,
    ModelMetadata,
    ProviderInfo,
    UNKNOWN_MODEL_BASE,
    builtin_model_metadata,
    dict_or_empty,
    extract_context as _extract_context,
    extract_limit as _extract_limit,
    merge_catalog_entry_with_override,
    model_info_from_entry,
    model_metadata_from_entry,
    override_int as _override_int,
    provider_info_from_entry,
    vision_marker_metadata,
)

from models.metadata.interpretation import _OVERRIDE_WARNED_KEYS
import models.models_dev_cache as models_dev_cache


import requests

logger = logging.getLogger(__name__)
MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_CACHE_TTL = 4 * 3600  # 4 hours — ETag conditional GET makes refresh cheap
_MODELS_DEV_RETRY_DELAY = 300  # 5 minutes after a failed refresh
# In-memory cache
_models_dev_cache: Dict[str, Any] = {}
_models_dev_cache_time: float = 0
_models_dev_retry_after: float = 0
_models_dev_fetch_lock = threading.Lock()
_models_dev_refresh_lock = threading.Lock()
_models_dev_refresh_in_flight = False


# Hermes provider names → models.dev provider IDs
PROVIDER_TO_MODELS_DEV: Dict[str, str] = {
    "openrouter": "openrouter", "novita": "novita-ai", "anthropic": "anthropic",
    "openai": "openai", "openai-api": "openai", "openai-codex": "openai", "zai": "zai",
    "kimi": "kimi-for-coding", "kimi-coding": "kimi-for-coding",
    "moonshot": "kimi-for-coding", "stepfun": "stepfun",
    "kimi-coding-cn": "kimi-for-coding", "minimax": "minimax",
    "minimax-oauth": "minimax", "minimax-cn": "minimax-cn", "deepseek": "deepseek",
    "alibaba": "alibaba", "qwen-oauth": "alibaba", "copilot": "github-copilot",
    "ai-gateway": "vercel", "opencode-zen": "opencode",
    "opencode-go": "opencode-go",
    "kilocode": "kilo", "fireworks": "fireworks-ai",
    "huggingface": "huggingface", "gemini": "google", "google": "google",
    "xai": "xai",
    "xai-oauth": "xai",  # OAuth is a transport path for the same xAI catalog
    "xiaomi": "xiaomi", "nvidia": "nvidia",
    # Meta Model API (Muse Spark, api.meta.ai): models.dev keys it "meta", the
    # Hermes provider is "meta-ai"; both aliases are needed or muse-spark-*
    # falls back to the generic 256K default instead of its true 1M window.
    "meta-ai": "meta", "meta": "meta", "groq": "groq", "mistral": "mistral",
    "togetherai": "togetherai", "perplexity": "perplexity", "cohere": "cohere",
    "ollama-cloud": "ollama-cloud",
}
# Reverse mapping: models.dev id → Hermes ids (built lazily; many-to-one).
_MODELS_DEV_TO_PROVIDER: Optional[Dict[str, List[str]]] = None


def _models_dev_to_hermes_ids(mdev_id: str) -> List[str]:
    """Return the Hermes provider ids that map to *mdev_id* (may be [])."""
    global _MODELS_DEV_TO_PROVIDER
    if _MODELS_DEV_TO_PROVIDER is None:
        _MODELS_DEV_TO_PROVIDER = {}
        for hermes_id, mapped in PROVIDER_TO_MODELS_DEV.items():
            _MODELS_DEV_TO_PROVIDER.setdefault(mapped, []).append(hermes_id)
    return _MODELS_DEV_TO_PROVIDER.get(mdev_id, [])


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _configured_catalog_provider(
    provider: str, *, config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """``catalog_provider`` declared on a custom provider's ``providers.<name>`` row (or legacy
    ``custom_providers[]`` entry): the catalogued vendor whose models it resells. None when unset."""
    name = (provider or "").strip()
    if name.lower().startswith("custom:"):
        name = name[len("custom:"):]
    if not name or name in PROVIDER_TO_MODELS_DEV:
        return None
    provider_config = (
        _cfg_get("providers", name, default=None, config=config)
        if config is not None
        else _cfg_get("providers", name, default=None)
    )
    alias = _dict_or_empty(provider_config).get("catalog_provider")
    if not alias:
        legacy = (
            _cfg_get("custom_providers", default=None, config=config)
            if config is not None
            else _cfg_get("custom_providers", default=None)
        )
        alias = next((e.get("catalog_provider") for e in (legacy if isinstance(legacy, list) else [])
                      if isinstance(e, dict) and str(e.get("name") or "").strip() == name), None)
    alias = str(alias or "").strip()
    return alias or None


def _models_dev_id(
    provider: str, *, config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """models.dev provider id for a Hermes provider id, or None. A custom provider reaches the
    catalog only through its configured ``catalog_provider`` alias (#112649)."""
    key = (provider or "").strip()
    mdev_id = PROVIDER_TO_MODELS_DEV.get(key)
    if mdev_id is None:
        alias = _configured_catalog_provider(key, config=config)
        mdev_id = PROVIDER_TO_MODELS_DEV.get(alias, alias) if alias else None
        if mdev_id is not None and mdev_id not in PROVIDER_TO_MODELS_DEV.values() \
                and mdev_id not in fetch_models_dev(allow_network=False):
            # A mistyped alias must not leak into ModelInfo.provider_id; the row stays on its own slug.
            if (key, alias) not in _UNKNOWN_CATALOG_PROVIDER_WARNED:
                _UNKNOWN_CATALOG_PROVIDER_WARNED.add((key, alias))
                logger.warning("providers.%s: catalog_provider %r is neither a Hermes provider id nor a "
                               "models.dev id; ignoring", key, alias)
            mdev_id = None
    return mdev_id


_UNKNOWN_CATALOG_PROVIDER_WARNED: set = set()  # (provider, alias) warned once per process


def _cfg_get(*keys: str, default: Any, config: Optional[Dict[str, Any]] = None) -> Any:
    """``cfg_get`` over the read-only config; *default* on any failure."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        return cfg_get(config if config is not None else load_config_readonly(), *keys, default=default)
    except Exception:
        return default








def _quietly(what: str, fn, default=None):
    """Run *fn*; on any exception log ``"Failed to <what>: %s"`` at debug and return *default*."""
    try:
        return fn()
    except Exception as e:
        logger.debug("Failed to %s: %s", what, e)
        return default








def _get_models_dev_url() -> str:
    """The models.dev API URL, honoring the ``models_dev.url`` config override."""
    url = _cfg_get("models_dev", "url", default="")
    # Module global (not a captured constant) so patching MODELS_DEV_URL works.
    return url.strip() if isinstance(url, str) and url.strip() else MODELS_DEV_URL












# Network refresh: all state mutation happens under _models_dev_fetch_lock.
class _NotModified(Exception):
    """Server returned 304 Not Modified — existing cache is still valid."""


def _fetch_models_dev_from_network(*, conditional: bool = False) -> Tuple[Dict[str, Any], str]:
    """Fetch the live registry; returns ``(registry, etag)`` (etag "" if none). Raises on network
    errors and on an empty/invalid payload. ``conditional`` sends ``If-None-Match`` with the sidecar's
    ETag and raises ``_NotModified`` on 304 — pass True ONLY while holding ``_models_dev_fetch_lock``
    AND a servable registry, or a 304 leaves the process with no data."""
    headers: Dict[str, str] = {}
    if conditional and (etag := models_dev_cache.load_models_dev_etag()):
        headers["If-None-Match"] = etag
    # (connect, read): 5 s connect fails fast on blackholed hosts; 10 s read tolerates a slow registry.
    response = requests.get(_get_models_dev_url(), headers=headers, timeout=(5, 10))
    if response.status_code == 304:
        raise _NotModified()
    response.raise_for_status()
    data = response.json()
    if not models_dev_cache.valid_models_dev_registry(data):
        raise ValueError("models.dev returned an empty or invalid registry")
    return data, response.headers.get("ETag", "")


def _mark_stale_cache_grace() -> None:
    """Give stale cache data a 5-minute in-memory grace before retrying refresh. Only ever moves the
    timestamp forward, so a background refresh that completed meanwhile keeps its fresh stamp."""
    global _models_dev_cache_time
    _models_dev_cache_time = max(_models_dev_cache_time, time.time() - _MODELS_DEV_CACHE_TTL + _MODELS_DEV_RETRY_DELAY)


def _serve_stale(msg: str, *args: Any) -> Dict[str, Any]:
    """Arm the grace window, kick off a background refresh, return the held cache."""
    _mark_stale_cache_grace()
    _start_background_refresh_models_dev()
    logger.debug(msg, *args)
    return _models_dev_cache


def _commit_registry(data: Dict[str, Any], *, etag: str = "", where: str) -> None:
    """Persist a fetched registry: disk + in-mem + clear backoff. Callers hold ``_models_dev_fetch_lock``
    so a failing refresh on one path can never stomp state a succeeding refresh just committed."""
    global _models_dev_cache, _models_dev_cache_time, _models_dev_retry_after
    models_dev_cache.save_models_dev_disk_cache(data, etag)
    _models_dev_cache = data
    _models_dev_cache_time = time.time()
    _models_dev_retry_after = 0
    logger.debug(
        "Refreshed models.dev registry (%s): %d providers, %d total models", where, len(data),
        sum(len(p.get("models", {})) for p in data.values() if isinstance(p, dict)),
    )


def _confirm_cache_not_modified(*, where: str) -> None:
    """After a 304: clear backoff and re-mark the held cache fresh (disk is untouched — only the
    freshness marker advances). Caller holds the lock."""
    global _models_dev_cache_time, _models_dev_retry_after
    if not _models_dev_cache:
        # Should be unreachable (conditional GETs require a servable cache) but previously caused a
        # permanent empty-registry loop: drop the sidecar and arm the backoff rather than marking {} "fresh".
        models_dev_cache.clear_models_dev_etag()
        _models_dev_retry_after = time.time() + _MODELS_DEV_RETRY_DELAY
        logger.warning("models.dev returned 304 but no cached registry is held (%s); "
                       "cleared ETag sidecar, will refetch unconditionally", where)
        return
    _models_dev_cache_time = time.time()
    _models_dev_retry_after = 0
    logger.debug("models.dev registry unchanged (304 Not Modified, %s); cache re-confirmed fresh", where)


def _note_refresh_failure(exc: Exception, *, where: str) -> None:
    """Arm the process-wide 5-minute backoff. Caller holds the lock."""
    global _models_dev_retry_after
    _models_dev_retry_after = time.time() + _MODELS_DEV_RETRY_DELAY
    logger.debug("models.dev refresh failed (%s); retry suppressed for %ds: %s", where, _MODELS_DEV_RETRY_DELAY, exc)


def _refresh_locked(where: str) -> Optional[Dict[str, Any]]:
    """One conditional fetch + state update; caller holds ``_models_dev_fetch_lock``. Returns the
    registry to serve, or None when the fetch failed (backoff armed)."""
    try:
        data, etag = _fetch_models_dev_from_network(conditional=bool(_models_dev_cache))
        _commit_registry(data, etag=etag, where=where)
        return data
    except _NotModified:
        _confirm_cache_not_modified(where=where)
        return _models_dev_cache
    except Exception as e:
        _note_refresh_failure(e, where=where)
        return None


def _background_refresh_models_dev() -> None:
    """Best-effort refresh after serving stale cache data."""
    global _models_dev_refresh_in_flight
    try:
        # Fetch INSIDE the lock, symmetric with the foreground path: the conditional-GET inputs (memory
        # cache + sidecar) can't be mutated mid-fetch by a concurrent force_refresh and the two paths
        # can't double-download. Hot-path callers never touch this lock.
        with _models_dev_fetch_lock:
            _refresh_locked("background")
    finally:
        with _models_dev_refresh_lock:
            _models_dev_refresh_in_flight = False


def _start_background_refresh_models_dev() -> None:
    """Start one daemon refresh worker if none is running and the failure backoff has elapsed."""
    global _models_dev_refresh_in_flight
    if time.time() < _models_dev_retry_after:
        return
    with _models_dev_refresh_lock:
        if _models_dev_refresh_in_flight:
            return
        _models_dev_refresh_in_flight = True
    thread = threading.Thread(target=_background_refresh_models_dev, name="models-dev-refresh", daemon=True)
    try:
        thread.start()
    except Exception as e:
        # Thread/fd exhaustion: clear the flag so refresh isn't disabled for the rest of the process.
        with _models_dev_refresh_lock:
            _models_dev_refresh_in_flight = False
        logger.debug("Failed to start models.dev refresh thread: %s", e)


def fetch_models_dev(force_refresh: bool = False, *, allow_network: bool = True) -> Dict[str, Any]:
    """Fetch the models.dev registry (dict keyed by provider ID; {} on failure). Cache hierarchy:
    fresh in-memory → stale in-memory (served now, refreshed in one background daemon thread — stale
    beats a foreground timeout) → disk of any age (stale triggers the same background refresh) →
    singleflight foreground fetch. A failed refresh suppresses automatic refreshes for 5 minutes.
    ``force_refresh=True`` bypasses the cache fast paths and the backoff, falling back to cached data
    only if the call fails. ``allow_network=False`` returns any memory/disk cache and never makes a request."""
    global _models_dev_cache, _models_dev_cache_time, _models_dev_retry_after
    if not allow_network:
        if not _models_dev_cache and (disk_data := models_dev_cache.load_models_dev_disk_cache()):
            _models_dev_cache = disk_data
            disk_age = models_dev_cache.models_dev_disk_cache_age_seconds()
            _models_dev_cache_time = time.time() - disk_age if disk_age is not None else 0
        return _models_dev_cache
    if not force_refresh:
        # Stage 1: fresh in-memory cache — the hot path, no I/O.
        if _models_dev_cache and (time.time() - _models_dev_cache_time) < _MODELS_DEV_CACHE_TTL:
            return _models_dev_cache
        # Stage 2: stale in-memory cache beats blocking on the network.
        if _models_dev_cache:
            return _serve_stale("Using stale in-memory models.dev cache; refreshing in background")
        # Stage 3: disk cache (cold-start only). A stale disk cache is deliberately usable so
        # resolution doesn't hang when models.dev is unreachable.
        disk_age = models_dev_cache.models_dev_disk_cache_age_seconds()
        if disk_age is not None and (disk_data := models_dev_cache.load_models_dev_disk_cache()):
            _models_dev_cache = disk_data
            if disk_age >= _MODELS_DEV_CACHE_TTL:
                return _serve_stale("Using stale models.dev disk cache (age=%.0fs); refreshing in background", disk_age)
            # Anchor the in-mem TTL to the file's age so an aging cache isn't extended by another full TTL.
            _models_dev_cache_time = time.time() - disk_age
            logger.debug("Loaded models.dev from fresh disk cache (%d providers, age=%.0fs)", len(disk_data), disk_age)
            return _models_dev_cache
        # Process-wide backoff: don't make every caller retry an unreachable endpoint while no usable cache exists.
        if time.time() < _models_dev_retry_after:
            return _models_dev_cache
    # Stage 4: singleflight foreground fetch. Recheck state under the lock — another caller may
    # have refreshed or armed the backoff while we waited.
    with _models_dev_fetch_lock:
        if not force_refresh and (_models_dev_cache or time.time() < _models_dev_retry_after):
            return _models_dev_cache
        # Cold force_refresh: stages 1-3 were skipped, so hydrate memory from disk first so the
        # conditional GET fires and a 304 can re-confirm it.
        if force_refresh and not _models_dev_cache and (disk := models_dev_cache.load_models_dev_disk_cache()):
            _models_dev_cache = disk
            _models_dev_cache_time = 0  # servable but not fresh
        served = _refresh_locked("foreground")
        if served is not None:
            return served
        # Stage 5: network failed — serve any stale memory/disk cache. Freshness stays expired;
        # the retry-after timestamp gates the next attempt.
        if not _models_dev_cache:
            _models_dev_cache = models_dev_cache.load_models_dev_disk_cache()
            _models_dev_cache_time = 0
            if _models_dev_cache:
                logger.debug("Loaded stale models.dev disk cache (%d providers)", len(_models_dev_cache))
        return _models_dev_cache


def _registry_provider(mdev_id: str, allow_network: bool) -> Optional[Dict[str, Any]]:
    """The raw models.dev provider entry, or None."""
    # Keep the zero-argument call on the allow_network path: dozens of test sites monkeypatch fetch_models_dev with zero-arg lambdas.
    registry = fetch_models_dev() if allow_network else fetch_models_dev(allow_network=False)
    provider_data = registry.get(mdev_id)
    return provider_data if isinstance(provider_data, dict) else None


def _registry_models(mdev_id: str, *, allow_network: bool) -> Optional[Dict[str, Any]]:
    """The ``models`` dict of a models.dev provider entry, or None."""
    provider_data = _registry_provider(mdev_id, allow_network)
    models = provider_data.get("models", {}) if provider_data is not None else None
    return models if isinstance(models, dict) else None


def _get_provider_models(
    provider: str, *, allow_network: bool = False, config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a Hermes provider ID to its models dict, or None if unknown.
    ``allow_network`` defaults to False — hot-path callers must never block."""
    mdev_id = _models_dev_id(provider, config=config)
    return _registry_models(mdev_id, allow_network=allow_network) if mdev_id else None


_OPENROUTER_CATALOG_PROVIDERS = frozenset({"openrouter"})


def _openrouter_catalog_lookup_base(provider: str, model: str) -> Optional[str]:
    """Return the base id to retry a catalog lookup with, or ``None``.

    OpenRouter's ``:nitro`` / ``:floor`` / ``:exacto`` / ``:online`` are
    request-time routing modifiers: they change which endpoint serves the
    request, never which model runs. ``/models`` and models.dev list only the
    base id, so a routed id must resolve to the base model's metadata.

    Scoped to OpenRouter so a genuine ``model:tag`` on another provider (an
    Ollama tag, a ``:cloud`` catalog key) is never rewritten.

    Deliberately excludes ``:free``, ``:batch``, ``:extended``, and
    ``:thinking``: those are REAL catalog SKUs with their own entries and
    their own — sometimes different — context windows. Stripping them would
    report a window LARGER than the model actually has. Their real entries
    are found by the exact/case-insensitive passes above, and a genuinely
    absent SKU must miss so ``model_overrides`` ``_default`` fill-gap
    semantics still apply.
    """
    if provider not in _OPENROUTER_CATALOG_PROVIDERS:
        return None
    return openrouter_variant_base(model)


def _iter_model_entries(
    models: Dict[str, Any], model: str, *, suffix_fallback: bool = True, provider: str = ""
):
    """Yield ``(model_id, entry)`` candidates: exact, case-insensitive, then (optionally)
    ``:cloud``/``-cloud`` suffixed forms. Suffix fallback: some providers (ollama-cloud) store
    ``kimi-k2.6:cloud`` while the live API returns the bare name; without it context lookup falls to
    stale OpenRouter metadata and trips the 64k minimum-context guard. Every consumer shares this
    order so a suffix-keyed catalog model counts as KNOWN for ``model_overrides`` fill-gap ``_default``.

    ``provider`` enables the OpenRouter routing-variant fallback as a LAST
    resort — after exact, case-insensitive, and ``:cloud`` matching — so a
    real catalog SKU always wins over its base.
    """
    for name in ([model] + [model + suffix for suffix in (":cloud", "-cloud")] if suffix_fallback else [model]):
        entry = models.get(name)
        if isinstance(entry, dict):
            yield name, entry
        name_lower = name.lower()
        for mid, mdata in models.items():
            if mid.lower() == name_lower and isinstance(mdata, dict):
                yield mid, mdata
    routed_base = _openrouter_catalog_lookup_base(provider, model)
    if routed_base is not None:
        # Recursion is bounded: the base never carries a recognized variant suffix,
        # and provider is cleared so the retry cannot loop.
        yield from _iter_model_entries(
            models, routed_base, suffix_fallback=suffix_fallback, provider=""
        )


def _find_model_entry(
    models: Dict[str, Any], model: str, provider: str = ""
) -> Optional[Dict[str, Any]]:
    """First catalog entry for *model* (exact, case-insensitive, suffix), or None."""
    return next((entry for _mid, entry in _iter_model_entries(models, model, provider=provider)), None)


def lookup_models_dev_context(provider: str, model: str, *, allow_network: bool = False) -> Optional[int]:
    """Context window in tokens for provider+model, or None if not found. An EXPLICIT ``model_overrides``
    entry wins over the catalog; ``_default`` fills the gap only when the catalog has no answer (the
    self-unblock path for wrong/missing context in models.dev). Catalog entries with context=0 are
    skipped in favour of later candidates. ``allow_network`` defaults to False — runs every turn.

    See #84482.
    """
    override_ctx = _override_context_window(provider, model)
    if override_ctx is not None:
        return override_ctx
    declared_ctx = _override_int(_provider_model_capabilities(provider, model), "context_window")
    if declared_ctx is not None:
        return declared_ctx
    models = _get_provider_models(provider, allow_network=allow_network)
    catalog_ctx = next((ctx for _mid, entry in _iter_model_entries(models, model, provider=provider) if (ctx := _extract_context(entry))), None) if models is not None else None
    return catalog_ctx if catalog_ctx is not None else _default_override_context(provider)


# Per-model overrides (config.yaml → model_overrides). Canonical schema (the ONLY key space consumers
# accept): context_window, supports_tools, supports_vision, supports_reasoning,
# model_family. ``<provider>.<model_id>`` is an explicit partial patch that always wins over the
# catalog. ``<provider>._default`` / top-level ``_default`` are FILL-GAP defaults: they apply ONLY to
# models the catalog does not know and never displace catalog data. Provider keys accept the Hermes
# or models.dev id; model ids match exactly, then case-insensitively (mirroring catalog lookup).
# Resolution semantics: 1. 2. See #84482, #8731.
# Interpretation lives in models.metadata; this module retains only override
# source resolution and catalogue acquisition.

def _load_model_overrides(*, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The ``model_overrides`` config section ({} on any failure). Deliberately not memoized:
    ``load_config_readonly()`` is already (mtime, size)-cached upstream, and an ``id(cfg)``-keyed
    layer can serve stale overrides after a reload when CPython reuses the dict address."""
    overrides = (
        _cfg_get("model_overrides", default={}, config=config)
        if config is not None
        else _cfg_get("model_overrides", default={})
    )
    return _dict_or_empty(overrides)


def _provider_override_section(provider: str, *, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Override section for *provider* (keyed by Hermes OR models.dev id), or None."""
    overrides = (
        _load_model_overrides(config=config)
        if config is not None
        else _load_model_overrides()
    )
    provider_key = (provider or "").strip()
    if not overrides or not provider_key:
        return None
    # Forward (Hermes → models.dev id) and reverse (caller passed a models.dev id, config keyed by Hermes id) aliases.
    candidates = [provider_key, PROVIDER_TO_MODELS_DEV.get(provider_key), *_models_dev_to_hermes_ids(provider_key)]
    return next((section for section in (overrides.get(key) if key else None for key in candidates) if isinstance(section, dict)), None)


def _explicit_model_override(provider: str, model: str, *, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Explicit per-provider+model override dict (exact, then case-insensitive skipping the ``_default`` sentinel), or None."""
    model_key = (model or "").strip()
    section = _provider_override_section(provider, config=config) if model_key else None
    if section is None:
        return None
    entry = section.get(model_key)
    if isinstance(entry, dict):
        return entry
    model_lower = model_key.lower()
    return next((mdata for mid, mdata in section.items() if mid != "_default" and mid.lower() == model_lower and isinstance(mdata, dict)), None)


def _default_model_override(provider: str, *, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Fill-gap ``_default`` override: per-provider first, then global; or None."""
    section = _provider_override_section(provider, config=config)
    if section is not None and isinstance(section.get("_default"), dict):
        return section["_default"]
    overrides = (
        _load_model_overrides(config=config)
        if config is not None
        else _load_model_overrides()
    )
    global_default = overrides.get("_default")
    return global_default if isinstance(global_default, dict) else None


def _override_for(
    provider: str, model: str, *, catalog_hit: bool, config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Explicit override if any; else the ``_default`` only on a catalog miss."""
    explicit = _explicit_model_override(provider, model, config=config)
    return explicit if explicit is not None or catalog_hit else _default_model_override(provider, config=config)


def _override_context_window(provider: str, model: str) -> Optional[int]:
    """EXPLICITLY overridden context_window, or None. Explicit-only on purpose: this runs early in the
    resolution chain (agent/model_metadata.py, before custom_providers and live probes) where a
    ``_default`` must not preempt more specific sources; fill-gap defaults apply in ``lookup_models_dev_context``."""
    ov = _explicit_model_override(provider, model)
    return _override_int(ov, "context_window") if ov is not None else None


# Catalog miss — a _default override may fill the gap (#84482).
def _default_override_context(provider: str) -> Optional[int]:
    """Fill-gap context from a ``_default`` override, for catalog misses."""
    default = _default_model_override(provider)
    return _override_int(default, "context_window") if default is not None else None


def _builtin_model_metadata(
    provider: str, model: str, *, config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Built-in metadata for a provider/model pair, if Hermes has a vendor-specific entry."""
    provider_key = _models_dev_id(provider, config=config) or (provider or "").strip()
    return builtin_model_metadata(provider_key, model)


def _relay_vision_marker_metadata(provider: str, model: str) -> Optional[Dict[str, Any]]:
    """Fill-gap base for an OpenCode Zen/Go ``*-vision*`` model id the catalog does not know. The relays
    resell vendor previews (``deepseek-v4-flash-vision-exp``) before models.dev indexes them, and the id's
    ``-vision`` token is the vendor's own capability marker; without it ``image_input_mode: auto`` treats
    the model as text-only and detours images through the lossy describe path (#96066). Every other field
    keeps the unknown-model defaults, so only vision is claimed."""
    from providers import opencode_provider_family

    if "-vision" not in (model or "").strip().lower():
        return None
    return vision_marker_metadata(
        is_opencode_family=opencode_provider_family(provider) is not None,
        model_id=model,
    )


def _provider_model_capabilities(provider: str, model: str) -> Dict[str, Any]:
    """Exact-model declaration from the registered ``ProviderProfile.model_capabilities`` (canonical
    ``model_overrides`` schema). The ONE plugin seam: every consumer that reads models.dev through this
    module (picker badges, image routing, ``/api/model/info``, context lookup) sees it (#102115)."""
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    return profile.model_capabilities.get(model, {}) if profile is not None else {}


def _apply_overrides(
    provider: str, model: str, entry: Optional[Dict[str, Any]], *, config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Catalog/builtin metadata, patched by the plugin's declaration, then by the explicit user override.
    ``_UNKNOWN_MODEL_BASE`` is the base on a catalog miss; a fill-gap ``_default`` applies only when no
    source knows the model. None when nothing knows it."""
    builtin = _builtin_model_metadata(provider, model, config=config)
    base = entry if entry is not None else builtin
    declared = _provider_model_capabilities(provider, model)
    if declared:
        base = merge_catalog_entry_with_override(
            base if base is not None else UNKNOWN_MODEL_BASE,
            declared,
        )
    override = _override_for(provider, model, catalog_hit=base is not None, config=config)
    if base is None:
        base = _relay_vision_marker_metadata(provider, model)
    return (
        base
        if override is None
        else merge_catalog_entry_with_override(
            base if base is not None else UNKNOWN_MODEL_BASE,
            override,
        )
    )


def query_model_metadata(
    provider: str, model: str, *, allow_network: bool = False, config: Optional[Dict[str, Any]] = None,
) -> Optional[ModelMetadata]:
    """Return canonical metadata for a provider/model, or None if unresolvable.

    Catalogue acquisition remains opt-in on ``allow_network``; the default is
    the no-network hot path. ``model_overrides`` and built-in/relay facts are
    applied before the raw entry is interpreted by ``models.metadata``.
    """
    models = _get_provider_models(provider, allow_network=allow_network, config=config)
    entry = _find_model_entry(models, model, provider) if models is not None else None
    unknown_base = (
        entry is None
        and _builtin_model_metadata(provider, model, config=config) is None
    )
    raw = _apply_overrides(provider, model, entry, config=config)
    if raw is None:
        return None
    return model_metadata_from_entry(
        ModelRef(provider, model),
        raw,
        unknown_model=unknown_base,
    )


def list_provider_models(provider: str, *, allow_network: bool = True) -> List[str]:
    """All model IDs for a provider ([] if unknown). ``allow_network`` defaults to True: the model
    picker is interactive and a fresh catalog is worth a short wait."""
    from providers import normalize_provider
    provider = normalize_provider(provider) or provider
    models = _get_provider_models(provider, allow_network=allow_network)
    return [mid for mid in models if not _should_hide_from_provider_catalog(provider, mid)] if models is not None else []


# Non-agentic or noise models (TTS, embedding, dated preview snapshots, live/streaming-only, image-only).
_NOISE_PATTERNS: re.Pattern = re.compile(
    r"-tts\b|embedding|live-|-(preview|exp)-\d{2,4}[-_]|" r"-image\b|-image-preview\b|-customtools\b", re.IGNORECASE)
# Hidden from the Gemini catalogs surfaced in setup/model selection (capability metadata stays available for direct use).
_GOOGLE_HIDDEN_MODELS = frozenset({
    # Low-TPM Gemma models that trip Google input-token quota walls under agent-style traffic.
    "gemma-4-31b-it", "gemma-4-26b-it", "gemma-4-26b-a4b-it",
    "gemma-3-1b", "gemma-3-1b-it", "gemma-3-2b", "gemma-3-2b-it",
    "gemma-3-4b", "gemma-3-4b-it", "gemma-3-12b", "gemma-3-12b-it",
    "gemma-3-27b", "gemma-3-27b-it",
    # Stale/retired Google slugs that 404 on the current endpoints.
    "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b",
    "gemini-2.0-flash", "gemini-2.0-flash-lite",
})


def _should_hide_from_provider_catalog(provider: str, model_id: str) -> bool:
    return (provider or "").strip().lower() in {"gemini", "google"} and (model_id or "").strip().lower() in _GOOGLE_HIDDEN_MODELS


def list_agentic_models(provider: str, *, allow_network: bool = True) -> List[str]:
    """Model IDs suitable for agentic use: tool_call=True, minus hidden and noise models. [] on any
    failure. ``allow_network`` defaults to True (called from interactive model selection)."""
    models = _get_provider_models(provider, allow_network=allow_network)
    return [
        mid for mid, entry in models.items()
        if isinstance(entry, dict) and not _should_hide_from_provider_catalog(provider, mid) and entry.get("tool_call", False) and not _NOISE_PATTERNS.search(mid)
    ] if models is not None else []


def get_provider_info(
    provider_id: str, *, allow_network: bool = True, config: Optional[Dict[str, Any]] = None,
) -> Optional[ProviderInfo]:
    """Provider metadata by Hermes or models.dev ID, or None if not cataloged. ``allow_network`` defaults to True (interactive setup)."""
    mdev_id = _models_dev_id(provider_id, config=config) or provider_id
    raw = _registry_provider(mdev_id, allow_network)
    return provider_info_from_entry(mdev_id, raw) if raw is not None else None


def get_model_info(
    provider_id: str, model_id: str, *, allow_network: bool = False, config: Optional[Dict[str, Any]] = None,
) -> Optional[ModelInfo]:
    """Full model metadata by Hermes or models.dev provider ID (exact match, then case-insensitive), or
    None if not found. EXPLICIT ``model_overrides`` patch known catalog models; ``_default`` fills the gap
    only for unknown ones. ``allow_network`` defaults to False — cost guard and inventory are hot paths.

    ``model_overrides`` entries use the SAME canonical schema as every other consumer (``context_window``,
    ``supports_*``, ``model_family``) — they are translated into the catalog shape at
    this boundary, and sub-dicts (``limit``, ``modalities``) are merged rather than clobbered. See #84482,
    #8731.
    """
    mdev_id = _models_dev_id(provider_id, config=config) or provider_id
    models = _registry_models(mdev_id, allow_network=allow_network)
    mid, entry = next(_iter_model_entries(models, model_id, suffix_fallback=False, provider=provider_id), (model_id, None)) if models is not None else (model_id, None)
    # Not in catalog — an override (explicit or _default) may still provide it.
    raw = _apply_overrides(provider_id, model_id, entry, config=config)
    return model_info_from_entry(mid, raw, mdev_id) if raw is not None else None

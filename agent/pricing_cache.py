"""Dynamic pricing cache for Hermes Agent.

Provides a JSON-based pricing snapshot cache that is refreshed on first
access per session if the last fetch is older than 24 hours.

Usage:
    from agent.pricing_cache import get_cached_pricing_entry
    entry = get_cached_pricing_entry("qwen3.6-plus", "nous")
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlopen, Request

from agent.usage_pricing import PricingEntry, CostSource, _to_decimal, _UTC_NOW

_CACHE_TTL_SECONDS = 86400  # 24 hours
_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _cache_path() -> Path:
    return _hermes_home() / "pricing-updates" / "openrouter-cache.json"


def _fetch_openrouter_pricing() -> Dict[str, Dict[str, Any]]:
    """Fetch all model pricing from OpenRouter and return as {model_id: pricing}."""
    try:
        req = Request(_OPENROUTER_URL, headers={"User-Agent": "hermes-pricing-cache/1.0"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception:
        return {}

    models = {}
    for m in data.get("data", []):
        mid = m.get("id", "")
        pricing = m.get("pricing", {})
        prompt = pricing.get("prompt")
        completion = pricing.get("completion")
        cache_read = pricing.get("cache_read") or pricing.get("cached_prompt")

        if prompt is not None or completion is not None:
            models[mid] = {
                "input_per_token": prompt,
                "output_per_token": completion,
                "cache_read_per_token": cache_read,
                "fetched_at": _UTC_NOW().isoformat(),
            }

    return models


def _load_cache() -> Optional[Dict[str, Any]]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(data: Dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(str(tmp), str(path))
    except Exception:
        pass


def _cache_is_fresh(cache: Dict[str, Any]) -> bool:
    fetched_at = cache.get("fetched_at", "")
    if not fetched_at:
        return False
    try:
        dt = datetime.fromisoformat(fetched_at)
        age = time.time() - dt.timestamp()
        return age < _CACHE_TTL_SECONDS
    except Exception:
        return False


def _ensure_cache() -> Dict[str, Any]:
    """Load cache and refresh if stale. Called once per session at start."""
    cache = _load_cache()
    if cache and _cache_is_fresh(cache):
        return cache

    # Fetch fresh data
    models = _fetch_openrouter_pricing()
    if models:
        cache = {
            "fetched_at": _UTC_NOW().isoformat(),
            "source": "openrouter",
            "model_count": len(models),
            "models": models,
        }
        _save_cache(cache)
    return cache


def get_cached_pricing_entry(model_id: str) -> Optional[PricingEntry]:
    """Look up pricing for a model from the cached OpenRouter data.

    Args:
        model_id: bare model ID (e.g. "qwen3.6-plus", "grok-4-1-fast-reasoning")

    Returns:
        PricingEntry if found in cache, None otherwise.
    """
    cache = _ensure_cache()
    models = cache.get("models", {})
    
    # Try exact match first
    if model_id in models:
        p = models[model_id]
        return _pricing_from_cache_dict(p, model_id)
    
    # Try with provider prefix stripped
    if "/" in model_id:
        bare = model_id.split("/", 1)[-1]
        if bare in models:
            p = models[bare]
            return _pricing_from_cache_dict(p, model_id)
    
    # Try all models that end with this ID
    for mid, p in models.items():
        if mid == model_id or mid.endswith("/" + model_id):
            return _pricing_from_cache_dict(p, model_id)
    
    return None


def _pricing_from_cache_dict(p: Dict[str, Any], model_id: str) -> PricingEntry:
    """Convert a cached pricing dict to a PricingEntry."""
    _MILLION = Decimal("1000000")
    return PricingEntry(
        input_cost_per_million=_to_decimal(p.get("input_per_token")) * _MILLION if p.get("input_per_token") is not None else None,
        output_cost_per_million=_to_decimal(p.get("output_per_token")) * _MILLION if p.get("output_per_token") is not None else None,
        cache_read_cost_per_million=_to_decimal(p.get("cache_read_per_token")) * _MILLION if p.get("cache_read_per_token") is not None else None,
        source=CostSource("provider_models_api"),
        source_url="https://openrouter.ai/api/v1/models",
        pricing_version="openrouter-cache",
        fetched_at=datetime.fromisoformat(p["fetched_at"]) if p.get("fetched_at") else None,
    )


def refresh_pricing_if_stale(force: bool = False) -> Dict[str, Any]:
    """Fetch and cache pricing if the cache is older than 24 hours.

    Args:
        force: If True, fetch even if cache is fresh.

    Returns:
        The cache dict (fresh or existing).
    """
    if not force:
        cache = _load_cache()
        if cache and _cache_is_fresh(cache):
            return cache

    models = _fetch_openrouter_pricing()
    cache = {
        "fetched_at": _UTC_NOW().isoformat(),
        "source": "openrouter",
        "model_count": len(models),
        "models": models,
    }
    _save_cache(cache)
    return cache

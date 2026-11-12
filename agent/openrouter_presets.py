"""Resolve OpenRouter ``@preset/<slug>`` model names to the model the preset actually runs.

A preset is an account-scoped alias: ``/api/v1/models`` has no row for it, so every cost
estimate made against a preset prices out as ``unknown`` and the CLI cost meter stays
silent for exactly the users who route through presets. ``GET /api/v1/presets/<slug>``
embeds the designated version, whose ``config.model`` is the real route model — that name
prices normally.

Failures are deliberately quiet and cheap: pricing is display-only, so a probe that times
out must never delay or fail a turn. Hits are cached 7 days (disk + memory), misses 10
minutes (memory only), and every path is best-effort.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

PRESET_PREFIX = "@preset/"
_HIT_TTL_S = 7 * 24 * 3600
_MISS_TTL_S = 600
_TIMEOUT_S = 4.0
_CACHE_FILENAME = "openrouter_preset_models.json"

# slug -> (model_or_None, fetched_at)
_memory: dict[str, tuple[Optional[str], float]] = {}
_disk_loaded = False


def preset_slug(model_name: str | None) -> Optional[str]:
    """``@preset/x`` and ``<model>@preset/x`` -> ``"x"``; anything else -> ``None``."""
    name = (model_name or "").strip()
    if PRESET_PREFIX not in name:
        return None
    slug = name.split(PRESET_PREFIX, 1)[1].strip().strip("/")
    return slug or None


def _cache_path():
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "cache" / _CACHE_FILENAME
    except Exception:
        return None


def _load_disk_cache() -> None:
    global _disk_loaded
    if _disk_loaded:
        return
    _disk_loaded = True
    path = _cache_path()
    if path is None:
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(raw, dict):
        return
    for slug, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        model = entry.get("model")
        fetched_at = entry.get("fetched_at")
        if isinstance(fetched_at, (int, float)) and isinstance(model, str) and model:
            _memory[str(slug)] = (model, float(fetched_at))


def _save_disk_cache() -> None:
    path = _cache_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        hits = {s: {"model": m, "fetched_at": t} for s, (m, t) in _memory.items() if m}
        path.write_text(json.dumps(hits, indent=1, sort_keys=True), encoding="utf-8")
    except Exception:
        logger.debug("preset pricing cache write failed", exc_info=True)


def _fetch_model(slug: str, api_key: str) -> Optional[str]:
    """The preset's designated route model, or None if the probe fails."""
    try:
        from agent import model_metadata_http

        url = f"https://openrouter.ai/api/v1/presets/{slug}"
        response = model_metadata_http.get(
            url, headers={"Authorization": f"Bearer {api_key}"}, timeout=_TIMEOUT_S,
        )
        if not getattr(response, "is_success", False) and not getattr(response, "status_code", 0) == 200:
            return None
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        version = (data or {}).get("designated_version") or {}
        model = (version.get("config") or {}).get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
    except Exception:
        logger.debug("preset model probe failed for %s", slug, exc_info=True)
    return None


def underlying_model(model_name: str | None, *, api_key: str | None = None) -> Optional[str]:
    """Route model behind ``@preset/<slug>``; None when it is not a preset or is unresolvable."""
    slug = preset_slug(model_name)
    if not slug:
        return None
    _load_disk_cache()
    cached = _memory.get(slug)
    now = time.time()
    if cached is not None:
        model, fetched_at = cached
        ttl = _HIT_TTL_S if model else _MISS_TTL_S
        if now - fetched_at < ttl:
            return model
    key = (api_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        return None
    model = _fetch_model(slug, key)
    _memory[slug] = (model, now)
    if model:
        _save_disk_cache()
    return model

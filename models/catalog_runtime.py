"""Remote model-catalog cache/fetch lifecycle with no application-layer imports."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from models.catalog_manifest import (
    CatalogSettings,
    DEFAULT_CATALOG_FALLBACK_URLS,
    curated_model_ids,
    curated_openrouter_models,
    default_model,
    provider_block,
    validate_manifest,
)
from utils import atomic_json_write

logger = logging.getLogger(__name__)
_FETCH_TIMEOUT = 8.0
_cache: dict[str, tuple[dict[str, Any], float]] = {}
_swr_lock = threading.Lock()
_swr_inflight: set[str] = set()


def _read(path: Path) -> tuple[dict[str, Any] | None, float]:
    try:
        mtime = path.stat().st_mtime
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None, 0.0
    return (data, mtime) if validate_manifest(data) else (None, 0.0)


def _write(path: Path, data: dict[str, Any]) -> None:
    try:
        atomic_json_write(path, data)
    except OSError as exc:
        logger.info("model catalog cache write failed: %s", exc)


def _fetch(url: str, user_agent: str, timeout: float = _FETCH_TIMEOUT):
    try:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": user_agent},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.info("model catalog fetch failed (%s): %s", url, exc)
        return None
    except Exception as exc:
        logger.info("model catalog fetch errored (%s): %s", url, exc)
        return None
    return data if validate_manifest(data) else None


def _fetch_with_fallback(settings: CatalogSettings, user_agent: str):
    urls = (settings.url, *DEFAULT_CATALOG_FALLBACK_URLS)
    seen: set[str] = set()
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        if data := _fetch(url, user_agent):
            return data
    return None


def _remember(path: Path, data: dict[str, Any], mtime: float):
    _cache[str(path)] = (data, mtime)
    return data


def _spawn_refresh(
    settings: CatalogSettings, cache_path: Path, user_agent: str
) -> None:
    key = str(cache_path)
    with _swr_lock:
        if key in _swr_inflight:
            return
        _swr_inflight.add(key)

    def refresh() -> None:
        try:
            if data := _fetch_with_fallback(settings, user_agent):
                _write(cache_path, data)
        except Exception:
            logger.debug("catalog SWR refresh failed", exc_info=True)
        finally:
            with _swr_lock:
                _swr_inflight.discard(key)

    threading.Thread(target=refresh, daemon=True, name="model-catalog-swr").start()


def get_catalog(
    settings: CatalogSettings,
    cache_path: Path,
    *,
    user_agent: str = "HermesAgent",
    force_refresh: bool = False,
) -> dict[str, Any]:
    if not settings.enabled:
        return {}
    disk, mtime = _read(cache_path)
    fresh = disk is not None and time.time() - mtime < settings.ttl_seconds
    memo = _cache.get(str(cache_path))
    if not force_refresh and disk is not None:
        if fresh and memo is not None and memo[1] == mtime:
            return memo[0]
        if not fresh:
            _spawn_refresh(settings, cache_path, user_agent)
        return _remember(cache_path, disk, mtime)
    fetched = _fetch_with_fallback(settings, user_agent)
    if fetched is not None:
        _write(cache_path, fetched)
        persisted, new_mtime = _read(cache_path)
        return _remember(
            cache_path,
            persisted if persisted is not None else fetched,
            new_mtime or time.time(),
        )
    return _remember(cache_path, disk, mtime) if disk is not None else {}


def provider_catalog_block(
    settings: CatalogSettings,
    cache_path: Path,
    provider: str,
    *,
    user_agent: str = "HermesAgent",
) -> dict[str, Any] | None:
    if not settings.enabled:
        return None
    override = settings.provider_url(provider)
    if override:
        fetched = _fetch(override, user_agent)
        if fetched is not None:
            block = provider_block(fetched, provider)
            if block is not None:
                return block
    return provider_block(
        get_catalog(settings, cache_path, user_agent=user_agent),
        provider,
    )


def cached_default_model(cache_path: Path, provider: str) -> str:
    memo = _cache.get(str(cache_path))
    found = default_model(provider_block(memo[0], provider)) if memo else ""
    if found:
        return found
    disk, _ = _read(cache_path)
    return default_model(provider_block(disk, provider)) if disk is not None else ""


def curated_openrouter(
    settings: CatalogSettings, cache_path: Path, *, user_agent: str = "HermesAgent"
) -> tuple[tuple[str, str], ...]:
    return curated_openrouter_models(
        provider_catalog_block(settings, cache_path, "openrouter", user_agent=user_agent)
    )


def curated_ids(
    settings: CatalogSettings,
    cache_path: Path,
    provider: str,
    *,
    user_agent: str = "HermesAgent",
) -> tuple[str, ...]:
    return curated_model_ids(
        provider_catalog_block(settings, cache_path, provider, user_agent=user_agent)
    )


def refresh_interval_seconds(settings: CatalogSettings) -> float:
    return max(60.0, settings.ttl_seconds)


def refresh_manifest(
    settings: CatalogSettings, cache_path: Path, *, user_agent: str = "HermesAgent"
) -> bool:
    return bool(get_catalog(settings, cache_path, user_agent=user_agent, force_refresh=True))


def reset_cache(cache_path: Path | None = None) -> None:
    if cache_path is None:
        _cache.clear()
    else:
        _cache.pop(str(cache_path), None)

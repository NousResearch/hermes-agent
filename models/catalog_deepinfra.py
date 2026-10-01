"""Single canonical cache and surface projection for DeepInfra catalogue observations.

Provider-specific HTTP and JSON decoding belong to the registered DeepInfra profile.
Credentials and the effective base URL are supplied by application consumers.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from typing import Callable

_DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
_NEGATIVE_TTL = 60.0
_SURFACE_TAGS = frozenset({"chat", "embed", "image-gen", "tts", "stt", "video-gen"})
_UNTAGGED_CHAT_EXCLUSION = re.compile(
    r"(?i)(embed|rerank|whisper|stable-diffusion|flux|sdxl|"
    r"tts|bark|speech|image-gen|clip|vit-|dpt-)"
)
_cache: dict[str, list[dict]] = {}
_negative_cache: dict[str, float] = {}
_cache_lock = threading.RLock()
_fetch_locks: dict[str, threading.Lock] = {}


def _cache_key(base_url: str, api_key: str) -> str:
    # Profile home as well as key fingerprint isolates multiplexed account catalogues.
    from hermes_constants import hermes_home_key

    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest() if api_key else "anon"
    return f"{hermes_home_key()}|{base_url.rstrip('/')}|{fingerprint}"


def reset_catalog_cache() -> None:
    """Explicit refresh/test hook; no raw credential is retained."""
    with _cache_lock:
        _cache.clear()
        _negative_cache.clear()
        _fetch_locks.clear()


def is_catalog_cached(*, base_url: str = _DEFAULT_BASE_URL, api_key: str = "") -> bool:
    with _cache_lock:
        return _cache_key(base_url, api_key) in _cache


def catalog(
    *, base_url: str = _DEFAULT_BASE_URL, api_key: str = "", timeout: float = 5.0,
    force_refresh: bool = False, cached_only: bool = False,
    fetch_catalog: Callable[..., list[dict] | None] | None = None,
) -> list[dict] | None:
    """Return one shared catalogue; None means missing/failed, [] is an authoritative empty list."""
    key = _cache_key(base_url, api_key)
    with _cache_lock:
        if not force_refresh and key in _cache:
            return _cache[key]
        if cached_only:
            return None
        last_failure = _negative_cache.get(key)
        if not force_refresh and last_failure is not None and time.monotonic() - last_failure < _NEGATIVE_TTL:
            return None
        fetch_lock = _fetch_locks.setdefault(key, threading.Lock())

    with fetch_lock:
        with _cache_lock:
            if not force_refresh and key in _cache:
                return _cache[key]
            last_failure = _negative_cache.get(key)
            if not force_refresh and last_failure is not None and time.monotonic() - last_failure < _NEGATIVE_TTL:
                return None

        if fetch_catalog is None:
            from providers import get_provider_profile

            profile = get_provider_profile("deepinfra")
            fetch_catalog = getattr(profile, "fetch_catalog", None)
        try:
            data = (
                fetch_catalog(api_key=api_key, base_url=base_url, timeout=timeout)
                if callable(fetch_catalog) else None
            )
        except Exception:
            data = None
        with _cache_lock:
            if not isinstance(data, list):
                _negative_cache[key] = time.monotonic()
                return None
            # Do not accept a malformed payload as an authoritative empty catalogue.
            if any(not isinstance(row, dict) for row in data):
                _negative_cache[key] = time.monotonic()
                return None
            _cache[key] = data
            _negative_cache.pop(key, None)
            return data


def models_by_tag(
    tag: str, *, base_url: str = _DEFAULT_BASE_URL, api_key: str = "",
    timeout: float = 5.0, force_refresh: bool = False, cached_only: bool = False,
    fetch_catalog: Callable[..., list[dict] | None] | None = None,
) -> list[dict] | None:
    """Project one tagged surface from the shared catalogue without separate surface caches."""
    rows = catalog(
        base_url=base_url, api_key=api_key, timeout=timeout,
        force_refresh=force_refresh, cached_only=cached_only, fetch_catalog=fetch_catalog,
    )
    if rows is None:
        return None
    result: list[dict] = []
    for row in rows:
        model = row.get("id")
        raw_metadata = row.get("metadata")
        if not isinstance(model, str) or not model or raw_metadata is None:
            continue
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_tags = metadata.get("tags")
        tags = raw_tags if isinstance(raw_tags, list) else []
        if any(surface in _SURFACE_TAGS for surface in tags):
            included = tag in tags
        else:
            included = tag == "chat" and not _UNTAGGED_CHAT_EXCLUSION.search(model)
        if included:
            result.append({"id": model, "metadata": metadata})
    return result

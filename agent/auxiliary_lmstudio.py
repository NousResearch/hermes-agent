"""LM Studio-specific projection helpers for auxiliary requests."""

from __future__ import annotations

import logging
import hashlib
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_PROBE_CACHE_TTL_SECONDS = 60.0
_PROBE_CACHE_MAX_ENTRIES = 64
_probe_cache: dict[tuple[str, str, str], tuple[float, list[str]]] = {}
_probe_cache_lock = threading.Lock()


def lmstudio_probe_api_key(client: Any, configured: str | None = None) -> str | None:
    """Use the configured key or the key owned by this request's client only."""
    for value in (configured, getattr(client, "api_key", None)):
        if isinstance(value, str) and value:
            return value
    return None


def clear_lmstudio_probe_cache() -> None:
    """Clear cached LM Studio capability results (primarily useful for tests and reconfiguration)."""
    with _probe_cache_lock:
        _probe_cache.clear()


def _probe_cache_key(model: str, base_url: str, api_key: str | None) -> tuple[str, str, str]:
    # Do not retain API keys in process memory or in diagnostic representations.
    key_fingerprint = hashlib.sha256(api_key.encode()).hexdigest() if api_key else ""
    return model, base_url, key_fingerprint


def lmstudio_reasoning_projection(
    provider_norm: str, model: str, base_url: str, api_key: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return the LM Studio capability context and whether reasoning is supported."""
    if provider_norm != "lmstudio":
        return {}, True
    cache_key = _probe_cache_key(model, base_url, api_key)
    now = time.monotonic()
    with _probe_cache_lock:
        cached = _probe_cache.get(cache_key)
        if cached is not None and now - cached[0] < _PROBE_CACHE_TTL_SECONDS:
            options = list(cached[1])
            return {"lmstudio_reasoning_options": options}, any(option != "off" for option in options)
    try:
        from hermes_cli.models_local import lmstudio_model_reasoning_options
        options = lmstudio_model_reasoning_options(model, base_url, api_key=api_key)
    except Exception:
        logger.debug("LM Studio auxiliary capability probe failed", exc_info=True)
        options = None
    if options is None:
        # A failed catalog read is not evidence that this model has no reasoning
        # support. Leave the result uncached so a recovered server is probed next.
        return {"lmstudio_reasoning_options": []}, False
    options = list(options or [])
    with _probe_cache_lock:
        if len(_probe_cache) >= _PROBE_CACHE_MAX_ENTRIES and cache_key not in _probe_cache:
            oldest = min(_probe_cache, key=lambda key: _probe_cache[key][0])
            _probe_cache.pop(oldest, None)
        _probe_cache[cache_key] = (time.monotonic(), options)
    return {"lmstudio_reasoning_options": options}, any(option != "off" for option in options)

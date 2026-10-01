"""Canonical lifecycle for the public Nous Portal recommendation catalogue.

Provider plugins own HTTP; applications provide the account's Portal endpoint. The
cache is scoped by profile and Portal, including the existing on-disk format.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home, hermes_home_key

logger = logging.getLogger(__name__)

RECOMMENDED_MODELS_PATH = "/api/nous/recommended-models"
CACHE_TTL = 600.0
_DEFAULT_PORTAL = "https://portal.nousresearch.com"
_cache: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}


def reset_cache() -> None:
    """Drop in-process observations; leave durable cache intact."""
    _cache.clear()


def _disk_path() -> Path:
    return get_hermes_home() / "cache" / "nous_recommended_cache.json"


def _read_map() -> dict[str, Any]:
    try:
        value = json.loads(_disk_path().read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeDecodeError):
        return {}


def _disk_entry(base: str) -> tuple[dict[str, Any], float] | None:
    item = _read_map().get(base)
    if not isinstance(item, dict) or not isinstance(item.get("data"), dict) or not item["data"]:
        return None
    try:
        age = time.time() - float(item["ts"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(age) or age < 0:
        return None
    return item["data"], age


def _write_disk(base: str, data: dict[str, Any]) -> None:
    from utils import atomic_json_write
    from hermes_constants import mkdir_under_hermes_home

    if not data:
        return
    path = _disk_path()
    try:
        mkdir_under_hermes_home(path.parent)
        blob = _read_map()
        blob[base] = {"data": data, "ts": time.time()}
        atomic_json_write(path, blob, indent=2)
    except OSError as exc:
        logger.debug("Nous recommendation disk cache write failed: %s", exc)


def fetch_recommended_models(
    portal_base_url: str = "", timeout: float = 5.0, *,
    force_refresh: bool = False,
    fetch_source: Callable[..., dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Read the shared Portal feed. Failures reuse stale data without renewing its TTL."""
    base = str(portal_base_url or _DEFAULT_PORTAL).strip().rstrip("/")
    key = (hermes_home_key(), base)
    now = time.monotonic()
    previous = _cache.get(key)
    if not force_refresh and previous is not None and now - previous[1] < CACHE_TTL:
        return previous[0]

    disk = _disk_entry(base)
    if not force_refresh and disk is not None and disk[1] < CACHE_TTL:
        data, age = disk
        _cache[key] = (data, now - age)
        return data

    if fetch_source is None:
        from providers import get_provider_profile

        profile = get_provider_profile("nous")
        fetch_source = getattr(profile, "fetch_recommended_models", None)
    try:
        payload = (
            fetch_source(base_url=base, timeout=timeout)
            if callable(fetch_source) else None
        )
    except Exception:
        payload = None

    if isinstance(payload, dict) and payload:
        _write_disk(base, payload)
        _cache[key] = (payload, now)
        return payload

    # Preserve the original age on failed refresh, including when a different
    # process supplied the last good observation.
    if disk is not None:
        _cache[key] = (disk[0], now - disk[1])
        return disk[0]
    if previous is not None:
        return previous[0]
    return {}

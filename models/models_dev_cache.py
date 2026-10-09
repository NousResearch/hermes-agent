"""Persistent models.dev cache ownership shared by catalogue and runtime refresh."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home, mkdir_under_hermes_home
from utils import atomic_json_write, atomic_write_text

logger = logging.getLogger(__name__)


def models_dev_cache_path() -> Path:
    return get_hermes_home() / "models_dev_cache.json"


def models_dev_etag_path() -> Path:
    return get_hermes_home() / "models_dev_cache.etag"


def _quietly(what: str, fn, default=None):
    try:
        return fn()
    except Exception as exc:
        logger.debug("Failed to %s: %s", what, exc)
        return default


def load_models_dev_etag() -> str:
    path = models_dev_etag_path()
    return _quietly(
        "load models.dev ETag",
        lambda: path.read_text(encoding="utf-8-sig").strip() if path.exists() else "",
        "",
    )


def save_models_dev_etag(etag: str) -> None:
    def write() -> None:
        path = models_dev_etag_path()
        mkdir_under_hermes_home(path.parent)
        atomic_write_text(path, etag)

    _quietly("save models.dev ETag", write)


def clear_models_dev_etag() -> None:
    _quietly(
        "clear models.dev ETag",
        lambda: models_dev_etag_path().unlink(missing_ok=True),
    )


def quarantine_models_dev_cache(cache_path: Path | None = None) -> None:
    path = cache_path or models_dev_cache_path()
    try:
        path.rename(path.with_suffix(".json.corrupt"))
    except Exception as exc:
        logger.debug("Could not quarantine corrupt models.dev cache: %s", exc)
    clear_models_dev_etag()


def valid_models_dev_registry(data: Any) -> bool:
    return isinstance(data, dict) and bool(data)


def valid_models_dev_registry(data: Any) -> bool:
    """Return whether a registry is non-empty and safe to serve."""
    return isinstance(data, dict) and bool(data)


def load_models_dev_disk_cache() -> Dict[str, Any]:
    """Load the non-empty registry or quarantine an invalid disk cache."""
    try:
        path = models_dev_cache_path()
        if path.exists():
            with path.open(encoding="utf-8-sig") as handle:
                data = json.load(handle)
            if valid_models_dev_registry(data):
                return data
            logger.warning(
                "models.dev disk cache is corrupt or empty; quarantining "
                "(will refetch from network)"
            )
            quarantine_models_dev_cache(path)
    except Exception as exc:
        logger.warning("Failed to load models.dev disk cache; quarantining: %s", exc)
        try:
            quarantine_models_dev_cache()
        except Exception:
            pass
    return {}


def models_dev_disk_cache_age_seconds() -> Optional[float]:
    def stat() -> Optional[float]:
        path = models_dev_cache_path()
        age = time.time() - path.stat().st_mtime if path.exists() else -1
        return age if age >= 0 else None

    return _quietly("stat models.dev disk cache", stat)


def save_models_dev_disk_cache(data: Dict[str, Any], etag: str = "") -> None:
    _quietly(
        "save models.dev disk cache",
        lambda: atomic_json_write(
            models_dev_cache_path(), data, indent=None, separators=(",", ":")
        ),
    )
    if etag:
        save_models_dev_etag(etag)


__all__ = [
    "clear_models_dev_etag",
    "load_models_dev_disk_cache",
    "load_models_dev_etag",
    "models_dev_cache_path",
    "models_dev_disk_cache_age_seconds",
    "models_dev_etag_path",
    "valid_models_dev_registry",
    "quarantine_models_dev_cache",
    "save_models_dev_disk_cache",
    "save_models_dev_etag",
    "valid_models_dev_registry",
]
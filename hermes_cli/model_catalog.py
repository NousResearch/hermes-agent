"""Remote model catalog fetcher.

The Hermes docs site hosts a JSON manifest of curated models for providers
we want to update without shipping a release (currently OpenRouter and
Nous Portal). This module fetches, validates, and caches that manifest,
falling back to the in-repo hardcoded lists when the network is unavailable.

Pipeline
--------
1. ``get_catalog()`` — returns a parsed manifest dict.
   - Checks in-process cache (invalidated by TTL).
   - Reads disk cache at ``~/.hermes/cache/model_catalog.json``.
   - Fetches the master URL if disk cache is stale or missing.
   - On any fetch failure, keeps using the stale cache (or empty dict).

2. ``get_curated_openrouter_models()`` / ``get_curated_nous_models()`` —
   thin accessors returning the shapes existing callers expect. Each
   falls back to the in-repo hardcoded list on any lookup failure.

Schema (version 1)
------------------
::

    {
      "version": 1,
      "updated_at": "2026-04-25T22:00:00Z",
      "metadata": {...},                # free-form
      "providers": {
        "openrouter": {
          "metadata": {...},            # free-form
          "models": [
            {"id": "vendor/model", "description": "recommended",
             "metadata": {...}}          # free-form, model-level
          ]
        },
        "nous": {...}
      }
    }

Unknown fields are ignored — extra metadata can be added at either level
without bumping ``version``. ``version`` bumps are reserved for
breaking changes (renaming ``providers``, changing ``models`` shape).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hermes_cli import __version__ as _HERMES_VERSION
from utils import atomic_replace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CATALOG_URL = (
    "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json"
)
DEFAULT_TTL_HOURS = 24
DEFAULT_FETCH_TIMEOUT = 8.0
SUPPORTED_SCHEMA_VERSION = 1

_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"

# In-process cache to avoid repeated disk + parse work across multiple
# calls within the same session. Invalidated by TTL against the disk file's
# mtime, so calling code never has to think about this.
_catalog_cache: dict[str, Any] | None = None
_catalog_cache_source_mtime: float = 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_catalog_config() -> dict[str, Any]:
    """Load the ``model_catalog`` config block with defaults filled in."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
    except Exception:
        cfg = {}

    raw = cfg.get("model_catalog")
    if not isinstance(raw, dict):
        raw = {}

    return {
        "enabled": bool(raw.get("enabled", True)),
        "url": str(raw.get("url") or DEFAULT_CATALOG_URL),
        "ttl_hours": float(raw.get("ttl_hours") or DEFAULT_TTL_HOURS),
        "providers": raw.get("providers") if isinstance(raw.get("providers"), dict) else {},
    }


def _cache_path() -> Path:
    """Return the disk cache path. Import lazily so tests can monkeypatch home."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "model_catalog.json"


# ---------------------------------------------------------------------------
# Fetch + validate + cache
# ---------------------------------------------------------------------------


def _fetch_manifest(url: str, timeout: float) -> dict[str, Any] | None:
    """HTTP GET the manifest URL and return a parsed dict, or None on failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": _HERMES_USER_AGENT,
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        logger.info("model catalog fetch failed (%s): %s", url, exc)
        return None
    except Exception as exc:  # pragma: no cover — defensive
        logger.info("model catalog fetch errored (%s): %s", url, exc)
        return None

    if not _validate_manifest(data):
        logger.info("model catalog at %s failed schema validation", url)
        return None

    return data


def _validate_manifest(data: Any) -> bool:
    """Return True when ``data`` matches the minimum manifest shape."""
    if not isinstance(data, dict):
        return False
    version = data.get("version")
    if not isinstance(version, int) or version > SUPPORTED_SCHEMA_VERSION:
        # Future schema version we don't understand — refuse rather than
        # guess. Older schemas (version < 1) aren't supported either.
        return False
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return False
    for pname, pblock in providers.items():
        if not isinstance(pname, str) or not isinstance(pblock, dict):
            return False
        models = pblock.get("models")
        if not isinstance(models, list):
            return False
        for m in models:
            if not isinstance(m, dict):
                return False
            if not isinstance(m.get("id"), str) or not m["id"].strip():
                return False
    return True


def _read_disk_cache() -> tuple[dict[str, Any] | None, float]:
    """Return ``(data_or_none, mtime)``. mtime is 0 if file is missing."""
    path = _cache_path()
    try:
        mtime = path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return (None, 0.0)
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return (None, 0.0)
    if not _validate_manifest(data):
        return (None, 0.0)
    return (data, mtime)


def _write_disk_cache(data: dict[str, Any]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        atomic_replace(tmp, path)
    except OSError as exc:
        logger.info("model catalog cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_catalog(*, force_refresh: bool = False) -> dict[str, Any]:
    """Return the parsed model catalog manifest, or an empty dict on failure.

    Callers should treat a missing provider/model as "use the in-repo fallback"
    — never raise from this function so the CLI keeps working offline.
    """
    global _catalog_cache, _catalog_cache_source_mtime

    cfg = _load_catalog_config()
    if not cfg["enabled"]:
        return {}

    ttl_seconds = max(0.0, cfg["ttl_hours"] * 3600.0)

    disk_data, disk_mtime = _read_disk_cache()
    now = time.time()
    disk_fresh = disk_data is not None and (now - disk_mtime) < ttl_seconds

    # In-process cache hit: disk hasn't changed since we loaded it and still fresh.
    if (
        not force_refresh
        and _catalog_cache is not None
        and disk_data is not None
        and disk_mtime == _catalog_cache_source_mtime
        and disk_fresh
    ):
        return _catalog_cache

    # Disk is fresh enough — use it without a network hit.
    if not force_refresh and disk_fresh and disk_data is not None:
        _catalog_cache = disk_data
        _catalog_cache_source_mtime = disk_mtime
        return disk_data

    # Need to (re)fetch. If it fails, fall back to any stale disk copy.
    fetched = _fetch_manifest(cfg["url"], DEFAULT_FETCH_TIMEOUT)
    if fetched is not None:
        _write_disk_cache(fetched)
        new_disk_data, new_mtime = _read_disk_cache()
        if new_disk_data is not None:
            _catalog_cache = new_disk_data
            _catalog_cache_source_mtime = new_mtime
            return new_disk_data
        _catalog_cache = fetched
        _catalog_cache_source_mtime = now
        return fetched

    if disk_data is not None:
        _catalog_cache = disk_data
        _catalog_cache_source_mtime = disk_mtime
        return disk_data

    return {}


def _fetch_provider_override(provider: str) -> dict[str, Any] | None:
    """If ``model_catalog.providers.<name>.url`` is set, fetch that instead."""
    cfg = _load_catalog_config()
    if not cfg["enabled"]:
        return None
    provider_cfg = cfg["providers"].get(provider)
    if not isinstance(provider_cfg, dict):
        return None
    override_url = provider_cfg.get("url")
    if not isinstance(override_url, str) or not override_url.strip():
        return None
    # Override fetches skip the disk cache because they're usually
    # third-party self-hosted. Re-request on every call but with a short
    # timeout so they don't block the picker.
    return _fetch_manifest(override_url.strip(), DEFAULT_FETCH_TIMEOUT)


def _get_provider_block(provider: str) -> dict[str, Any] | None:
    """Return the provider's manifest block, respecting per-provider overrides."""
    override = _fetch_provider_override(provider)
    if override is not None:
        block = override.get("providers", {}).get(provider)
        if isinstance(block, dict):
            return block

    catalog = get_catalog()
    if not catalog:
        return None
    block = catalog.get("providers", {}).get(provider)
    return block if isinstance(block, dict) else None


def get_curated_openrouter_models() -> list[tuple[str, str]] | None:
    """Return OpenRouter's curated ``[(id, description), ...]`` from the manifest.

    Returns ``None`` when the manifest is unavailable, so callers can fall
    back to their hardcoded list.
    """
    block = _get_provider_block("openrouter")
    if not block:
        return None
    out: list[tuple[str, str]] = []
    for m in block.get("models", []):
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        desc = str(m.get("description") or "")
        out.append((mid, desc))
    return out or None


def get_curated_nous_models() -> list[str] | None:
    """Return Nous Portal's curated list of model ids from the manifest.

    Returns ``None`` when the manifest is unavailable.
    """
    block = _get_provider_block("nous")
    if not block:
        return None
    out: list[str] = []
    for m in block.get("models", []):
        mid = str(m.get("id") or "").strip()
        if mid:
            out.append(mid)
    return out or None


def reset_cache() -> None:
    """Clear the in-process cache. Used by tests and ``hermes model --refresh``."""
    global _catalog_cache, _catalog_cache_source_mtime
    _catalog_cache = None
    _catalog_cache_source_mtime = 0.0

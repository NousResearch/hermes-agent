"""Remote model catalog fetcher.

``get_catalog()`` returns the parsed manifest: in-process cache (TTL) → disk cache at
``~/.hermes/cache/model_catalog.json`` → master URL fetch; any fetch failure keeps the stale copy
(or ``{}``). ``get_curated_openrouter_models()`` / ``get_curated_nous_models()`` are thin accessors
whose callers fall back to the in-repo lists on ``None``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

from hermes_cli import __version__ as _HERMES_VERSION
from utils import atomic_replace

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_URL = (
    "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json")
# The Docusaurus site sits behind Vercel, which occasionally 403s non-browser clients (bot
# challenge); the raw GitHub copy is the same manifest and is not bot-gated.
DEFAULT_CATALOG_FALLBACK_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/website/static/api/model-catalog.json",
)
DEFAULT_TTL_MINUTES = 20
# Legacy key, honoured only when the user set it explicitly; ``ttl_minutes`` is the shipped default.
DEFAULT_TTL_HOURS = DEFAULT_TTL_MINUTES / 60.0
DEFAULT_FETCH_TIMEOUT = 8.0
SUPPORTED_SCHEMA_VERSION = 1

_HERMES_USER_AGENT = f"hermes-cli/{_HERMES_VERSION}"


class _CatalogCache(NamedTuple):
    path: Path
    mtime: float
    data: dict[str, Any]


# Publish identity and data together: one process can serve several profile homes.
_catalog_cache: _CatalogCache | None = None


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

    # ``ttl_hours`` (legacy) is honoured only when ``ttl_minutes`` is still at its default —
    # load_config() deep-merges the default in, so "present" alone doesn't mean "user-set".
    ttl_minutes = raw.get("ttl_minutes")
    try:
        ttl_minutes = float(ttl_minutes) if ttl_minutes not in (None, "") else DEFAULT_TTL_MINUTES
    except (TypeError, ValueError):
        ttl_minutes = DEFAULT_TTL_MINUTES
    if ttl_minutes == DEFAULT_TTL_MINUTES and raw.get("ttl_hours"):
        try:
            ttl_minutes = float(raw["ttl_hours"]) * 60.0
        except (TypeError, ValueError):
            pass
    if ttl_minutes <= 0:
        ttl_minutes = DEFAULT_TTL_MINUTES

    return {
        "enabled": bool(raw.get("enabled", True)),
        "url": str(raw.get("url") or DEFAULT_CATALOG_URL),
        "ttl_hours": ttl_minutes / 60.0,
        "providers": raw.get("providers") if isinstance(raw.get("providers"), dict) else {}}


def _cache_path() -> Path:
    """Disk cache path; imported lazily so tests can monkeypatch home."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "model_catalog.json"


def _fetch_manifest(url: str, timeout: float) -> dict[str, Any] | None:
    """HTTP GET the manifest URL and return a validated dict, or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": _HERMES_USER_AGENT})
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


def _fetch_manifest_with_fallback(
    primary_url: str, timeout: float, fallback_urls: tuple[str, ...] = DEFAULT_CATALOG_FALLBACK_URLS
) -> dict[str, Any] | None:
    """First manifest that fetches and validates from ``primary_url`` then ``fallback_urls`` (skipping
    any equal to the primary so a raw-GitHub-configured operator doesn't double-fetch), or None."""
    data = _fetch_manifest(primary_url, timeout)
    if data is not None:
        return data
    for url in fallback_urls:
        if not url or url == primary_url:
            continue
        data = _fetch_manifest(url, timeout)
        if data is not None:
            logger.info("model catalog primary URL failed; using fallback %s", url)
            return data
    return None


def _validate_manifest(data: Any) -> bool:
    """Return True when ``data`` matches the minimum manifest shape."""
    if not isinstance(data, dict):
        return False
    version = data.get("version")
    if not isinstance(version, int) or version > SUPPORTED_SCHEMA_VERSION:
        return False  # future schema we don't understand — refuse rather than guess
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return False
    for pname, pblock in providers.items():
        if not isinstance(pname, str) or not isinstance(pblock, dict):
            return False
        models = pblock.get("models")
        if not isinstance(models, list):
            return False
        if not all(isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"].strip() for m in models):
            return False
    return True


def _read_disk_cache() -> tuple[dict[str, Any] | None, float]:
    """Return ``(data_or_none, mtime)``. mtime is 0 if file is missing."""
    path = _cache_path()
    try:
        mtime = path.stat().st_mtime
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return (None, 0.0)
    return (data, mtime) if _validate_manifest(data) else (None, 0.0)


def _write_disk_cache(data: dict[str, Any]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        atomic_replace(tmp, path)
    except OSError as exc:
        logger.info("model catalog cache write failed: %s", exc)


# Stale-while-revalidate: at most one background manifest refresh in flight per process. The
# refreshed manifest lands on disk; the NEXT get_catalog() call picks it up via the mtime check.
_catalog_swr_lock = threading.Lock()
_catalog_swr_inflight = False


def _spawn_catalog_swr_refresh(url: str) -> None:
    """Refresh the catalog manifest off-thread (fire-and-forget, deduped)."""
    global _catalog_swr_inflight
    with _catalog_swr_lock:
        if _catalog_swr_inflight:
            return
        _catalog_swr_inflight = True

    def _refresh() -> None:
        global _catalog_swr_inflight
        try:
            fetched = _fetch_manifest_with_fallback(url, DEFAULT_FETCH_TIMEOUT)
            if fetched is not None:
                _write_disk_cache(fetched)
        except Exception:
            logger.debug("catalog SWR refresh failed", exc_info=True)
        finally:
            with _catalog_swr_lock:
                _catalog_swr_inflight = False

    threading.Thread(target=_refresh, daemon=True, name="model-catalog-swr").start()


def _remember(data: dict[str, Any], mtime: float) -> dict[str, Any]:
    global _catalog_cache
    _catalog_cache = _CatalogCache(_cache_path(), mtime, data)
    return data


def get_catalog(*, force_refresh: bool = False) -> dict[str, Any]:
    """Parsed model catalog manifest, or ``{}`` on failure — never raises, so the CLI works offline
    (callers treat a missing provider/model as "use the in-repo fallback")."""
    cfg = _load_catalog_config()
    if not cfg["enabled"]:
        return {}
    ttl_seconds = max(0.0, cfg["ttl_hours"] * 3600.0)
    disk_data, disk_mtime = _read_disk_cache()
    now = time.time()
    disk_fresh = disk_data is not None and (now - disk_mtime) < ttl_seconds

    if not force_refresh and disk_data is not None:
        cached = _catalog_cache
        if disk_fresh and cached is not None and cached.path == _cache_path() and cached.mtime == disk_mtime:
            return cached.data
        if not disk_fresh:
            # Stale-while-revalidate: serve the expired disk copy now and refresh off-thread so the
            # /model picker (which calls this on every open) never blocks on the manifest fetch.
            # Only a cold cache (no disk copy at all) still blocks.
            _spawn_catalog_swr_refresh(cfg["url"])
        return _remember(disk_data, disk_mtime)

    fetched = _fetch_manifest_with_fallback(cfg["url"], DEFAULT_FETCH_TIMEOUT)
    if fetched is not None:
        _write_disk_cache(fetched)
        new_disk_data, new_mtime = _read_disk_cache()
        if new_disk_data is not None:
            return _remember(new_disk_data, new_mtime)
        return _remember(fetched, now)
    if disk_data is not None:
        return _remember(disk_data, disk_mtime)
    return {}


def refresh_interval_seconds() -> float:
    """Return the configured catalog TTL in seconds (the gateway poll cadence)."""
    return max(60.0, _load_catalog_config()["ttl_hours"] * 3600.0)


def refresh_catalogs() -> bool:
    """Force-refresh every remote catalog the picker reads (manifest, OpenRouter live list, Nous Portal
    recommendations), writing each disk cache so the next ``/model`` open in ANY process sees them.
    Blocking; run it off the event loop."""
    if not _load_catalog_config()["enabled"]:
        return False
    catalog = get_catalog(force_refresh=True)
    try:
        from hermes_cli.models import fetch_nous_recommended_models, fetch_openrouter_models

        fetch_openrouter_models(force_refresh=True)
        fetch_nous_recommended_models(force_refresh=True)
    except Exception:
        logger.debug("provider catalog refresh failed", exc_info=True)
    return bool(catalog)


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
    # Overrides are usually third-party self-hosted: skip the disk cache, re-request every call.
    return _fetch_manifest(override_url.strip(), DEFAULT_FETCH_TIMEOUT)


def _block_of(manifest: dict[str, Any] | None, provider: str) -> dict[str, Any] | None:
    block = (manifest or {}).get("providers", {}).get(provider)
    return block if isinstance(block, dict) else None


def _get_provider_block(provider: str) -> dict[str, Any] | None:
    """Return the provider's manifest block, respecting per-provider overrides."""
    return _block_of(_fetch_provider_override(provider), provider) or _block_of(get_catalog(), provider)


def _block_ids(block: dict[str, Any] | None) -> list[tuple[str, dict[str, Any]]]:
    """``(id, entry)`` for every model entry of ``block`` with a non-empty id."""
    models = (block or {}).get("models", [])
    return [(mid, m) for m in models if isinstance(m, dict) and (mid := str(m.get("id") or "").strip())]


def get_curated_openrouter_models() -> list[tuple[str, str]] | None:
    """OpenRouter's curated ``[(id, description), ...]`` from the manifest."""
    rows = _block_ids(_get_provider_block("openrouter"))
    return [(mid, str(m.get("description") or "")) for mid, m in rows] or None


def get_curated_nous_models() -> list[str] | None:
    """Nous Portal's curated model ids from the manifest."""
    return [mid for mid, _ in _block_ids(_get_provider_block("nous"))] or None


def _default_model_from_block(block: dict[str, Any] | None) -> str | None:
    """Id of the model entry labeled ``"default": true``, or None."""
    return next((mid for mid, m in _block_ids(block) if m.get("default")), None)


def get_default_model_from_cache(provider: str) -> str | None:
    """The manifest's labeled default for ``provider`` (the model Hermes silently lands on when the
    user never picked one) — in-process then disk cache only, never a fetch."""
    cached = _catalog_cache
    found = (_default_model_from_block(_block_of(cached.data, provider))
             if cached is not None and cached.path == _cache_path() else None)
    if found:
        return found
    disk_data, _mtime = _read_disk_cache()
    return _default_model_from_block(_block_of(disk_data, provider)) if disk_data is not None else None


def seed_cache_from_checkout(project_root: "Path | str") -> bool:
    """Overwrite the disk cache with the checkout's ``website/static/api/model-catalog.json``.
    After ``hermes update`` that file IS the newest catalog, so the picker stays current even when
    the remote fetch is bot-gated. Validated, then written via the same atomic writer."""
    src = Path(project_root) / "website" / "static" / "api" / "model-catalog.json"
    try:
        with open(src, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("model catalog seed from checkout skipped (%s): %s", src, exc)
        return False
    if not _validate_manifest(data):
        logger.debug("model catalog seed from checkout skipped: invalid manifest at %s", src)
        return False
    _write_disk_cache(data)
    reset_cache()  # drop the in-process copy so the next read picks up the seed
    return True


def reset_cache() -> None:
    """Clear the in-process cache. Used by tests and ``hermes model --refresh``."""
    global _catalog_cache
    _catalog_cache = None

"""Application config/path projection for the canonical model catalogue."""

from __future__ import annotations

from pathlib import Path

from models.catalog_manifest import CatalogSettings, catalog_settings


def catalog_runtime_context() -> tuple[CatalogSettings, Path]:
    from hermes_cli.config import load_config
    from hermes_constants import get_hermes_home

    try:
        config = load_config() or {}
    except Exception:
        config = {}
    settings = catalog_settings(
        config.get("model_catalog") if isinstance(config, dict) else None
    )
    return settings, get_hermes_home() / "cache" / "model_catalog.json"


def catalog_cache_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "cache" / "model_catalog.json"


__all__ = ["catalog_cache_path", "catalog_runtime_context"]

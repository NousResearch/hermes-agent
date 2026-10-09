"""Gateway application projection for the canonical remote model catalogue."""

from __future__ import annotations

from pathlib import Path

from models.catalog_manifest import CatalogSettings, catalog_settings
from models.catalog_runtime import (
    cached_default_model as _cached_default_model,
    refresh_interval_seconds as _refresh_interval_seconds,
    refresh_manifest,
)

_USER_AGENT = "HermesAgent/gateway"


def _catalog_runtime() -> tuple[CatalogSettings, Path]:
    from gateway.run import _gateway_config_home, _load_gateway_config

    config = _load_gateway_config()
    settings = catalog_settings(config.get("model_catalog"))
    return settings, _gateway_config_home() / "cache" / "model_catalog.json"


def cached_default_model(provider: str) -> str:
    _settings, path = _catalog_runtime()
    return _cached_default_model(path, provider)


def refresh_catalogs() -> bool:
    settings, path = _catalog_runtime()
    if not settings.enabled:
        return False
    refreshed = refresh_manifest(settings, path, user_agent=_USER_AGENT)
    try:
        from hermes_cli.inventory import refresh_picker_catalog_sources

        refresh_picker_catalog_sources()
    except Exception:
        pass
    return refreshed


def refresh_interval_seconds() -> float:
    settings, _path = _catalog_runtime()
    return _refresh_interval_seconds(settings)


__all__ = [
    "cached_default_model",
    "refresh_catalogs",
    "refresh_interval_seconds",
]

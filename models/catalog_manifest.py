"""Pure model-catalog manifest policy and interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_CATALOG_URL = (
    "https://hermes-agent.nousresearch.com/docs/api/model-catalog.json"
)
DEFAULT_CATALOG_FALLBACK_URLS = (
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/"
    "website/static/api/model-catalog.json",
)
DEFAULT_TTL_MINUTES = 20.0
SUPPORTED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CatalogSettings:
    enabled: bool = True
    url: str = DEFAULT_CATALOG_URL
    ttl_seconds: float = DEFAULT_TTL_MINUTES * 60.0
    provider_urls: tuple[tuple[str, str], ...] = ()

    def provider_url(self, provider: str) -> str:
        wanted = str(provider or "").strip().lower()
        return next((url for name, url in self.provider_urls if name == wanted), "")


def catalog_settings(raw: object) -> CatalogSettings:
    block = raw if isinstance(raw, Mapping) else {}
    minutes = block.get("ttl_minutes", DEFAULT_TTL_MINUTES)
    try:
        minutes = float(minutes) if minutes not in (None, "") else DEFAULT_TTL_MINUTES
    except (TypeError, ValueError):
        minutes = DEFAULT_TTL_MINUTES
    if minutes == DEFAULT_TTL_MINUTES and block.get("ttl_hours"):
        try:
            minutes = float(block["ttl_hours"]) * 60.0
        except (TypeError, ValueError):
            pass
    if minutes <= 0:
        minutes = DEFAULT_TTL_MINUTES

    providers = block.get("providers")
    overrides: list[tuple[str, str]] = []
    if isinstance(providers, Mapping):
        for name, value in providers.items():
            if not isinstance(value, Mapping):
                continue
            url = str(value.get("url") or "").strip()
            if url:
                overrides.append((str(name).strip().lower(), url))
    return CatalogSettings(
        enabled=bool(block.get("enabled", True)),
        url=str(block.get("url") or DEFAULT_CATALOG_URL),
        ttl_seconds=minutes * 60.0,
        provider_urls=tuple(overrides),
    )


def validate_manifest(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    version = data.get("version")
    if not isinstance(version, int) or version > SUPPORTED_SCHEMA_VERSION:
        return False
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return False
    for name, block in providers.items():
        if not isinstance(name, str) or not isinstance(block, dict):
            return False
        models = block.get("models")
        if not isinstance(models, list):
            return False
        if not all(
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and item["id"].strip()
            for item in models
        ):
            return False
    return True


def provider_block(manifest: object, provider: str) -> dict[str, Any] | None:
    providers = manifest.get("providers") if isinstance(manifest, dict) else None
    block = providers.get(provider) if isinstance(providers, dict) else None
    return block if isinstance(block, dict) else None


def block_models(block: object) -> tuple[tuple[str, dict[str, Any]], ...]:
    models = block.get("models", []) if isinstance(block, dict) else []
    return tuple(
        (model_id, item)
        for item in models
        if isinstance(item, dict)
        and (model_id := str(item.get("id") or "").strip())
    )


def curated_openrouter_models(block: object) -> tuple[tuple[str, str], ...]:
    return tuple(
        (model_id, str(item.get("description") or ""))
        for model_id, item in block_models(block)
    )


def curated_model_ids(block: object) -> tuple[str, ...]:
    return tuple(model_id for model_id, _ in block_models(block))


def default_model(block: object) -> str:
    return next(
        (model_id for model_id, item in block_models(block) if item.get("default")),
        "",
    )


__all__ = [
    "CatalogSettings",
    "DEFAULT_CATALOG_FALLBACK_URLS",
    "DEFAULT_CATALOG_URL",
    "catalog_settings",
    "curated_model_ids",
    "curated_openrouter_models",
    "default_model",
    "provider_block",
    "validate_manifest",
]

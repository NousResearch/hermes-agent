"""Application-supplied DeepInfra endpoint and credential facts.

The tagged catalogue and its cache live under models/. This module reads profile-scoped
configuration but never reinterprets model membership or owns another catalogue cache.
"""
from __future__ import annotations

from typing import Any


def _env(name: str) -> str:
    from hermes_cli.config import get_env_value_prefer_dotenv

    return str(get_env_value_prefer_dotenv(name) or "").strip()


def deepinfra_base_url(section: dict[str, Any] | None = None) -> str:
    """Explicit media-section endpoint > profile-scoped environment > registered default."""
    candidate = section.get("base_url") if isinstance(section, dict) else None
    if candidate:
        return str(candidate).strip().rstrip("/")
    from providers import get_provider_profile
    from models.catalog_deepinfra import _DEFAULT_BASE_URL

    profile = get_provider_profile("deepinfra")
    declared_env = profile.base_url_env_var if profile is not None else "DEEPINFRA_BASE_URL"
    return (_env(declared_env) or (profile.base_url if profile else _DEFAULT_BASE_URL)).strip().rstrip("/")


def models_by_tag(
    tag: str, *, api_key: str | None = None, base_url: str | None = None,
    timeout: float = 5.0, force_refresh: bool = False, cached_only: bool = False,
) -> list[dict] | None:
    """One model-domain catalogue across chat, image, video, TTS, STT and pricing."""
    from models.catalog_deepinfra import models_by_tag as query

    return query(
        tag, api_key=_env("DEEPINFRA_API_KEY") if api_key is None else api_key,
        base_url=base_url or deepinfra_base_url(), timeout=timeout,
        force_refresh=force_refresh, cached_only=cached_only,
    )


def deepinfra_model_ids(tag: str, *, force_refresh: bool = False) -> list[str]:
    rows = models_by_tag(tag, force_refresh=force_refresh)
    return [row["id"] for row in rows] if rows else []


def cached_catalog() -> list[dict] | None:
    """Only resident data; never fetches or triggers the negative-cache path."""
    from models.catalog_deepinfra import catalog

    return catalog(base_url=deepinfra_base_url(), api_key=_env("DEEPINFRA_API_KEY"), cached_only=True)

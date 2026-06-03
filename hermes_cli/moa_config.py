"""Shared helpers for configuring the Mixture-of-Agents council."""

from __future__ import annotations

from typing import Any


def resolve_moac_spec(provider_model: str) -> dict[str, str] | None:
    """Resolve a ``provider/model`` string to a MoA model spec.

    The resolver validates that the provider is authenticated/configured using
    the same inventory path as the model picker. Model IDs are allowed even when
    absent from the curated catalog because custom providers may expose private
    model names.
    """
    pm = provider_model.strip()
    if "/" not in pm:
        return None
    provider_slug, model_name = pm.split("/", 1)
    provider_slug = provider_slug.strip()
    model_name = model_name.strip()
    if not provider_slug or not model_name:
        return None

    try:
        from hermes_cli.model_switch import list_authenticated_providers

        providers = list_authenticated_providers()
        for provider in providers:
            if provider.get("slug") == provider_slug:
                return {"provider": provider_slug, "model": model_name}
    except Exception:
        return None
    return None


def list_moac_available(max_models: int = 20) -> list[dict[str, Any]]:
    """Return available authenticated providers and model samples for /moac."""
    try:
        from hermes_cli.model_switch import list_authenticated_providers

        providers = list_authenticated_providers(max_models=max_models)
        result: list[dict[str, Any]] = []
        for provider in providers:
            slug = provider.get("slug", "")
            name = provider.get("name", slug)
            models = provider.get("models", [])
            if slug and models:
                result.append({"provider": slug, "name": name, "models": models[:12]})
        return result
    except Exception:
        return []

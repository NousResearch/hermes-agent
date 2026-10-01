"""Boundary adapters for model metadata capability sources.

The canonical precedence and result type live in ``models.metadata``. This
module is the agent-side wiring for facts that belong to existing provider,
models.dev, Ollama, and managed-runtime integrations.
"""

from __future__ import annotations

from models import ModelRef
from models.metadata import CapabilitySources
from models.metadata import managed_runtime
from models.metadata.types import ModelMetadataContext, ModelMetadataPatch


def is_managed_route(provider: str, base_url: str = "") -> bool:
    """Boundary adapter for the managed endpoint identity check."""
    from hermes_cli.local_runtime.growth import is_managed_endpoint

    return managed_runtime.is_managed_provider(
        provider,
        base_url,
        endpoint_matcher=is_managed_endpoint,
    )


def _managed_live_props(model_id: str) -> bool | None:
    from hermes_cli.local_runtime.endpoint import managed_get_json, managed_root

    endpoint = managed_root()
    if endpoint is None:
        return None
    payload = managed_get_json(*endpoint, f"/props?model={model_id}", timeout_s=3)
    modalities = payload.get("modalities") if isinstance(payload, dict) else None
    if isinstance(modalities, dict) and "vision" in modalities:
        return bool(modalities["vision"])
    return None


def _managed(ref: ModelRef, context: ModelMetadataContext) -> ModelMetadataPatch | None:
    provider = context.route_provider or ref.provider
    if not is_managed_route(provider, context.base_url):
        return None
    from hermes_cli.local_runtime.bootstrap import assets_dir, staged_model_ids
    from hermes_cli.local_runtime.catalog import entry_for_model

    return managed_runtime.managed_model_metadata(
        ref.model,
        staged_model_ids=staged_model_ids,
        entry_for_model=entry_for_model,
        assets_dir=assets_dir,
        live_props=_managed_live_props,
    )


def _models_dev(ref: ModelRef, context: ModelMetadataContext) -> ModelMetadataPatch | None:
    from agent.models_dev import query_model_metadata
    from models.metadata.context import strip_codex_context_variant_suffix

    provider = context.route_provider or ref.provider
    model = strip_codex_context_variant_suffix(ref.model) if provider == "openai-codex" else ref.model
    caps = query_model_metadata(provider, model, allow_network=context.allow_network)
    if caps is None:
        return None
    return ModelMetadataPatch(
        supports_tools=getattr(caps, "supports_tools", None),
        supports_vision=getattr(caps, "supports_vision", None),
        supports_reasoning=getattr(caps, "supports_reasoning", None),
        context_window=getattr(caps, "context_window", None),
        max_output_tokens=getattr(caps, "max_output_tokens", None),
        model_family=getattr(caps, "model_family", None),
        input_modalities=getattr(caps, "input_modalities", None),
        output_modalities=getattr(caps, "output_modalities", None),
    )


def _should_probe_ollama_vision(provider: str, base_url: str, api_key: str = "") -> bool:
    """Only fingerprint local Ollama endpoints; never spray remote APIs."""
    if (provider or "").strip().lower() == "ollama":
        return True
    if not base_url:
        return False
    from models.metadata.context import detect_local_server_type, is_local_endpoint

    return bool(is_local_endpoint(base_url)) and detect_local_server_type(base_url, api_key=api_key) == "ollama"


def _ollama(ref: ModelRef, context: ModelMetadataContext) -> ModelMetadataPatch | None:
    from models.metadata.context import (
        query_ollama_supports_vision,
        strip_codex_context_variant_suffix,
    )

    provider = context.route_provider or ref.provider
    base_url = context.base_url
    if not base_url and provider.lower() == "ollama":
        base_url = "http://localhost:11434/v1"
    if not _should_probe_ollama_vision(provider, base_url, api_key=context.api_key):
        return None
    model = strip_codex_context_variant_suffix(ref.model) if provider == "openai-codex" else ref.model
    verdict = query_ollama_supports_vision(model, base_url, api_key=context.api_key)
    return None if verdict is None else ModelMetadataPatch(supports_vision=verdict)


def _provider(ref: ModelRef, context: ModelMetadataContext) -> ModelMetadataPatch | None:
    from providers import get_provider_profile

    profile = get_provider_profile(ref.provider)
    if profile is None or profile.supports_vision is not True:
        return None
    return ModelMetadataPatch(supports_vision=True)


def default_capability_sources() -> CapabilitySources:
    """Return the application adapters in canonical discovery order."""
    return CapabilitySources(
        live=_managed,
        catalog=_models_dev,
        local=_ollama,
        provider=_provider,
    )


__all__ = ["default_capability_sources"]
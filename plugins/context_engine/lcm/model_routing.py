"""LCM model override routing helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRoute:
    provider: str | None
    model: str


ProviderResolver = Callable[[str], bool]


# Conservative allowlist for built-in provider registry fallbacks. Non-canonical
# named custom providers from the Hermes config are safe to split; registry-only
# providers still need an allowlist because many provider IDs are also valid
# OpenRouter model namespaces, for example ``anthropic/...``, ``google/...``
# and ``x-ai/...``. ``custom:<name>/...`` is accepted for non-canonical named
# custom providers because Hermes auxiliary routing normalizes ``provider`` the
# same way, but canonical built-ins such as ``custom:openai-codex/...`` remain
# model-only to avoid accidentally selecting the built-in provider.
_PROVIDER_PREFIXES = frozenset({"cerebras"})


def _provider_route_is_resolvable(provider: str) -> bool:
    """Return whether Hermes can route an explicit auxiliary provider."""
    provider = (provider or "").strip().lower()
    if not provider:
        return False
    if provider.startswith("custom:"):
        provider = provider.split(":", 1)[1].strip()
        if not provider:
            return False

    try:
        from hermes_cli.auth import PROVIDER_REGISTRY

        if provider in PROVIDER_REGISTRY:
            return provider in _PROVIDER_PREFIXES
    except Exception:
        pass

    try:
        from hermes_cli.runtime_provider import _get_named_custom_provider

        if _get_named_custom_provider(provider):
            return True
    except Exception:
        pass

    return False


def parse_lcm_model_override(
    value: str | None,
    *,
    provider_resolver: ProviderResolver | None = None,
) -> ModelRoute:
    """Parse an LCM model override into explicit provider/model routing.

    Values whose first path segment is resolvable by the Hermes host are split
    into ``provider=<prefix>`` and ``model=<rest>``. The default resolver only
    treats non-canonical named custom providers (plus conservative registry
    allowlist entries) as resolvable so OpenRouter-style model slugs and
    canonical built-in provider names remain model-only overrides.
    """
    model = (value or "").strip()
    if not model:
        return ModelRoute(provider=None, model="")

    provider, sep, rest = model.partition("/")
    provider = provider.strip().lower()
    rest = rest.strip()
    route_provider = provider
    if provider.startswith("custom:"):
        route_provider = provider.split(":", 1)[1].strip()
    can_resolve_provider = provider_resolver or _provider_route_is_resolvable
    if sep and rest and route_provider and can_resolve_provider(route_provider):
        return ModelRoute(provider=route_provider, model=rest)

    return ModelRoute(provider=None, model=model)


def apply_lcm_model_route(call_kwargs: dict, model: str | None) -> None:
    """Apply parsed LCM provider/model overrides to Hermes auxiliary kwargs."""
    route = parse_lcm_model_override(model)
    if route.provider:
        call_kwargs["provider"] = route.provider
    if route.model:
        call_kwargs["model"] = route.model
    if model and route.model:
        logger.debug(
            "LCM auxiliary model override routed: raw=%r provider=%s model=%s",
            model,
            route.provider or "(task default)",
            route.model,
        )

"""Profile-scoped registry for plugin-provided probabilistic decision backends."""

from __future__ import annotations

import logging
from typing import Optional

from agent.decision_provider import DecisionProvider
from agent.provider_registry import ProviderRegistry, is_available_safe, lower_key

logger = logging.getLogger(__name__)

_registry: ProviderRegistry[DecisionProvider] = ProviderRegistry(
    label="Decision", provider_cls=DecisionProvider, logger=logger, normalize=lower_key,
)
_registry.export(globals())


def resolve_provider(name: Optional[str] = None, *, scope: Optional[str] = None) -> Optional[DecisionProvider]:
    """Resolve an explicit provider, or the sole available provider when unambiguous."""
    if name:
        provider = _registry.get_provider(name, scope=scope)
        if provider is None:
            return None
        return provider if is_available_safe(
            provider, logger, "Decision provider %s.is_available() raised %s", level=logging.WARNING,
        ) else None
    available = [
        provider for provider in _registry.list_providers(scope=scope)
        if is_available_safe(
            provider, logger, "Decision provider %s.is_available() raised %s", level=logging.WARNING,
        )
    ]
    return available[0] if len(available) == 1 else None

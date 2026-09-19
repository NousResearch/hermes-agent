"""Profile-scoped registry for plugin-provided probabilistic decision backends."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from agent.decision_provider import DecisionProvider
from agent.provider_registry import ProviderRegistry, lower_key

logger = logging.getLogger(__name__)

_registry: ProviderRegistry[DecisionProvider] = ProviderRegistry(
    label="Decision", provider_cls=DecisionProvider, logger=logger, normalize=lower_key,
)
_registry.export(globals())


def resolve_provider(
    name: Optional[str] = None,
    *,
    scope: Optional[str] = None,
    available: Optional[Callable[[DecisionProvider], bool]] = None,
) -> Optional[DecisionProvider]:
    """Resolve an explicit provider, or the sole available provider when unambiguous."""
    predicate = available or (lambda provider: provider.is_available())

    def is_available(provider: DecisionProvider) -> bool:
        try:
            return bool(predicate(provider))
        except Exception as exc:  # noqa: BLE001 - plugin availability is a provider boundary
            logger.warning(
                "Decision provider %s.is_available() raised %s", provider.name, exc,
            )
            return False

    if name:
        provider = _registry.get_provider(name, scope=scope)
        if provider is None:
            return None
        return provider if is_available(provider) else None
    available = [
        provider for provider in _registry.list_providers(scope=scope)
        if is_available(provider)
    ]
    return available[0] if len(available) == 1 else None

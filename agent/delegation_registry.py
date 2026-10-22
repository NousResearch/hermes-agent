"""Delegation provider registry.

Populated by plugins via :meth:`PluginContext.register_delegation_provider`; ``delegate_task``
asks :func:`get_active_provider` which strategy turns each task into a child run. Selection is
``delegation.backend`` in config.yaml. Unset (or ``default``) is the built-in strategy, today's
behaviour: every child follows the ``delegation`` section (``delegation.provider`` /
``delegation.model`` / ``delegation.base_url`` ..., else the parent's route). A configured backend
that is not registered or reports unavailable falls back to ``default`` with one warning, so a
disabled plugin never takes delegation down; its schema parameters are absent too, since the
schema is built from the same resolution.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from agent.delegation_provider import DelegationProvider
from agent.provider_registry import ProviderRegistry, is_available_safe, lower_key

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "default"


class DefaultDelegationProvider(DelegationProvider):
    """Today's strategy: every task takes the route the ``delegation`` section describes."""

    @property
    def name(self) -> str:
        return DEFAULT_BACKEND

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        # The delegation section has no speed tier: an unknown key there never becomes one.
        return {k: v for k, v in config.items() if k != "service_tier"}


def _reject_builtin_collision(key: str) -> None:
    logger.warning("Delegation provider '%s' shadows the built-in strategy; registration ignored.", key)


_registry: ProviderRegistry[DelegationProvider] = ProviderRegistry(
    label="Delegation", provider_cls=DelegationProvider, logger=logger, normalize=lower_key,
    builtin_names=frozenset({DEFAULT_BACKEND}), on_builtin_collision=_reject_builtin_collision,
)
_registry.export(globals())

_BUILTIN_DEFAULT = DefaultDelegationProvider()
_warned_fallbacks: set[str] = set()


def configured_backend(config: Mapping[str, Any]) -> str:
    """``delegation.backend`` normalized; blank/non-string means ``default``."""
    raw = config.get("backend") if isinstance(config, Mapping) else None
    return lower_key(raw) if isinstance(raw, str) and raw.strip() else DEFAULT_BACKEND


def get_active_provider(config: Mapping[str, Any]) -> DelegationProvider:
    """The provider ``config`` (the ``delegation`` section) selects; ``default`` when unset,
    unregistered or unavailable."""
    name = configured_backend(config)
    if name == DEFAULT_BACKEND:
        return _BUILTIN_DEFAULT
    provider: Optional[DelegationProvider] = _registry.get_provider(name)
    if provider is not None and is_available_safe(
            provider, logger, "delegation provider %s.is_available() raised %s"):
        return provider
    if name not in _warned_fallbacks:
        _warned_fallbacks.add(name)
        logger.warning(
            "delegation.backend '%s' is %s; subagents use the default strategy. Install/enable the "
            "plugin that provides it (`hermes plugins list`).", name,
            "not registered" if provider is None else "unavailable",
        )
    return _BUILTIN_DEFAULT

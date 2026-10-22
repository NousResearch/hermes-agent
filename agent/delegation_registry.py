"""Delegation provider registry.

Populated by plugins via :meth:`PluginContext.register_delegation_provider`; ``delegate_task``
asks :func:`get_active_provider` which strategy turns each task into a child run. Selection is
``delegation.backend`` in config.yaml (a session may override it, ``agent/delegation_backends.py``).
Unset (or ``default``) is the built-in strategy, today's behaviour: every child follows the
``delegation`` section (``delegation.provider`` / ``delegation.model`` / ``delegation.base_url`` ...,
else the parent's route). A configured backend that is not registered or reports unavailable fails
closed: ``delegate_task`` returns a tool error naming the backend and the fix, and never quietly runs
the children on another route. Its schema parameters are absent, since the schema is built from the
same resolution.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from agent.delegation_provider import DelegationProvider, DelegationRouteError
from agent.provider_registry import ProviderRegistry, is_available_safe, lower_key

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "default"


class DefaultDelegationProvider(DelegationProvider):
    """Today's strategy: every task takes the route the ``delegation`` section describes."""

    @property
    def name(self) -> str:
        return DEFAULT_BACKEND

    @property
    def display_name(self) -> str:
        return "Default"

    @property
    def summary(self) -> str:
        return "Every subagent uses delegation.model / delegation.provider, else the parent's model."

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


def configured_backend(config: Mapping[str, Any]) -> str:
    """``delegation.backend`` normalized; blank/non-string means ``default``."""
    raw = config.get("backend") if isinstance(config, Mapping) else None
    return lower_key(raw) if isinstance(raw, str) and raw.strip() else DEFAULT_BACKEND


def backend_status(name: str) -> tuple[Optional[DelegationProvider], str]:
    """``(provider, "")`` when backend ``name`` can run here, else ``(provider or None, reason)``; the reason is
    the user-facing fix (it becomes the ``delegate_task`` tool error)."""
    if name == DEFAULT_BACKEND:
        return _BUILTIN_DEFAULT, ""
    provider: Optional[DelegationProvider] = _registry.get_provider(name)
    if provider is None:
        return None, (
            f"Delegation backend '{name}' is not installed: no enabled plugin provides it. Install or enable the "
            f"plugin that does (`hermes plugins list`), or switch back to the default backend "
            f"(`hermes config set delegation.backend default`).")
    if not is_available_safe(provider, logger, "delegation provider %s.is_available() raised %s"):
        return provider, (
            f"Delegation backend '{name}' is unavailable here (its plugin reports it cannot run). Fix its setup, or "
            f"switch back to the default backend: `/delegation default` for this session, or "
            f"`hermes config set delegation.backend default`.")
    return provider, ""


def get_active_provider(config: Mapping[str, Any]) -> DelegationProvider:
    """The provider ``config`` (the ``delegation`` section) selects; ``default`` when unset. Raises
    :class:`DelegationRouteError` naming the backend and the fix when it is not registered or unavailable."""
    provider, reason = backend_status(configured_backend(config))
    if reason or provider is None:
        raise DelegationRouteError(f"{reason} No subagent was started.")
    return provider

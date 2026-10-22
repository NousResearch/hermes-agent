"""Delegation provider registry.

Populated by plugins via :meth:`PluginContext.register_delegation_provider`; ``delegate_task``
asks :func:`get_active_provider` which strategy turns each task into a child run. Selection is
``delegation.backend`` in config.yaml, ``default`` when unset. The default strategy (today's
behaviour: every child follows the ``delegation`` section, else the parent's route) is itself a
plugin, the bundled ``plugins/delegation/default``, registered through the same seam as any other.
A session may override the selection (``agent/delegation_backends.py``). A configured backend that is
not registered or reports unavailable fails closed: ``delegate_task`` returns a tool error naming the
backend and the fix, and never quietly runs the children on another route. Its schema parameters are
absent, since the schema is built from the same resolution.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional

from agent.delegation_provider import DelegationProvider, DelegationRouteError
from agent.provider_registry import ProviderRegistry, is_available_safe, lower_key

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "default"

_registry: ProviderRegistry[DelegationProvider] = ProviderRegistry(
    label="Delegation", provider_cls=DelegationProvider, logger=logger, normalize=lower_key,
)
_registry.export(globals())

_BUNDLED_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def configured_backend(config: Mapping[str, Any]) -> str:
    """``delegation.backend`` normalized; blank/non-string means ``default``."""
    raw = config.get("backend") if isinstance(config, Mapping) else None
    return lower_key(raw) if isinstance(raw, str) and raw.strip() else DEFAULT_BACKEND


_first_use: dict[str, DelegationProvider] = {}


class _BundledRegistrar:
    """The one ``ctx`` method a bundled strategy's ``register()`` calls on first use. It keeps the provider
    outside the plugin registry, whose entries only the plugin manager adds and removes."""

    def __init__(self, name: str) -> None:
        self.name = name

    def register_delegation_provider(self, provider: DelegationProvider) -> None:
        _first_use[self.name] = provider


def _bundled_disabled(name: str) -> bool:
    from hermes_cli.plugins_discovery import _get_disabled_plugins
    return bool({name, f"delegation/{name}"} & _get_disabled_plugins())


def _load_bundled(name: str) -> Optional[DelegationProvider]:
    """The bundled strategy ``plugins/delegation/<name>/`` on first use when the plugin sweep has not
    registered it (a runtime that never ran discovery, or pinned it with zero plugins): delegation is a
    core tool and must not depend on the sweep. ``plugins.disabled`` wins on every call, so a later
    disable takes effect without a restart."""
    from hermes_cli.plugins import get_bundled_plugins_dir
    from plugins import plugin_loader
    if not _BUNDLED_NAME_RE.match(name) or _bundled_disabled(name):  # the name comes from config.yaml: never a path
        return None
    if name in _first_use:
        return _first_use[name]
    plugin_dir = get_bundled_plugins_dir() / "delegation" / name
    if not (plugin_dir / "plugin.yaml").is_file():
        return None
    module = plugin_loader.load_plugin_module(
        f"plugins.delegation.{name.replace('-', '_')}", plugin_dir, parents=("plugins", "plugins.delegation"),
        logger=logger)
    if module is None:
        return None
    module.register(_BundledRegistrar(name))
    return _first_use.get(name)


def _available(provider: Optional[DelegationProvider]) -> bool:
    return provider is not None and is_available_safe(
        provider, logger, "delegation provider %s.is_available() raised %s")


def _usable(name: str) -> Optional[DelegationProvider]:
    """The available provider for ``name``: the registered one, else a bundled strategy of that name
    (also when the registered one reports unavailable), else one plugin discovery sweep (a caller
    that never imported ``model_tools``)."""
    registered = _registry.get_provider(name)
    if _available(registered):
        return registered
    bundled = _load_bundled(name)
    if _available(bundled):
        return bundled
    if registered is None and bundled is None:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        registered = _registry.get_provider(name)
        if _available(registered):
            return registered
    return None


def backend_status(name: str) -> tuple[Optional[DelegationProvider], str]:
    """``(provider, "")`` when backend ``name`` can run here, else ``(provider or None, reason)``; the reason is
    the user-facing fix (it becomes the ``delegate_task`` tool error)."""
    provider = _usable(name)
    if provider is not None:
        return provider, ""
    registered = _registry.get_provider(name)
    if registered is None and name == DEFAULT_BACKEND:
        return None, ("No default delegation strategy is available: the bundled plugin delegation/default is "
                      "disabled. Remove it from plugins.disabled in config.yaml.")
    if registered is None:
        return None, (
            f"Delegation backend '{name}' is not installed: no enabled plugin provides it. Install or enable the "
            f"plugin that does (`hermes plugins list`), or switch back to the default backend "
            f"(`hermes config set delegation.backend default`).")
    return registered, (
        f"Delegation backend '{name}' is unavailable here (its plugin reports it cannot run). Fix its setup, or "
        f"switch back to the default backend: `/delegation default` for this session, or "
        f"`hermes config set delegation.backend default`.")


def get_active_provider(config: Mapping[str, Any]) -> DelegationProvider:
    """The provider ``config`` (the ``delegation`` section) selects; ``default`` when unset. Raises
    :class:`DelegationRouteError` naming the backend and the fix when it is not registered or unavailable."""
    provider, reason = backend_status(configured_backend(config))
    if reason or provider is None:
        raise DelegationRouteError(f"{reason} No subagent was started.")
    return provider

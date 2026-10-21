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


class _BundledRegistrar:
    """The one ``ctx`` method a bundled strategy's ``register()`` calls."""

    def register_delegation_provider(self, provider: DelegationProvider) -> None:
        _registry.register(provider)


def _load_bundled(name: str) -> Optional[DelegationProvider]:
    """Load the bundled strategy ``plugins/delegation/<name>/`` on first use when the plugin sweep has
    not registered it (a runtime that never ran discovery, or pinned it with zero plugins): delegation
    is a core tool and must not depend on the sweep. ``plugins.disabled`` still wins."""
    import importlib.util
    from hermes_cli.plugins import get_bundled_plugins_dir
    from hermes_cli.plugins_discovery import _get_disabled_plugins
    if not _BUNDLED_NAME_RE.match(name):  # the name comes from config.yaml: never a path
        return None
    plugin_dir = get_bundled_plugins_dir() / "delegation" / name
    if not (plugin_dir / "plugin.yaml").is_file() or {name, f"delegation/{name}"} & _get_disabled_plugins():
        return None
    spec = importlib.util.spec_from_file_location(
        f"hermes_bundled_delegation_{name.replace('-', '_')}", plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)])
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register(_BundledRegistrar())
    return _registry.get_provider(name)


def _usable(name: str) -> Optional[DelegationProvider]:
    """The registered, available provider for ``name``: registry, else a bundled strategy of that
    name, else one plugin discovery sweep (a caller that never imported ``model_tools``)."""
    provider = _registry.get_provider(name) or _load_bundled(name)
    if provider is None:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()
        provider = _registry.get_provider(name)
    if provider is None or not is_available_safe(provider, logger, "delegation provider %s.is_available() raised %s"):
        return None
    return provider


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

"""Scoped provider registration shared by PluginContext's provider methods."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from hermes_cli.plugins_ledger import PluginRegistration

if TYPE_CHECKING:
    from hermes_cli.plugins import PluginContext

logger = logging.getLogger("hermes_cli.plugins")


def _register_scoped_provider(
    self: PluginContext, provider: Any, *, kind: str, base_class: type, registry: Any, label: str,
    article: str = "a", normalize: Optional[Callable[[str], str]] = lambda n: n.strip(),
    register: Optional[Callable[..., Any]] = None, reject_message: Optional[str] = None,
    class_registration: bool = False,
) -> Optional[PluginRegistration]:
    """Type-check and lease a scope-keyed provider so unload restores its predecessor.

    Return None when the registry refuses or replaces the provider.
    """
    if class_registration:
        if not isinstance(provider, type) or not issubclass(provider, base_class):
            logger.warning("Plugin '%s' tried to register a %s class that does not inherit from %s. Ignoring.",
                           self.manifest.name, label, base_class.__name__)
            return None
    elif self._wrong_type(provider, base_class, label, article):
        return None
    raw_name = getattr(provider, "name", "") if class_registration else provider.name
    if class_registration and not isinstance(raw_name, str):
        logger.warning(
            "Plugin '%s' tried to register a %s class with a non-string name. Ignoring.",
            self.manifest.name,
            label,
        )
        return None
    registry_name = raw_name if normalize is None else normalize(raw_name)
    scope = self._manager.scope_key
    previous = registry.snapshot_registration(registry_name, scope=scope)
    try:
        accepted = (register or registry.register_provider)(provider, scope=scope)
    except ValueError as exc:
        if reject_message is None:
            raise
        logger.warning(reject_message, self.manifest.name, exc)
        return None
    if (register is not None and not accepted) or registry.snapshot_registration(
        registry_name, scope=scope
    ) is not provider:
        return None
    handle = self._manager._track_scoped_registration(
        self.manifest, kind, registry_name, registry, provider, previous
    )
    logger.info("Plugin '%s' registered %s: %s", self.manifest.name, label, registry_name)
    return handle

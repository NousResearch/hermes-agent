"""Registration ownership ledger and unload: every plugin registration is recorded with its inverse so force
reload / targeted unload unwind registries in reverse order. Mixed into :class:`hermes_cli.plugins.PluginManager`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Union

from registration_lifecycle import replacement_coordinator
from hermes_cli.plugins_loader import _plugin_home_scope
from hermes_cli.plugins_manifest import PluginManifest, manifest_key

if TYPE_CHECKING:  # pragma: no cover
    from hermes_cli.plugins import LoadedPlugin

logger = logging.getLogger("hermes_cli.plugins")


@dataclass
class PluginRegistration:
    """One host-owned registration plus its inverse, so force reload unwinds registries in reverse order
    (including restoring the entry an override replaced)."""

    kind: str
    key: str
    release: Callable[[], None]
    plugin_key: str = ""
    # Process-global host infrastructure (e.g. dashboard-auth providers): kept out of ``_registration_order``
    # so unload-all cannot dispose it, but still disposed by a *targeted* unload and evicted on force
    # re-discovery when the plugin no longer re-registers it.
    # See #91701.
    persistent: bool = False
    _disposed: bool = field(default=False, init=False, repr=False)
    _on_dispose: Optional[Callable[["PluginRegistration"], None]] = field(default=None, init=False, repr=False)

    @property
    def active(self) -> bool:
        """Whether this handle still owns an active registration."""
        return not self._disposed

    def dispose(self) -> None:
        """Release this registration once; repeated disposal is harmless."""
        if self._disposed:
            return
        self._disposed = True
        try:
            self.release()
        finally:
            if self._on_dispose is not None:
                self._on_dispose(self)


class PluginLedgerMixin:
    def _track_registration(
        self, manifest: PluginManifest, kind: str, key: str, release: Callable[[], None], *,
        persistent: bool = False,
    ) -> PluginRegistration:
        """Record one registration under its canonical plugin key. ``persistent`` ones (process-global host
        infrastructure) stay in the ownership ledger for attribution but NOT in ``_registration_order``, so a
        routine unload cannot dispose them; the handle still releases on explicit ``dispose()``.

        See #91701.
        """
        registration = PluginRegistration(
            kind=kind, key=key, release=release, plugin_key=manifest_key(manifest), persistent=persistent)
        registration._on_dispose = lambda disposed: self._forget_registrations([disposed])
        self._ownership_ledger.setdefault(registration.plugin_key, []).append(registration)
        if not persistent:
            self._registration_order.append(registration)
        return registration

    def _track_scoped_registration(
        self, manifest: PluginManifest, kind: str, name: str, registry: Any, current: Any,
        previous: Any, *, finalize: Optional[Callable[[], None]] = None,
    ) -> PluginRegistration:
        """Lease one ``(kind, scope, name)`` slot of a scope-keyed process-global registry. Unload calls
        ``registry.restore_registration(name, current, replacement, scope=...)`` — identity-conditional, so a
        later generation is never removed by an earlier owner."""
        scope = self.scope_key
        lease = replacement_coordinator.acquire(
            (kind, scope, name), current=current, previous=previous,
            restore=lambda replacement: registry.restore_registration(name, current, replacement, scope=scope),
            finalize=finalize,
        )
        return self._track_registration(manifest, kind, name, lease.dispose)

    def _active_persistent(self) -> List[PluginRegistration]:
        """Live persistent registrations across every plugin in the ownership ledger."""
        return [r for owned in self._ownership_ledger.values() for r in owned if r.persistent and r.active]

    def _evict_stale_persistent_registrations(self) -> None:
        """After re-discovery, dispose parked persistent handles whose plugin did not re-register the same
        ``(kind, key)``. Re-registered ones are dropped WITHOUT disposing — the same object re-registered would
        pass the identity check and evict the live entry.

        Persistent registrations (process-global host infrastructure such as dashboard-auth providers)
        survive an unload-all by design (#91701); ``_unload_scoped`` parks their handles in
        ``_persistent_carryover``. After a re-discovery pass, three cases exist for each parked handle:
        """
        if not self._persistent_carryover:
            return
        parked, self._persistent_carryover = self._persistent_carryover, []
        current = {(r.kind, r.key) for r in self._active_persistent()}
        stale = [r for r in parked if r.active and (r.kind, r.key) not in current]
        for registration in stale:
            logger.info(
                "Evicting persistent registration %s/%s: plugin '%s' no "
                "longer supplies it after re-discovery", registration.kind, registration.key, registration.plugin_key,
            )
        self._dispose_registrations(stale)

    @staticmethod
    def _remove_identity(values: list, target: Any) -> bool:
        """Remove the last exact object match from a registration list."""
        index = next((i for i in range(len(values) - 1, -1, -1) if values[i] is target), None)
        if index is None:
            return False
        del values[index]
        return True

    def _remove_callback(self, mapping: Dict[str, List[Callable]], key: str, callback: Callable) -> None:
        callbacks = mapping.get(key)
        if callbacks is None:
            return
        self._remove_identity(callbacks, callback)
        if not callbacks:
            mapping.pop(key, None)

    def _restore_mapping(self, mapping: Dict[str, Any], key: str, current: Any, previous: Optional[Any]) -> bool:
        """Restore a manager-local mapping only when *current* is still present."""
        if mapping.get(key) is not current:
            return False
        if previous is None:
            mapping.pop(key, None)
        else:
            mapping[key] = previous
        return True

    def _restore_value(self, attribute: str, current: Any, previous: Any) -> bool:
        """Restore a manager-local value only when *current* is still active."""
        if getattr(self, attribute) is not current:
            return False
        setattr(self, attribute, previous)
        return True

    def _remove_name_if_unowned(self, kind: str, names: Set[str], name: str) -> None:
        """Drop *name* from the manager-local name set once no active ledger entry owns it."""
        if not any(r.active and r.kind == kind and r.key == name for r in self._registration_order):
            names.discard(name)

    def _remove_tool_name_if_unowned(self, name: str) -> None:
        self._remove_name_if_unowned("tool", self._plugin_tool_names, name)

    def _remove_platform_name_if_unowned(self, name: str) -> None:
        self._remove_name_if_unowned("platform", self._plugin_platform_names, name)

    def _forget_registrations(self, registrations: List[PluginRegistration]) -> None:
        if not registrations:
            return
        ids = {id(r) for r in registrations}
        self._registration_order = [r for r in self._registration_order if id(r) not in ids]
        for plugin_key, owned in list(self._ownership_ledger.items()):
            remaining = [r for r in owned if id(r) not in ids]
            if remaining:
                self._ownership_ledger[plugin_key] = remaining
            else:
                self._ownership_ledger.pop(plugin_key, None)

    def _dispose_registrations(self, registrations: List[PluginRegistration]) -> None:
        """Dispose registrations in reverse acquisition order, best effort."""
        from hermes_cli.plugins import _PLUGINS_DEBUG
        for registration in reversed(registrations):
            try:
                registration.dispose()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning(
                    "Failed to unload plugin registration %s/%s: %s", registration.plugin_key,
                    registration.key, exc, exc_info=_PLUGINS_DEBUG,
                )

    @staticmethod
    def _resolve_plugin_key(plugin: Union[str, PluginManifest, LoadedPlugin]) -> str:
        from hermes_cli.plugins import LoadedPlugin
        if isinstance(plugin, LoadedPlugin):
            return manifest_key(plugin.manifest)
        return manifest_key(plugin) if isinstance(plugin, PluginManifest) else str(plugin)

    def unload(self, plugin: Union[str, PluginManifest, LoadedPlugin, None] = None) -> bool:
        """Unload registrations while excluding discovery/deferred loading."""
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            return self._unload_scoped(plugin)

    def _unload_scoped(self, plugin: Union[str, PluginManifest, LoadedPlugin, None] = None) -> bool:
        """Unload one plugin (or all when ``plugin=None``, as force rediscovery does). Every ledger registration
        — including on_unload callbacks and supervised tasks — is disposed in reverse acquisition order with
        identity-conditional inverses. Returns ``True`` when anything was found."""
        unload_all = plugin is None
        if unload_all:
            target_keys = set(self._ownership_ledger) | set(self._plugins)
            registrations = list(self._registration_order)
        else:
            target_keys = self._unload_target_keys(self._resolve_plugin_key(plugin))
            registrations = [r for r in self._registration_order if r.plugin_key in target_keys]
            # Persistent registrations are absent from _registration_order (unload-all keeps them), but a
            # *targeted* unload is the disable/uninstall path: a disabled auth plugin's provider must NOT stay
            # live process-wide.
            # See #91701.
            registrations.extend(
                r for key in target_keys for r in self._ownership_ledger.get(key, []) if r.persistent and r.active
            )
        found = bool(target_keys or registrations)
        self._dispose_registrations(registrations)
        self._forget_registrations(registrations)
        if unload_all:
            self._reset_after_unload_all(registrations)
        else:
            for key in target_keys:
                self._plugins.pop(key, None)
        return found

    def _unload_target_keys(self, requested: str) -> Set[str]:
        """Resolve a targeted-unload request to canonical plugin keys (exact key, else by name)."""
        if requested in self._ownership_ledger or requested in self._plugins:
            return {requested}
        return {key for key, loaded in self._plugins.items() if loaded.manifest.name == requested}

    def _reset_after_unload_all(self, registrations: List[PluginRegistration]) -> None:
        """Sweep pre-ledger global state and clear every manager-local container."""
        # Handles are authoritative for global registries; names present in the manager-local sets without a
        # ledger entry (pre-ledger or manually set state) are swept here so they do not survive a force reload
        # as zombies.
        from gateway.platform_registry import platform_registry
        for platform_name in tuple(self._plugin_platform_names):
            platform_registry.unregister(platform_name)
        # Ledger-owned tool names are excluded: their handles already restored the previous entry, and blanket
        # deregistration would remove what the ledger just restored.
        ledger_tool_names = {r.key for r in registrations if r.kind == "tool"}
        preledger_tools = tuple(n for n in self._plugin_tool_names if n not in ledger_tool_names)
        if preledger_tools:
            try:
                from tools.registry import registry as tool_registry
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("unload: tools.registry unavailable: %s", exc)
            else:
                for tool_name in preledger_tools:
                    try:
                        tool_registry.deregister(tool_name)
                    except Exception as exc:
                        logger.debug("unload: tool deregister %s failed: %s", tool_name, exc)
        # Persistent registrations survive unload-all but must not be orphaned by the ledger clear: carry them
        # over so force re-discovery can evict the ones whose plugin does not come back.
        carryover_ids = {id(r) for r in self._persistent_carryover}
        self._persistent_carryover.extend(r for r in self._active_persistent() if id(r) not in carryover_ids)
        for container in (
            self._ownership_ledger, self._plugins, self._hooks, self._middleware,
            self._plugin_tool_names, self._plugin_platform_names, self._cli_commands,
            self._plugin_commands, self._plugin_skills, self._portable_mcp_servers,
            self._aux_tasks, self._system_prompt_sections, self._approval_transports,
            self._slack_action_handlers, self._predeclared_modules, self._predeclared_tools,
            self._platform_handler_factories, self._agent_runtimes,
        ):
            container.clear()
        self._context_engine = None
        with self._hook_timeout_lock:
            self._hook_running_callbacks.clear()
            self._hook_timeout_suppressed_until.clear()
        self._discovered = False

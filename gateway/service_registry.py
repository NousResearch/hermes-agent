"""
Background Service Registry

Allows long-running background services (Nextcloud notification pollers,
file sync watchers, etc.) to self-register so the gateway can discover
and instantiate them without hardcoded ``_create_service()`` if/elif
chains in ``gateway/run.py``.

Background services differ from platform adapters in two ways:

* They do not receive user messages — they observe an external source
  (a notification queue, a file system watcher, a webhook) and forward
  events to platforms.
* They have no ``MessageHandler`` plumbing — start/stop is the entire
  interface plus whatever the service emits internally.

Registrations follow the same profile-scoped ownership protocol as
``platform_registry``: plugin registrations are isolated per resolved
HERMES_HOME scope and overlay the process-global entries, registration
state can be snapshotted and CAS-restored by the plugin ownership
ledger, and deferred loaders let a profile defer the owning plugin
import until a lookup asks for the service.

Usage (plugin side — via ``PluginContext.register_background_service()``,
which carries the plugin's scope and ownership handle)::

    ctx.register_background_service(
        name="nextcloud_notifications",
        label="Nextcloud Notifications",
        service_factory=lambda cfg, gateway: NextcloudNotificationService(cfg),
        check_fn=lambda: True,
        validate_config=lambda cfg: bool(cfg.get("enabled")),
    )

Usage (gateway side)::

    svc = service_registry.create_service("nextcloud_notifications", cfg, self)
    if svc is not None:
        await svc.start()

Service lifecycle contract (see ``gateway/run_services.py``): the factory
returns an object exposing ``async start()`` and ``async stop()``.
``start()`` reports success by completing without raising and without
returning ``False`` — i.e. plain ``return``/``return None`` counts as
started (the common untyped coroutine shape), an explicit ``return
False`` or an exception counts as failed. On failure the runtime still
owns cleanup: it calls ``stop()`` (bounded) on the failed instance so a
partial start never leaks a task or socket.
"""

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

_LoadKey = tuple[Optional[str], str]
_Loader = Callable[[], None]


def _plugin_scope_from_callable(callback: Callable) -> Optional[str]:
    """Infer a plugin profile from code registered outside PluginContext."""
    try:
        from tools.registry import registry as tool_registry
        return tool_registry.plugin_scope_for_callable(callback)
    except (ImportError, AttributeError):
        return None


def _caller_plugin_scope() -> Optional[str]:
    try:
        module_name = sys._getframe(2).f_globals.get("__name__", "") or ""
    except Exception:
        return None
    return _plugin_scope_from_callable(type("_Caller", (), {"__module__": module_name}))


@dataclass
class BackgroundServiceEntry:
    """Metadata and factory for a single background service."""

    # Identifier used in config.yaml under ``services.<name>``.
    name: str

    # Human-readable label.
    label: str

    # Factory callable: receives (config_dict, gateway_runner) and returns
    # a service instance. The instance MUST expose ``async start()`` and
    # ``async stop()`` methods (see the module docstring for the lifecycle
    # contract, and the bundled service plugins under ``plugins/services/``
    # for reference implementations). Using a factory instead of a bare
    # class lets plugins do custom init / dependency injection without
    # subclassing.
    service_factory: Callable[[dict, Any], Any]

    # Returns True when the service's dependencies are importable. Called
    # before instantiation; if it returns False the gateway logs the install
    # hint and skips creation.
    check_fn: Callable[[], bool]

    # Optional config-validity check. Receives the service's config dict
    # (with ``enabled`` already verified by the caller) and returns True
    # when the config is complete enough to start. If None, the registry
    # skips this check.
    validate_config: Optional[Callable[[dict], bool]] = None

    # Env vars this service needs (for ``hermes config`` / dashboard display).
    required_env: list = field(default_factory=list)

    # Hint shown when ``check_fn`` returns False (e.g. ``pip install httpx``).
    install_hint: str = ""

    # ``"builtin"`` or ``"plugin"``.
    source: str = "plugin"

    # Name of the plugin that registered this entry (for diagnostics).
    plugin_name: str = ""


class BackgroundServiceRegistry:
    """Central registry of background services. Registrations are serialized;
    concurrent lazy lookups share an in-flight event while the loader runs
    outside the registry lock — the same scoped-ownership shape as
    ``PlatformRegistry``, so plugin unload/reload can snapshot and
    CAS-restore displaced registrations through the plugin ledger."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, BackgroundServiceEntry] = {}  # process-global
        # Plugin services are isolated per resolved HERMES_HOME and overlay the
        # process-global entries for lookups in that profile's runtime scope.
        self._scoped_entries: dict[str, dict[str, BackgroundServiceEntry]] = {}
        # Deferred loaders: name -> callable importing the owning plugin module
        # (which calls register()); the import happens only when a lookup asks.
        self._deferred: dict[str, _Loader] = {}
        self._scoped_deferred: dict[str, dict[str, _Loader]] = {}
        self._inflight: dict[_LoadKey, threading.Event] = {}
        self._inflight_loaders: dict[_LoadKey, _Loader] = {}
        self._inflight_owners: dict[_LoadKey, int] = {}
        self._cancelled_inflight: set[_LoadKey] = set()
        # A failed loader is no longer discoverable, but its identity remains
        # until ownership teardown can CAS-restore the displaced predecessor.
        self._consumed_loaders: dict[_LoadKey, _Loader] = {}

    @staticmethod
    def current_scope_key() -> str:
        return hermes_home_key()

    def _scope_maps(
        self, scope: Optional[str], *, create: bool = False
    ) -> tuple[dict[str, BackgroundServiceEntry], dict[str, _Loader]]:
        if scope is None:
            return self._entries, self._deferred
        if create:
            return self._scoped_entries.setdefault(scope, {}), self._scoped_deferred.setdefault(scope, {})
        return self._scoped_entries.get(scope, {}), self._scoped_deferred.get(scope, {})

    def _registration_state(
        self, scope: Optional[str], name: str, *, create: bool = False
    ) -> tuple[Optional[BackgroundServiceEntry], Optional[_Loader]]:
        """(entry, loader) for *name*; the loader falls back to in-flight, then consumed."""
        entries, deferred = self._scope_maps(scope, create=create)
        entry = entries.get(name)
        loader = deferred.get(name)
        if entry is None and loader is None:
            loader = self._inflight_loaders.get((scope, name)) or self._consumed_loaders.get((scope, name))
        return entry, loader

    def _prune_scope(self, scope: Optional[str]) -> None:
        for maps in (self._scoped_entries, self._scoped_deferred) if scope is not None else ():
            if not maps.get(scope):
                maps.pop(scope, None)

    # -- deferred loading ----------------------------------------------------

    def register_deferred(self, name: str, loader: _Loader, *, scope: Optional[str] = None) -> None:
        """Register a lazy loader (imports the plugin module, which must call :meth:`register`);
        runs at most once, on first lookup; a concrete registration drops it."""
        with self._lock:
            entries, deferred = self._scope_maps(scope, create=True)
            self._consumed_loaders.pop((scope, name), None)
            if name not in entries:
                deferred[name] = loader

    def snapshot_registration(
        self, name: str, *, scope: Optional[str] = None
    ) -> tuple[Optional[BackgroundServiceEntry], Optional[_Loader]]:
        """Concrete and deferred state for *name* without resolving it, so the plugin ledger can
        restore a deferred loader displaced by a concrete registration without importing it."""
        with self._lock:
            return self._registration_state(scope, name)

    def restore_registration(
        self, name: str, current: tuple[Optional[BackgroundServiceEntry], Optional[_Loader]],
        previous: tuple[Optional[BackgroundServiceEntry], Optional[_Loader]], *, scope: Optional[str] = None,
    ) -> bool:
        """Restore a registration if its full state is still *current* (CAS): a later
        registration is never removed, and deferred loaders are part of the state."""
        with self._lock:
            entry, loader = self._registration_state(scope, name, create=True)
            if entry is not current[0] or loader is not current[1]:
                return False
            load_key = (scope, name)
            for mapping, value in zip(self._scope_maps(scope), previous):
                if value is None:
                    mapping.pop(name, None)
                else:
                    mapping[name] = value
            if load_key in self._inflight:
                self._cancelled_inflight.add(load_key)
            self._consumed_loaders.pop(load_key, None)
            self._prune_scope(scope)
            return True

    def _resolve(self, name: str, scope: Optional[str] = None) -> None:
        """Run the deferred loader for *name* if one is pending."""
        loader: Optional[_Loader] = None
        is_loader = False
        with self._lock:
            active_scope = scope or self.current_scope_key()
            entries, deferred = self._scope_maps(active_scope)
            scoped_key = (active_scope, name)
            global_key = (None, name)
            event = self._inflight.get(scoped_key)
            load_key = scoped_key
            if event is None and name not in entries:
                loader = deferred.pop(name, None)
            if event is None and loader is None and name not in entries:
                load_key = global_key
                event = self._inflight.get(global_key)
                if event is None:
                    loader = self._deferred.pop(name, None)
            if event is None and loader is not None:
                event = threading.Event()
                self._inflight[load_key] = event
                self._inflight_loaders[load_key] = loader
                self._inflight_owners[load_key] = threading.get_ident()
                is_loader = True
            if event is None:
                return
            if not is_loader and self._inflight_owners.get(load_key) == threading.get_ident():
                logger.warning("Deferred background service '%s' recursively requested while loading", name)
                return
        if not is_loader:
            event.wait()
            # Teardown may have restored an older deferred generation while cancelling the one
            # we waited for; resolve that predecessor instead of a one-shot false negative.
            self._resolve(name, active_scope)
            return
        try:
            loader()
        except Exception as e:
            logger.warning("Deferred load of background service '%s' failed: %s", name, e, exc_info=True)
        finally:
            with self._lock:
                was_cancelled = load_key in self._cancelled_inflight
                entries, deferred = self._scope_maps(load_key[0])
                if not was_cancelled and name not in entries and name not in deferred:
                    self._consumed_loaders[load_key] = loader
                self._inflight.pop(load_key, None)
                self._inflight_loaders.pop(load_key, None)
                self._inflight_owners.pop(load_key, None)
                self._cancelled_inflight.discard(load_key)
                event.set()
        if was_cancelled:
            self._resolve(name, active_scope)

    def is_deferred_load_cancelled(self, name: str, *, scope: Optional[str] = None) -> bool:
        """Whether ownership teardown cancelled an in-flight loader."""
        with self._lock:
            return (scope, name) in self._cancelled_inflight

    def _resolve_all(self) -> None:
        """Run every pending deferred loader (only ``all_entries``/``plugin_entries`` call this)."""
        active_scope = self.current_scope_key()
        with self._lock:
            _entries, scoped_deferred = self._scope_maps(active_scope)
            scoped_names = set(scoped_deferred)
            global_names = set(self._deferred)
            for inflight_scope, name in self._inflight:
                if inflight_scope == active_scope:
                    scoped_names.add(name)
                elif inflight_scope is None:
                    global_names.add(name)
        # Load outside the registry lock; each name has an in-flight event so concurrent
        # readers wait for the same materialization.
        for name in (*sorted(scoped_names), *sorted(global_names)):
            self._resolve(name, active_scope)

    def register(self, entry: BackgroundServiceEntry, *, scope: Optional[str] = None) -> None:
        """Register a background service entry (last writer wins on name clash
        WITHIN a scope; distinct profile scopes never overwrite each other)."""
        with self._lock:
            if scope is None and entry.source == "plugin":
                scope = (
                    _caller_plugin_scope()
                    or _plugin_scope_from_callable(entry.service_factory)
                    or _plugin_scope_from_callable(entry.check_fn)
                )
            # A concrete registration supersedes any pending deferred loader.
            entries, deferred = self._scope_maps(scope, create=True)
            self._consumed_loaders.pop((scope, entry.name), None)
            deferred.pop(entry.name, None)
            prev = entries.get(entry.name)
            if prev is not None:
                logger.info(
                    "Background service '%s' re-registered (was %s, now %s)",
                    entry.name, prev.source, entry.source,
                )
            entries[entry.name] = entry
            logger.debug("Registered background service: %s (%s)", entry.name, entry.source)

    def unregister(self, name: str, *, scope: Optional[str] = None) -> bool:
        """Remove a service entry. Returns True if it existed."""
        with self._lock:
            inferred_scope = scope if scope is not None else _caller_plugin_scope()
            active_scope = inferred_scope or self.current_scope_key()
            entries, deferred = self._scope_maps(active_scope)
            if inferred_scope is not None or name in entries or name in deferred:
                deferred.pop(name, None)
                removed = entries.pop(name, None) is not None
                self._prune_scope(active_scope)
                return removed
            self._deferred.pop(name, None)
            return self._entries.pop(name, None) is not None

    def _load_pending(self, scope: str, name: str) -> bool:
        """True when a lookup of *name* must run/await a deferred loader (lock held)."""
        _entries, deferred = self._scope_maps(scope)
        return (
            name in deferred or (name not in self._entries and name in self._deferred)
            or (scope, name) in self._inflight or (None, name) in self._inflight
        )

    def get(self, name: str) -> Optional[BackgroundServiceEntry]:
        """Look up a service entry by name (current profile scope, then global)."""
        scope = self.current_scope_key()
        with self._lock:
            entries, _deferred = self._scope_maps(scope)
            needs_resolve = name not in entries and self._load_pending(scope, name)
        if needs_resolve:
            self._resolve(name, scope)
        with self._lock:
            entries, _deferred = self._scope_maps(scope)
            return entries.get(name) or self._entries.get(name)

    def all_entries(self) -> list[BackgroundServiceEntry]:
        """Return all registered service entries (current profile scope overlaying global)."""
        self._resolve_all()
        with self._lock:
            return list({**self._entries, **self._scoped_entries.get(self.current_scope_key(), {})}.values())

    def plugin_entries(self) -> list[BackgroundServiceEntry]:
        """Return only plugin-registered service entries."""
        return [e for e in self.all_entries() if e.source == "plugin"]

    def registered_names(self) -> set[str]:
        """Concrete and deferred names (current profile scope AND process-global, like
        ``is_registered``) without loading services."""
        with self._lock:
            entries, deferred = self._scope_maps(self.current_scope_key())
            return entries.keys() | deferred.keys() | self._entries.keys() | self._deferred.keys()

    def is_registered(self, name: str) -> bool:
        # A deferred (not-yet-imported) service still counts as registered so cheap
        # membership checks never trigger a heavy import.
        with self._lock:
            scope = self.current_scope_key()
            entries, _deferred = self._scope_maps(scope)
            return name in entries or name in self._entries or self._load_pending(scope, name)

    def create_service(
        self, name: str, config: dict, gateway_runner: Any
    ) -> Optional[Any]:
        """Create a service instance for the given service name.

        Resolution is scope-aware via :meth:`get`, so a gateway running under
        profile A never consumes a factory registered by profile B.

        Returns ``None`` if:

        * No entry is registered for ``name``.
        * ``check_fn()`` returns False (missing deps).
        * ``validate_config()`` returns False (misconfigured).
        * The factory raises an exception.
        """
        entry = self.get(name)
        if entry is None:
            return None

        try:
            deps_ok = bool(entry.check_fn())
        except Exception as e:
            # A raising check_fn must not escape — the caller iterates all
            # configured services and an escaped exception would skip every
            # service after this one.
            logger.warning(
                "Background service '%s' dependency check raised: %s",
                entry.label,
                e,
            )
            deps_ok = False
        if not deps_ok:
            hint = f" ({entry.install_hint})" if entry.install_hint else ""
            logger.warning(
                "Background service '%s' requirements not met%s",
                entry.label,
                hint,
            )
            return None

        if entry.validate_config is not None:
            try:
                if not entry.validate_config(config):
                    logger.warning(
                        "Background service '%s' config validation failed",
                        entry.label,
                    )
                    return None
            except Exception as e:
                logger.warning(
                    "Background service '%s' config validation error: %s",
                    entry.label,
                    e,
                )
                return None

        try:
            service = entry.service_factory(config, gateway_runner)
            return service
        except Exception as e:
            logger.error(
                "Failed to create background service '%s': %s",
                entry.label,
                e,
                exc_info=True,
            )
            return None


# Module-level singleton — same pattern as platform_registry.
service_registry = BackgroundServiceRegistry()

"""Platform adapter registry.

Adapters (built-in and plugin) self-register here so the gateway can discover and
instantiate them without hardcoded if/elif chains. Plugins register via
``PluginContext.register_platform()``; ``GatewayRunner._create_adapter()`` consults the
registry first, then the legacy built-in path. Plugin side: ``platform_registry
.register(PlatformEntry(...))``; gateway side: ``create_adapter("irc", platform_config)``.
"""

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

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
class PlatformEntry:
    """Metadata and factory for a single platform adapter."""

    name: str  # config.yaml identifier (e.g. "irc")
    label: str  # human-readable
    adapter_factory: Callable[[Any], Any]  # PlatformConfig -> adapter (factory allows custom init)
    # PASSIVE dependency probe (deps importable RIGHT NOW); must be side-effect free since it
    # runs from status displays and the config enablement pass, which may never pip-install.
    check_fn: Callable[[], bool]
    validate_config: Optional[Callable[[Any], bool]] = None  # None = let connect() fail descriptively
    # ACTIVE installer, run by ``create_adapter()`` only when ``check_fn`` is False (platform
    # enabled+configured, about to connect); None = a False check_fn is a hard block. Split
    # from check_fn because one field either installed from every status display or never.
    # ACTIVE dependency installer: make the platform's dependencies available, installing them (pip /
    # lazy_deps) if needed. Returns True once deps are importable, False if they could not be installed.
    # None = no auto-install; a False ``check_fn`` is then a hard block (correct for platforms with no
    # optional deps). Why two fields (#79812): when the ACTIVE installer was registered as ``check_fn``,
    # every status display pip-installed SDKs as a side effect (desktop boot-loop at 94%, see
    # gateway/config.py enablement comments); when the PASSIVE probe was registered instead,
    # ``create_adapter()`` returned None before ``connect()`` could lazy-install, so the deps never
    # installed at all (Teams deadlock). Splitting the two roles makes both call sites correct by
    # construction.
    ensure_deps_fn: Optional[Callable[[], bool]] = None
    # Connected/configured for this PlatformConfig (``get_connected_platforms``, setup UI);
    # None falls back to ``validate_config`` or ``check_fn``.
    is_connected: Optional[Callable[[Any], bool]] = None
    required_env: list = field(default_factory=list)  # ``hermes setup`` display
    install_hint: str = ""  # shown when check_fn is False
    setup_fn: Optional[Callable[[], None]] = None  # None = _setup_standard_platform / env display
    source: str = "plugin"  # "builtin" or "plugin"
    plugin_name: str = ""  # owning manifest so ``hermes gateway setup`` can auto-enable it
    allowed_users_env: str = ""  # comma-separated allowed user IDs (_is_user_authorized)
    allow_all_env: str = ""  # truthy "allow everyone" switch
    max_message_length: int = 0  # smart-chunking cap; 0 = no limit
    pii_safe: bool = False  # session descriptions redact PII (phone numbers, etc.)
    emoji: str = "🔌"  # CLI/gateway display
    allow_update_command: bool = True  # /update may be issued from this platform
    platform_hint: str = ""  # injected into the system prompt; empty = none
    # ``() -> Optional[dict]`` of ``extra`` fields to seed when auto-enabled from env; runs in
    # ``_apply_env_overrides`` BEFORE adapter construction so ``gateway status`` sees it.
    env_enablement_fn: Optional[Callable[[], Optional[dict]]] = None
    # YAML->env bridge ``(yaml_cfg, platform_cfg) -> Optional[dict]`` merged into ``extra``; runs
    # after the shared-key loop, before ``_apply_env_overrides``. May set ``os.environ`` (guard
    # with ``not os.getenv(...)`` to keep env > YAML). Contract: docs/developer-guide/adding-platform-adapters.md.
    apply_yaml_config_fn: Optional[Callable[[dict, dict], Optional[dict]]] = None
    cron_deliver_env_var: str = ""  # home-channel env var read for cron ``deliver=<name>``
    # ``(target_ref) -> Optional[(chat_id, thread_id)]`` run before channel-directory
    # fallback so plugins can declare native target syntax; None = continue resolution.
    parse_target_ref_fn: Optional[Callable[[str], Optional[tuple[str, Optional[str]]]]] = None
    # Post-resolution validation: True accept, False reject, non-empty str = reject + diagnostic.
    validate_target_ref_fn: Optional[Callable[[str], bool | str]] = None
    # Whole-request delivery ``(args, normalized_chat_id, platform_name, pconfig)``, sync or
    # async; prefer standalone_sender_fn when the standard send contract suffices.
    send_message_handler: Optional[Callable[[dict, str, str, Any], Any]] = None
    # Out-of-process sender for cron without a co-resident gateway:
    # ``async (pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False)
    # -> {"success": True, "message_id": ...} | {"error": str}``.
    standalone_sender_fn: Optional[Callable[..., Awaitable[dict]]] = None


class PlatformRegistry:
    """Central registry of platform adapters. Registrations are serialized; concurrent
    lazy lookups share an in-flight event while the loader runs outside the registry lock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, PlatformEntry] = {}  # process-global (e.g. the built-in relay)
        # Plugin adapters are isolated per resolved HERMES_HOME and overlay the
        # process-global entries for lookups in that profile's runtime scope.
        self._scoped_entries: dict[str, dict[str, PlatformEntry]] = {}
        # Deferred loaders: name -> callable importing the owning plugin module (which calls
        # register()); eagerly importing ~20 SDK-heavy adapters added seconds to every
        # `hermes` invocation, so the import happens only when a lookup asks for it.
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
    ) -> tuple[dict[str, PlatformEntry], dict[str, _Loader]]:
        if scope is None:
            return self._entries, self._deferred
        if create:
            return self._scoped_entries.setdefault(scope, {}), self._scoped_deferred.setdefault(scope, {})
        return self._scoped_entries.get(scope, {}), self._scoped_deferred.get(scope, {})

    def _registration_state(
        self, scope: Optional[str], name: str, *, create: bool = False
    ) -> tuple[Optional[PlatformEntry], Optional[_Loader]]:
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
    ) -> tuple[Optional[PlatformEntry], Optional[_Loader]]:
        """Concrete and deferred state for *name* without resolving it, so the plugin ledger can
        restore a deferred loader displaced by a concrete registration without importing it."""
        with self._lock:
            return self._registration_state(scope, name)

    def restore_registration(
        self, name: str, current: tuple[Optional[PlatformEntry], Optional[_Loader]],
        previous: tuple[Optional[PlatformEntry], Optional[_Loader]], *, scope: Optional[str] = None,
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
                logger.warning("Deferred platform '%s' recursively requested while loading", name)
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
            logger.warning("Deferred load of platform '%s' failed: %s", name, e, exc_info=True)
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
        """Run every pending deferred loader (only ``all_entries``/``plugin_entries`` call this;
        CLI chat never iterates the full set)."""
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

    def register(self, entry: PlatformEntry, *, scope: Optional[str] = None) -> None:
        """Register a platform adapter entry (last writer wins on name clash)."""
        with self._lock:
            if scope is None and entry.source == "plugin":
                scope = (
                    _caller_plugin_scope()
                    or _plugin_scope_from_callable(entry.adapter_factory)
                    or _plugin_scope_from_callable(entry.check_fn)
                )
            # A concrete registration supersedes any pending deferred loader.
            entries, deferred = self._scope_maps(scope, create=True)
            self._consumed_loaders.pop((scope, entry.name), None)
            deferred.pop(entry.name, None)
            prev = entries.get(entry.name)
            if prev is not None:
                logger.info("Platform '%s' re-registered (was %s, now %s)", entry.name, prev.source, entry.source)
            entries[entry.name] = entry
            logger.debug("Registered platform adapter: %s (%s)", entry.name, entry.source)

    def unregister(self, name: str, *, scope: Optional[str] = None) -> bool:
        """Remove a platform entry. Returns True if it existed."""
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

    def get(self, name: str) -> Optional[PlatformEntry]:
        """Look up a platform entry by name."""
        scope = self.current_scope_key()
        with self._lock:
            entries, _deferred = self._scope_maps(scope)
            needs_resolve = name not in entries and self._load_pending(scope, name)
        if needs_resolve:
            self._resolve(name, scope)
        with self._lock:
            entries, _deferred = self._scope_maps(scope)
            return entries.get(name) or self._entries.get(name)

    def all_entries(self) -> list[PlatformEntry]:
        """Return all registered platform entries."""
        self._resolve_all()
        with self._lock:
            return list({**self._entries, **self._scoped_entries.get(self.current_scope_key(), {})}.values())

    def plugin_entries(self) -> list[PlatformEntry]:
        """Return only plugin-registered platform entries."""
        return [e for e in self.all_entries() if e.source == "plugin"]

    def registered_names(self) -> set[str]:
        """Concrete and deferred names (current profile scope AND process-global, like
        ``is_registered``) without loading adapters."""
        with self._lock:
            entries, deferred = self._scope_maps(self.current_scope_key())
            return entries.keys() | deferred.keys() | self._entries.keys() | self._deferred.keys()

    def is_registered(self, name: str) -> bool:
        # A deferred (not-yet-imported) platform still counts as registered so cheap membership
        # checks (toolset resolution, webhook deliver-target checks) never trigger a heavy import.
        with self._lock:
            scope = self.current_scope_key()
            entries, _deferred = self._scope_maps(scope)
            return name in entries or name in self._entries or self._load_pending(scope, name)

    def create_adapter(self, name: str, config: Any) -> Optional[Any]:
        """Create an adapter instance for *name*; None when no entry exists, deps are missing
        and cannot be installed, ``validate_config`` fails, or the factory raises."""
        entry = self.get(name)
        if entry is None:
            return None
        def _probe(fn: Callable[[], Any], failure_msg: str) -> bool:
            try:
                return bool(fn())
            except Exception as e:
                logger.warning(failure_msg, entry.label, e)
                return False
        deps_ok = _probe(entry.check_fn, "Platform '%s' check_fn raised: %s")
        if not deps_ok and entry.ensure_deps_fn is not None:
            # The ONE place the active installer runs: the platform is enabled+configured
            # and about to connect, so an install is what the user wants.
            logger.info("Platform '%s' dependencies missing — attempting install...", entry.label)
            deps_ok = _probe(entry.ensure_deps_fn, "Platform '%s' dependency install raised: %s")
        if not deps_ok:
            hint = f" ({entry.install_hint})" if entry.install_hint else ""
            logger.warning("Platform '%s' requirements not met%s", entry.label, hint)
            return None
        if entry.validate_config is not None:
            try:
                if not entry.validate_config(config):
                    logger.warning("Platform '%s' config validation failed", entry.label)
                    return None
            except Exception as e:
                logger.warning("Platform '%s' config validation error: %s", entry.label, e)
                return None
        try:
            return entry.adapter_factory(config)
        except Exception as e:
            logger.error("Failed to create adapter for platform '%s': %s", entry.label, e, exc_info=True)
            return None


# Module-level singleton
platform_registry = PlatformRegistry()

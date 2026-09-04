"""
Platform Adapter Registry

Allows platform adapters (built-in and plugin) to self-register so the gateway
can discover and instantiate them without hardcoded if/elif chains.

Built-in adapters continue to use the existing if/elif in _instantiate_adapter()
for now.  Plugin adapters register here via PluginContext.register_platform()
and are looked up first -- if nothing is found the gateway falls through to
the legacy code path.  GatewayRunner._create_adapter() wraps both paths and
binds every successful adapter to its runner.

Usage (plugin side):

    from gateway.platform_registry import platform_registry, PlatformEntry

    platform_registry.register(PlatformEntry(
        name="irc",
        label="IRC",
        adapter_factory=lambda cfg: IRCAdapter(cfg),
        check_fn=check_requirements,
        validate_config=lambda cfg: bool(cfg.extra.get("server")),
        required_env=["IRC_SERVER"],
        install_hint="pip install irc",
    ))

Usage (gateway side):

    adapter = platform_registry.create_adapter("irc", platform_config)
"""

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)


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
    return _plugin_scope_from_callable(
        type("_Caller", (), {"__module__": module_name})
    )


@dataclass
class PlatformEntry:
    """Metadata and factory for a single platform adapter."""

    # Identifier used in config.yaml (e.g. "irc", "viber").
    name: str

    # Human-readable label (e.g. "IRC", "Viber").
    label: str

    # Factory callable: receives a PlatformConfig, returns an adapter instance.
    # Using a factory instead of a bare class lets plugins do custom init
    # (e.g. passing extra kwargs, wrapping in try/except).
    adapter_factory: Callable[[Any], Any]

    # PASSIVE dependency probe: returns True when the platform's dependencies
    # are available RIGHT NOW.  Must be side-effect free — it is called from
    # status displays (``hermes setup``, ``hermes status``, the dashboard
    # readiness probe) and the config enablement pass, none of which may
    # trigger a pip install.  Put install logic in ``ensure_deps_fn`` instead.
    check_fn: Callable[[], bool]

    # Optional: given a PlatformConfig, is it properly configured?
    # If None, the registry skips config validation and lets the adapter
    # fail at connect() time with a descriptive error.
    validate_config: Optional[Callable[[Any], bool]] = None

    # ACTIVE dependency installer: make the platform's dependencies available,
    # installing them (pip / lazy_deps) if needed.  Returns True once deps are
    # importable, False if they could not be installed.  Called by
    # ``create_adapter()`` when ``check_fn`` returns False — i.e. exactly at
    # the moment the gateway is about to bring the platform up and the user
    # has it enabled/configured.  None = no auto-install; a False ``check_fn``
    # is then a hard block (correct for platforms with no optional deps).
    #
    # Why two fields (#79812): when the ACTIVE installer was registered as
    # ``check_fn``, every status display pip-installed SDKs as a side effect
    # (desktop boot-loop at 94%, see gateway/config.py enablement comments);
    # when the PASSIVE probe was registered instead, ``create_adapter()``
    # returned None before ``connect()`` could lazy-install, so the deps
    # never installed at all (Teams deadlock).  Splitting the two roles makes
    # both call sites correct by construction.
    ensure_deps_fn: Optional[Callable[[], bool]] = None

    # Optional: given a PlatformConfig, is the platform connected/enabled?
    # Used by ``GatewayConfig.get_connected_platforms()`` and setup UI status.
    # If None, falls back to ``validate_config`` or ``check_fn``.
    is_connected: Optional[Callable[[Any], bool]] = None

    # Env vars this platform needs (for ``hermes setup`` display).
    required_env: list = field(default_factory=list)

    # Hint shown when check_fn returns False.
    install_hint: str = ""

    # Optional setup function for interactive configuration.
    # Signature: () -> None (prompts user, saves env vars).
    # If None, falls back to _setup_standard_platform (needs token_var + vars)
    # or a generic "set these env vars" display.
    setup_fn: Optional[Callable[[], None]] = None

    # "builtin" or "plugin"
    source: str = "plugin"

    # Name of the plugin manifest that registered this entry (empty for
    # built-ins).  Used by ``hermes gateway setup`` to auto-enable the
    # owning plugin when the user configures its platform.
    plugin_name: str = ""

    # ── Auth env var names (for _is_user_authorized integration) ──
    # E.g. "IRC_ALLOWED_USERS" — checked for comma-separated user IDs.
    allowed_users_env: str = ""
    # Optional chat-scoped group allowlist env var. Generic authz consumes
    # this instead of hardcoding plugin platform names.
    group_allowed_chats_env: str = ""
    # E.g. "IRC_ALLOW_ALL_USERS" — if truthy, all users authorized.
    allow_all_env: str = ""

    # ── Message limits ──
    # Max message length for smart-chunking.  0 = no limit.
    max_message_length: int = 0

    # ── Privacy ──
    # If True, session descriptions redact PII (phone numbers, etc.)
    pii_safe: bool = False

    # ── Display ──
    # Emoji for CLI/gateway display (e.g. "💬")
    emoji: str = "🔌"

    # Whether this platform should appear in _UPDATE_ALLOWED_PLATFORMS
    # (allows /update command from this platform).
    allow_update_command: bool = True

    # ── LLM guidance ──
    # Platform hint injected into the system prompt (e.g. "You are on IRC.
    # Do not use markdown.").  Empty string = no hint.
    platform_hint: str = ""

    # ── Env-driven auto-configuration ──
    # Optional: read env vars, return a dict of ``PlatformConfig.extra`` fields
    # to seed when the platform is auto-enabled.  Called during
    # ``_apply_env_overrides`` BEFORE the adapter is constructed, so
    # ``gateway status`` etc. can reflect env-only configuration without
    # instantiating the adapter.  Return ``None`` (or an empty dict) to skip.
    # Signature: () -> Optional[dict[str, Any]]
    env_enablement_fn: Optional[Callable[[], Optional[dict]]] = None

    # ── YAML→env config bridge ──
    # Optional: translate this platform's ``config.yaml`` keys into env vars
    # and/or seed ``PlatformConfig.extra`` directly.  Lets a plugin own its
    # YAML config translation instead of forcing core ``gateway/config.py``
    # to know every platform's schema.
    #
    # Signature: (yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]
    # Called from ``load_gateway_config()`` after the generic shared-key loop
    # and before ``_apply_env_overrides``.  Mutating ``os.environ`` is allowed
    # (use ``not os.getenv(...)`` guards to preserve env > YAML precedence);
    # any returned dict is merged into ``PlatformConfig.extra``.  Exceptions
    # are caught and logged at debug level.
    # See website/docs/developer-guide/adding-platform-adapters.md for the
    # full contract and a worked example.
    apply_yaml_config_fn: Optional[Callable[[dict, dict], Optional[dict]]] = None

    # Optional: home-channel env var name for cron/notification delivery
    # (e.g. ``"IRC_HOME_CHANNEL"``).  When set, ``cron.scheduler`` treats this
    # platform as a valid ``deliver=<name>`` target and reads the env var to
    # resolve the default chat/room ID.  Empty = no cron home-channel support.
    cron_deliver_env_var: str = ""

    # ── Target parsing ──
    # Optional: callable that parses a raw target string for this platform into
    # a (chat_id, thread_id) tuple, or None if the string is not a recognized
    # explicit target.  Invoked by ``tools/send_message_tool._parse_target_ref``
    # before channel-directory fallback so plugin platforms can declare their
    # own native target syntax (e.g. ``fmsg:@alice@example.com``) without
    # hard-casing in Hermes core.
    #
    # Signature:
    #     (target_ref: str) -> Optional[tuple[str, Optional[str]]]
    #
    # If the callable returns None the target proceeds to channel-directory
    # resolution. No opaque fallback is applied.
    parse_target_ref_fn: Optional[Callable[[str], Optional[tuple[str, Optional[str]]]]] = None

    # Optional validation applied after parsing/normalization or
    # channel-directory resolution. Return True to accept, False to reject, or
    # a non-empty string to reject with that diagnostic.
    validate_target_ref_fn: Optional[Callable[[str], bool | str]] = None

    # Optional whole-request handler for custom platform delivery. Receives
    # (args, normalized_chat_id, platform_name, pconfig) and may be sync/async.
    # Prefer standalone_sender_fn when the standard send contract is enough.
    send_message_handler: Optional[Callable[[dict, str, str, Any], Any]] = None

    # ── Standalone (out-of-process) sending ──
    # Optional: async coroutine that delivers a message without a live
    # gateway adapter.  Called by ``tools/send_message_tool._send_via_adapter``
    # when ``cron`` runs in a separate process from the gateway and the
    # in-process adapter weakref is therefore ``None``.
    #
    # Signature:
    #     async (pconfig, chat_id, message, *, thread_id=None,
    #            media_files=None, force_document=False) -> dict
    #
    # Returns ``{"success": True, "message_id": ...}`` on success or
    # ``{"error": str}`` on failure.  Plugin authors typically open an
    # ephemeral connection / acquire a fresh OAuth token, send, and close.
    # Without this hook, plugin platforms cannot serve as cron ``deliver=``
    # targets when the gateway is not co-resident with the cron process.
    standalone_sender_fn: Optional[Callable[..., Awaitable[dict]]] = None


class PlatformRegistry:
    """Central registry of platform adapters.

    Registrations are serialized, and concurrent lazy lookups share an
    in-flight event while the loader runs outside the registry lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Process-global registrations (for example the built-in relay).
        self._entries: dict[str, PlatformEntry] = {}
        # Plugin adapters are isolated per resolved HERMES_HOME and overlay the
        # process-global entries for lookups in that profile's runtime scope.
        self._scoped_entries: dict[str, dict[str, PlatformEntry]] = {}
        # Deferred platform loaders: name -> zero-arg callable that imports the
        # owning plugin module (which calls register() and populates _entries).
        #
        # Why this exists: platform adapter modules import heavy, platform-
        # specific SDKs at module level (lark_oapi, microsoft_teams, discord.py,
        # slack_bolt, ...). Eagerly loading all ~20 bundled platform plugins at
        # plugin-discovery time added several seconds to *every* `hermes`
        # invocation -- including plain `hermes chat`, which never touches any
        # gateway platform. Discovery now registers a cheap deferred loader per
        # platform; the real module is imported only when a registry lookup
        # actually asks for that platform (gateway start, cron delivery,
        # `hermes setup`/`gateway status`, send_message).
        self._deferred: dict[str, Callable[[], None]] = {}
        self._scoped_deferred: dict[str, dict[str, Callable[[], None]]] = {}
        self._inflight: dict[tuple[Optional[str], str], threading.Event] = {}
        self._inflight_loaders: dict[
            tuple[Optional[str], str], Callable[[], None]
        ] = {}
        self._inflight_owners: dict[tuple[Optional[str], str], int] = {}
        self._cancelled_inflight: set[tuple[Optional[str], str]] = set()
        # A failed loader is no longer discoverable, but its identity remains
        # until ownership teardown can CAS-restore the displaced predecessor.
        self._consumed_loaders: dict[
            tuple[Optional[str], str], Callable[[], None]
        ] = {}

    @staticmethod
    def current_scope_key() -> str:
        return hermes_home_key()

    def _scope_maps(
        self,
        scope: Optional[str],
        *,
        create: bool = False,
    ) -> tuple[dict[str, PlatformEntry], dict[str, Callable[[], None]]]:
        if scope is None:
            return self._entries, self._deferred
        if create:
            return (
                self._scoped_entries.setdefault(scope, {}),
                self._scoped_deferred.setdefault(scope, {}),
            )
        return (
            self._scoped_entries.get(scope, {}),
            self._scoped_deferred.get(scope, {}),
        )

    # -- deferred loading ----------------------------------------------------

    def register_deferred(
        self,
        name: str,
        loader: Callable[[], None],
        *,
        scope: Optional[str] = None,
    ) -> None:
        """Register a lazy loader for a platform that hasn't been imported yet.

        *loader* is a zero-arg callable that imports the owning plugin module,
        which is expected to call :meth:`register` with the real entry for
        *name*.  The loader runs at most once, the first time *name* is looked
        up (or when the full entry list is materialized).  A real entry that is
        registered directly (e.g. a built-in) takes precedence -- the deferred
        loader is then dropped.
        """
        with self._lock:
            entries, deferred = self._scope_maps(scope, create=True)
            self._consumed_loaders.pop((scope, name), None)
            if name in entries:
                # Already concretely registered; no need to defer.
                return
            deferred[name] = loader

    def snapshot_registration(
        self,
        name: str,
        *,
        scope: Optional[str] = None,
    ) -> tuple[Optional[PlatformEntry], Optional[Callable[[], None]]]:
        """Return the concrete and deferred state for *name* without resolving it.

        This host-facing snapshot lets the plugin ledger restore a deferred
        platform loader that a concrete registration displaced, without
        importing the displaced adapter as a side effect of taking the
        snapshot.
        """
        with self._lock:
            entries, deferred = self._scope_maps(scope)
            loader = deferred.get(name)
            if entries.get(name) is None and loader is None:
                loader = self._inflight_loaders.get((scope, name))
            if entries.get(name) is None and loader is None:
                loader = self._consumed_loaders.get((scope, name))
            return entries.get(name), loader

    def restore_registration(
        self,
        name: str,
        current: tuple[Optional[PlatformEntry], Optional[Callable[[], None]]],
        previous: tuple[Optional[PlatformEntry], Optional[Callable[[], None]]],
        *,
        scope: Optional[str] = None,
    ) -> bool:
        """Restore a platform registration if its full state is still current.

        The identity checks protect a later registration from being removed
        while still allowing an unloaded override to reveal the registration
        it displaced.  Both concrete entries and deferred loaders are part of
        the state because bundled platform plugins load lazily.
        """
        with self._lock:
            entries, deferred = self._scope_maps(scope, create=True)
            entry = entries.get(name)
            loader = deferred.get(name)
            load_key = (scope, name)
            if entry is None and loader is None:
                loader = self._inflight_loaders.get(load_key)
            if entry is None and loader is None:
                loader = self._consumed_loaders.get(load_key)
            current_state = (entry, loader)
            is_current = not (
                current_state[0] is not current[0]
                or current_state[1] is not current[1]
            )
            if not is_current:
                return False

            previous_entry, previous_loader = previous
            if previous_entry is None:
                entries.pop(name, None)
            else:
                entries[name] = previous_entry
            if previous_loader is None:
                deferred.pop(name, None)
            else:
                deferred[name] = previous_loader
            if load_key in self._inflight:
                self._cancelled_inflight.add(load_key)
            self._consumed_loaders.pop(load_key, None)
            if scope is not None:
                if not entries:
                    self._scoped_entries.pop(scope, None)
                if not deferred:
                    self._scoped_deferred.pop(scope, None)
            return True

    def _resolve(self, name: str, scope: Optional[str] = None) -> None:
        """Run the deferred loader for *name* if one is pending."""
        loader: Optional[Callable[[], None]] = None
        event: Optional[threading.Event] = None
        load_key: tuple[Optional[str], str]
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
                event = self._inflight.get(global_key)
                load_key = global_key
            if event is None and loader is None and name not in entries:
                loader = self._deferred.pop(name, None)
                load_key = global_key
            if event is None and loader is not None:
                event = threading.Event()
                self._inflight[load_key] = event
                self._inflight_loaders[load_key] = loader
                self._inflight_owners[load_key] = threading.get_ident()
                is_loader = True
            if event is None:
                return
            if (
                not is_loader
                and self._inflight_owners.get(load_key) == threading.get_ident()
            ):
                logger.warning(
                    "Deferred platform '%s' recursively requested while loading",
                    name,
                )
                return

        if not is_loader:
            event.wait()
            # Teardown may have restored an older deferred generation while
            # cancelling the one we waited for. Resolve that predecessor in
            # the same lookup instead of returning a one-shot false negative.
            self._resolve(name, active_scope)
            return

        try:
            loader()
        except Exception as e:
            logger.warning(
                "Deferred load of platform '%s' failed: %s",
                name,
                e,
                exc_info=True,
            )
        finally:
            with self._lock:
                was_cancelled = load_key in self._cancelled_inflight
                load_scope, _load_name = load_key
                entries, deferred = self._scope_maps(load_scope)
                if (
                    not was_cancelled
                    and name not in entries
                    and name not in deferred
                ):
                    self._consumed_loaders[load_key] = loader
                self._inflight.pop(load_key, None)
                self._inflight_loaders.pop(load_key, None)
                self._inflight_owners.pop(load_key, None)
                self._cancelled_inflight.discard(load_key)
                event.set()
        if was_cancelled:
            self._resolve(name, active_scope)

    def is_deferred_load_cancelled(
        self,
        name: str,
        *,
        scope: Optional[str] = None,
    ) -> bool:
        """Return whether ownership teardown cancelled an in-flight loader."""
        with self._lock:
            return (scope, name) in self._cancelled_inflight

    def _resolve_all(self) -> None:
        """Run every pending deferred loader.

        Used by the iterate-all accessors (``all_entries``/``plugin_entries``),
        which are only called by paths that genuinely need every adapter:
        gateway startup, ``hermes setup``/``gateway status``, channel
        directory.  CLI chat never iterates the full set.
        """
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
        # Load outside the registry lock; each name has an in-flight event so
        # concurrent readers wait for the same materialization.
        for name in sorted(scoped_names):
            self._resolve(name, active_scope)
        for name in sorted(global_names):
            self._resolve(name, active_scope)

    def register(
        self,
        entry: PlatformEntry,
        *,
        scope: Optional[str] = None,
    ) -> None:
        """Register a platform adapter entry.

        If an entry with the same name exists, it is replaced (last writer
        wins -- this lets plugins override built-in adapters if desired).
        """
        with self._lock:
            if scope is None and entry.source == "plugin":
                scope = _caller_plugin_scope()
                if scope is None:
                    scope = _plugin_scope_from_callable(entry.adapter_factory)
                if scope is None:
                    scope = _plugin_scope_from_callable(entry.check_fn)
            # A concrete registration supersedes any pending deferred loader.
            entries, deferred = self._scope_maps(scope, create=True)
            self._consumed_loaders.pop((scope, entry.name), None)
            deferred.pop(entry.name, None)
            if entry.name in entries:
                prev = entries[entry.name]
                logger.info(
                    "Platform '%s' re-registered (was %s, now %s)",
                    entry.name,
                    prev.source,
                    entry.source,
                )
            entries[entry.name] = entry
            logger.debug("Registered platform adapter: %s (%s)", entry.name, entry.source)

    def unregister(self, name: str, *, scope: Optional[str] = None) -> bool:
        """Remove a platform entry.  Returns True if it existed."""
        with self._lock:
            inferred_scope = scope if scope is not None else _caller_plugin_scope()
            active_scope = inferred_scope or self.current_scope_key()
            entries, deferred = self._scope_maps(active_scope)
            if inferred_scope is not None or name in entries or name in deferred:
                deferred.pop(name, None)
                removed = entries.pop(name, None) is not None
                if not entries:
                    self._scoped_entries.pop(active_scope, None)
                if not deferred:
                    self._scoped_deferred.pop(active_scope, None)
                return removed
            self._deferred.pop(name, None)
            return self._entries.pop(name, None) is not None

    def get(self, name: str) -> Optional[PlatformEntry]:
        """Look up a platform entry by name."""
        scope = self.current_scope_key()
        with self._lock:
            entries, deferred = self._scope_maps(scope)
            needs_resolve = name not in entries and (
                name in deferred
                or (name not in self._entries and name in self._deferred)
                or (scope, name) in self._inflight
                or (None, name) in self._inflight
            )
        if needs_resolve:
            self._resolve(name, scope)
        with self._lock:
            entries, _deferred = self._scope_maps(scope)
            return entries.get(name) or self._entries.get(name)

    def all_entries(self) -> list[PlatformEntry]:
        """Return all registered platform entries."""
        self._resolve_all()
        with self._lock:
            entries = dict(self._entries)
            entries.update(self._scoped_entries.get(self.current_scope_key(), {}))
            return list(entries.values())

    def plugin_entries(self) -> list[PlatformEntry]:
        """Return only plugin-registered platform entries."""
        self._resolve_all()
        return [e for e in self.all_entries() if e.source == "plugin"]

    def registered_names(self) -> set[str]:
        """Return concrete and deferred platform names without loading adapters.

        Mirrors ``is_registered()``'s scope semantics: names registered under
        the current profile scope AND process-global names both count. Plugin
        platforms register deferred loaders under a profile scope, so reading
        only the global maps would miss every plugin platform.
        """
        with self._lock:
            scope = self.current_scope_key()
            entries, deferred = self._scope_maps(scope)
            return (
                entries.keys()
                | deferred.keys()
                | self._entries.keys()
                | self._deferred.keys()
            )

    def is_registered(self, name: str) -> bool:
        # A deferred (not-yet-imported) platform still counts as registered --
        # the loader will materialize it on first real use.  This keeps cheap
        # membership checks (toolset resolution, webhook deliver-target checks)
        # from triggering a heavy import.
        with self._lock:
            scope = self.current_scope_key()
            entries, deferred = self._scope_maps(scope)
            return (
                name in entries
                or name in deferred
                or name in self._entries
                or name in self._deferred
                or (scope, name) in self._inflight
                or (None, name) in self._inflight
            )

    def create_adapter(self, name: str, config: Any) -> Optional[Any]:
        """Create an adapter instance for the given platform name.

        Returns None if:
        - No entry registered for *name*
        - check_fn() returns False and deps can't be installed
          (no ensure_deps_fn, or ensure_deps_fn() returned False)
        - validate_config() returns False (misconfigured)
        - The factory raises an exception
        """
        entry = self.get(name)
        if entry is None:
            return None

        deps_ok = False
        try:
            deps_ok = bool(entry.check_fn())
        except Exception as e:
            logger.warning(
                "Platform '%s' check_fn raised: %s", entry.label, e
            )
        if not deps_ok and entry.ensure_deps_fn is not None:
            # Deps missing but the platform can install them on demand.
            # This is the ONE place the active installer runs in the adapter
            # path: the platform is enabled+configured and the gateway is
            # about to connect it, so an install is what the user wants
            # (#79812 — Teams' installer previously lived behind this very
            # gate inside connect(), which could never be reached).
            logger.info(
                "Platform '%s' dependencies missing — attempting install...",
                entry.label,
            )
            try:
                deps_ok = bool(entry.ensure_deps_fn())
            except Exception as e:
                logger.warning(
                    "Platform '%s' dependency install raised: %s",
                    entry.label,
                    e,
                )
                deps_ok = False
        if not deps_ok:
            hint = f" ({entry.install_hint})" if entry.install_hint else ""
            logger.warning(
                "Platform '%s' requirements not met%s",
                entry.label,
                hint,
            )
            return None

        if entry.validate_config is not None:
            try:
                if not entry.validate_config(config):
                    logger.warning(
                        "Platform '%s' config validation failed",
                        entry.label,
                    )
                    return None
            except Exception as e:
                logger.warning(
                    "Platform '%s' config validation error: %s",
                    entry.label,
                    e,
                )
                return None

        try:
            adapter = entry.adapter_factory(config)
            return adapter
        except Exception as e:
            logger.error(
                "Failed to create adapter for platform '%s': %s",
                entry.label,
                e,
                exc_info=True,
            )
            return None


# Module-level singleton
platform_registry = PlatformRegistry()

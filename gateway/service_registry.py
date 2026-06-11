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

Usage (plugin side)::

    from gateway.service_registry import service_registry, BackgroundServiceEntry

    service_registry.register(BackgroundServiceEntry(
        name="nextcloud_notifications",
        label="Nextcloud Notifications",
        service_factory=lambda cfg, gateway: NextcloudNotificationService(cfg),
        check_fn=lambda: True,
        validate_config=lambda cfg: bool(cfg.get("enabled")),
    ))

Usage (gateway side)::

    svc = service_registry.create_service("nextcloud_notifications", cfg, self)
    if svc is not None:
        await svc.start()
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class BackgroundServiceEntry:
    """Metadata and factory for a single background service."""

    # Identifier used in config.yaml under ``services.<name>``.
    name: str

    # Human-readable label.
    label: str

    # Factory callable: receives (config_dict, gateway_runner) and returns
    # a service instance. The instance MUST expose ``async start()`` and
    # ``async stop()`` methods (see ``gateway.services.base.BaseService``
    # for the canonical interface). Using a factory instead of a bare class
    # lets plugins do custom init / dependency injection without subclassing.
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
    """Central registry of background services.

    Thread-safe for reads (dict lookups are atomic under GIL).
    Writes happen at startup during sequential plugin discovery.
    """

    def __init__(self) -> None:
        self._entries: dict[str, BackgroundServiceEntry] = {}

    def register(self, entry: BackgroundServiceEntry) -> None:
        """Register a background service entry.

        If an entry with the same name exists, it is replaced (last writer
        wins — lets plugins override built-in services if desired).
        """
        if entry.name in self._entries:
            prev = self._entries[entry.name]
            logger.info(
                "Background service '%s' re-registered (was %s, now %s)",
                entry.name,
                prev.source,
                entry.source,
            )
        self._entries[entry.name] = entry
        logger.debug(
            "Registered background service: %s (%s)", entry.name, entry.source
        )

    def unregister(self, name: str) -> bool:
        """Remove a service entry. Returns True if it existed."""
        return self._entries.pop(name, None) is not None

    def get(self, name: str) -> Optional[BackgroundServiceEntry]:
        return self._entries.get(name)

    def all_entries(self) -> list[BackgroundServiceEntry]:
        return list(self._entries.values())

    def plugin_entries(self) -> list[BackgroundServiceEntry]:
        return [e for e in self._entries.values() if e.source == "plugin"]

    def is_registered(self, name: str) -> bool:
        return name in self._entries

    def create_service(
        self, name: str, config: dict, gateway_runner: Any
    ) -> Optional[Any]:
        """Create a service instance for the given service name.

        Returns ``None`` if:

        * No entry is registered for ``name``.
        * ``check_fn()`` returns False (missing deps).
        * ``validate_config()`` returns False (misconfigured).
        * The factory raises an exception.
        """
        entry = self._entries.get(name)
        if entry is None:
            return None

        if not entry.check_fn():
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

"""Plugin manager teardown at profile lifecycle boundaries."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def _dispose_manager(plugins, manager) -> None:
    host = getattr(manager, "_plugin_host_instance", None)
    try:
        plugins._clear_plugin_submodules(manager)
    finally:
        try:
            manager.unload()
        finally:
            if host is not None:
                host.shutdown()


def _get_or_create_reentrant_manager(home_key: Path, thread_id: int):
    """Supply the reservation owner a manager without admitting another thread."""
    from hermes_cli import plugins

    manager = plugins._plugin_managers_by_home.get(home_key)
    if manager is None:
        manager = plugins.PluginManager(scope_key=plugins.hermes_home_key(home_key))
        plugins._plugin_managers_by_home[home_key] = manager
    plugins._plugin_manager = manager
    plugins._plugin_manager_teardown_owners[home_key] = (thread_id, manager)
    return manager


@contextmanager
def reserve_plugin_manager_for_home(home: Path) -> Iterator[tuple[bool, Exception | None]]:
    """Hold same-home lookups off through teardown and the caller's profile mutation.

    The yielded result is ``(had_manager, teardown_error)`` so callers can roll back
    inside the reservation if teardown failed. Errors from an owner-reentrant manager
    created during the reservation are logged during final cleanup.
    """
    from hermes_cli import plugins

    try:
        home_key = Path(home).expanduser().resolve()
    except (OSError, RuntimeError):
        home_key = Path(home).expanduser()

    thread_id = threading.get_ident()
    reentrant = False
    manager = None
    with plugins._plugin_manager_teardown_condition:
        while home_key in plugins._plugin_manager_teardown_owners:
            owner, _manager = plugins._plugin_manager_teardown_owners[home_key]
            if owner == thread_id:
                reentrant = True
                break
            plugins._plugin_manager_teardown_condition.wait()
        if not reentrant:
            manager = plugins._plugin_managers_by_home.get(home_key)
            if manager is None and plugins._plugin_manager is not None:
                manager_home = getattr(plugins._plugin_manager, "home_path", None)
                if manager_home is not None:
                    try:
                        matches = Path(manager_home).expanduser().resolve() == home_key
                    except (OSError, RuntimeError):
                        matches = Path(manager_home).expanduser() == home_key
                    if matches:
                        manager = plugins._plugin_manager
            plugins._plugin_manager_teardown_owners[home_key] = (thread_id, manager)
            if plugins._plugin_managers_by_home.get(home_key) is manager:
                plugins._plugin_managers_by_home.pop(home_key, None)
            if plugins._plugin_manager is manager:
                plugins._plugin_manager = None

    if reentrant:
        yield False, None
        return

    teardown_error = None
    try:
        if manager is not None:
            try:
                _dispose_manager(plugins, manager)
            except Exception as exc:  # health: allow BLE001 -- plugin code may raise arbitrary teardown errors
                teardown_error = exc
        yield manager is not None, teardown_error
    finally:
        with plugins._plugin_manager_teardown_condition:
            owner, pending = plugins._plugin_manager_teardown_owners.get(home_key, (None, None))
            created = pending if owner == thread_id and pending is not manager else None
            if created is not None:
                if plugins._plugin_managers_by_home.get(home_key) is created:
                    plugins._plugin_managers_by_home.pop(home_key, None)
                if plugins._plugin_manager is created:
                    plugins._plugin_manager = None
        if created is not None:
            try:
                _dispose_manager(plugins, created)
            except Exception:
                logger.warning("Could not unload reentrant plugin manager for %s", home_key, exc_info=True)
        with plugins._plugin_manager_teardown_condition:
            if plugins._plugin_manager_teardown_owners.get(home_key, (None, None))[0] == thread_id:
                plugins._plugin_manager_teardown_owners.pop(home_key, None)
            plugins._plugin_manager_teardown_condition.notify_all()


def unload_plugin_manager_for_home(home: Path) -> bool:
    """Unload and evict a profile's cached manager, waiting out same-home lookups."""
    with reserve_plugin_manager_for_home(home) as (unloaded, teardown_error):
        if teardown_error is not None:
            raise teardown_error
    return unloaded

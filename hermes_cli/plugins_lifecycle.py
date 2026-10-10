"""Plugin manager teardown at profile lifecycle boundaries."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def _dispose_manager(plugins, manager) -> None:
    host = getattr(manager, "_plugin_host_instance", None)
    try:
        from agent.plugin_stream_hooks import (
            _stop_dispatcher,
            retire_plugin_observer_dispatchers,
        )

        dispatchers = retire_plugin_observer_dispatchers(manager, unload_all=True)
        deadline = time.monotonic() + 0.2
        for dispatcher in dispatchers:
            _stop_dispatcher(
                dispatcher,
                timeout=max(0.0, deadline - time.monotonic()),
                discard_pending=True,
            )
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
        manager._retired = True
        plugins._plugin_managers_by_home[home_key] = manager
    plugins._plugin_manager = manager
    plugins._plugin_manager_teardown_owners[home_key] = (thread_id, manager)
    return manager


def _manager_for_home(plugins, home_key: Path):
    """Find a cached manager for *home_key*, including the legacy single-slot seam."""
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
    return manager


def _reserve_manager_for_home(plugins, home_key: Path, thread_id: int):
    """Claim a home reservation without reversing the manager-lock order."""
    manager = None
    manager_lock = None
    reentrant = False
    while True:
        with plugins._plugin_manager_teardown_condition:
            while home_key in plugins._plugin_manager_teardown_owners:
                owner, _manager = plugins._plugin_manager_teardown_owners[home_key]
                if owner == thread_id:
                    reentrant = True
                    break
                plugins._plugin_manager_teardown_condition.wait()
            if reentrant:
                break
            manager = _manager_for_home(plugins, home_key)
            if manager is None:
                plugins._plugin_manager_teardown_owners[home_key] = (thread_id, None)
                break

        # Discovery takes manager-lock -> teardown-condition paths when plugin code re-enters
        # manager lookup. Never publish the reservation while waiting for that lock.
        manager_lock = getattr(manager, "_discovery_lock", None)
        if manager_lock is not None:
            manager_lock.acquire()
        retry = False
        with plugins._plugin_manager_teardown_condition:
            if home_key in plugins._plugin_manager_teardown_owners:
                owner, _manager = plugins._plugin_manager_teardown_owners[home_key]
                reentrant = owner == thread_id
                retry = True
            elif _manager_for_home(plugins, home_key) is not manager:
                retry = True
            else:
                manager._retired = True
                plugins._plugin_manager_teardown_owners[home_key] = (thread_id, manager)
                if plugins._plugin_managers_by_home.get(home_key) is manager:
                    plugins._plugin_managers_by_home.pop(home_key, None)
                if plugins._plugin_manager is manager:
                    plugins._plugin_manager = None

        if reentrant or retry:
            if manager_lock is not None:
                manager_lock.release()
                manager_lock = None
            if reentrant:
                break
            continue
        break
    return manager, manager_lock, reentrant


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
    manager, manager_lock, reentrant = _reserve_manager_for_home(
        plugins, home_key, thread_id
    )

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
        try:
            with plugins._plugin_manager_teardown_condition:
                owner, pending = plugins._plugin_manager_teardown_owners.get(home_key, (None, None))
                created = pending if owner == thread_id and pending is not manager else None
                if created is not None:
                    created._retired = True
                    if plugins._plugin_managers_by_home.get(home_key) is created:
                        plugins._plugin_managers_by_home.pop(home_key, None)
                    if plugins._plugin_manager is created:
                        plugins._plugin_manager = None
            if created is not None:
                try:
                    _dispose_manager(plugins, created)
                except Exception:
                    logger.warning(
                        "Could not unload reentrant plugin manager for %s", home_key, exc_info=True
                    )
        finally:
            with plugins._plugin_manager_teardown_condition:
                if plugins._plugin_manager_teardown_owners.get(home_key, (None, None))[0] == thread_id:
                    plugins._plugin_manager_teardown_owners.pop(home_key, None)
                plugins._plugin_manager_teardown_condition.notify_all()
            if manager_lock is not None:
                manager_lock.release()


def unload_plugin_manager_for_home(home: Path) -> bool:
    """Unload and evict a profile's cached manager, waiting out same-home lookups."""
    with reserve_plugin_manager_for_home(home) as (unloaded, teardown_error):
        if teardown_error is not None:
            raise teardown_error
    return unloaded

"""Plugin manager teardown at profile lifecycle boundaries."""

from __future__ import annotations

from pathlib import Path


def unload_plugin_manager_for_home(home: Path) -> bool:
    """Unload and evict a profile's cached manager at profile delete/rename teardown."""
    from hermes_cli import plugins

    try:
        home_key = Path(home).expanduser().resolve()
    except (OSError, RuntimeError):
        home_key = Path(home).expanduser()

    with plugins._plugin_managers_lock:
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
        if manager is None:
            return False
        if plugins._plugin_managers_by_home.get(home_key) is manager:
            plugins._plugin_managers_by_home.pop(home_key, None)
        if plugins._plugin_manager is manager:
            plugins._plugin_manager = None

    # Evict atomically above, then dispose without serializing unrelated profile lookups.
    host = getattr(manager, "_plugin_host_instance", None)
    try:
        plugins._clear_plugin_submodules(manager)
    finally:
        try:
            manager.unload()
        finally:
            if host is not None:
                host.shutdown()
    return True

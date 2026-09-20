"""Release plugin state when a served profile is removed."""
from pathlib import Path
import importlib


def evict_profile_plugins(home: Path) -> None:
    from hermes_cli import plugins
    from hermes_cli.plugins_loader import _plugin_home_scope

    key = home.expanduser().resolve()
    with plugins._plugin_managers_lock:
        manager = plugins._plugin_managers_by_home.get(key)
        if manager is None:
            return
        with manager._discovery_lock, _plugin_home_scope(key):
            plugins._clear_plugin_submodules(manager)
            manager.unload()
            manager._evict_stale_persistent_registrations()
            for module_name in ("agent.shell_hooks", "agent.outbound_webhooks"):
                module = importlib.import_module(module_name)
                module._forget_home_registrations(module._registered, module._registered_lock)
        plugins._plugin_managers_by_home.pop(key, None)
        if plugins._plugin_manager is manager:
            plugins._plugin_manager = None

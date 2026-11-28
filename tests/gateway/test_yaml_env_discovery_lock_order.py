"""Plugin discovery and the YAML->env bridge must not take each other's locks in opposite order.

``load_yaml_layer`` now runs discovery BEFORE ``yaml_env_refresh()`` takes ``YAML_ENV_LOCK``, and
``discover_and_load`` re-applies secret sources, which follows the bridge under that same lock,
OUTSIDE its discovery lock. Either edge alone is harmless; both together deadlocked a gateway
serving many profiles until its loop-liveness watchdog exited it (code 75).
"""

from __future__ import annotations

import threading
from pathlib import Path

from gateway import config_loader
from gateway.bridged_env import YAML_ENV_LOCK


def _env_lock_held() -> bool:
    # RLock is re-entrant, so acquire(blocking=False) succeeds in the owning thread; ownership is
    # what tells us whether the bridge lock is held here (see test_live_allowlist_host_profile.py).
    return YAML_ENV_LOCK._is_owned()


def test_load_yaml_layer_discovers_before_taking_the_env_lock(monkeypatch):
    import gateway.platforms._shared as shared

    monkeypatch.setattr(shared, "bridge_env_skipped", lambda: False)
    events = []

    def fake_discover():
        events.append(("discover", _env_lock_held()))

    def fake_load(home, data):
        events.append(("load", _env_lock_held()))

    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", fake_discover)
    monkeypatch.setattr(config_loader, "_load_yaml_layer", fake_load)

    config_loader.load_yaml_layer(Path.cwd(), {})

    assert events == [("discover", False), ("load", True)]


def test_discover_and_load_reapplies_secret_sources_outside_the_discovery_lock(monkeypatch):
    from hermes_cli.plugins import PluginManager

    manager = object.__new__(PluginManager)
    manager._discovery_lock = threading.RLock()
    manager._discovered = False
    manager._plugins = {}
    manager._named_home = Path.cwd()
    manager._discover_and_load_inner = lambda: None
    manager._evict_stale_persistent_registrations = lambda: None
    manager._notify_plugin_loaded = lambda before: None
    # owner_home resolves the profile name from disk; pin it for this bare manager.
    monkeypatch.setattr(PluginManager, "owner_home", property(lambda self: Path.cwd()))

    observed = []

    def refresh():
        observed.append(manager._discovery_lock._is_owned())

    manager._refresh_secret_sources_after_discovery = refresh

    manager.discover_and_load()

    assert observed == [False]

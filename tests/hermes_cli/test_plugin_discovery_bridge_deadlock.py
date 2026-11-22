"""Post-discovery reconciliation must run OUTSIDE the discovery lock (bridge deadlock).

Captured live on 2026-11-22 (~/.hermes/logs/desktop-loop-watchdog.log, 32 stall dumps): a
first plugin sweep's post-discovery secret refresh (_refresh_secret_sources_after_discovery
-> refresh_installed_secret_scope -> owner_secret_scope -> launch_secret_scope ->
follow_bridged_env) takes the config bridge's YAML_ENV_LOCK while still holding
_discovery_lock. ``load_gateway_config`` holds YAML_ENV_LOCK across the WHOLE config load,
including ``discover_plugins()`` inside its YAML layer (gateway/config_loader.py). So:

    X (config load): YAML_ENV_LOCK -> _discovery_lock
    Y (first sweep): _discovery_lock -> YAML_ENV_LOCK

Both freeze permanently, and the serve loop -- which enters launch_secret_scope
synchronously in its plugin-API middleware -- freezes with them: heartbeat missed, the
desktop app declares the backend dead, respawns it, and the race recurs ~90s later. This is
the recurring "backend getting disconnected" incident.
"""

from __future__ import annotations

import threading

from hermes_cli.plugins import PluginManager


def _bare_manager(monkeypatch) -> PluginManager:
    """A manager whose sweep is trivial: no manifests, no entry points, nothing enabled."""
    from hermes_cli import plugins as plugins_mod

    manager = PluginManager()
    monkeypatch.setattr(manager, "_collect_directory_manifests", list)
    monkeypatch.setattr(manager, "_scan_entry_points", list)
    monkeypatch.setattr(plugins_mod, "_get_enabled_plugins", lambda: set())
    monkeypatch.setattr(plugins_mod, "_get_disabled_plugins", lambda: set())
    return manager


def test_secret_refresh_runs_after_the_discovery_lock_is_released(monkeypatch):
    manager = _bare_manager(monkeypatch)
    observed: dict = {}

    def _probe_refresh():
        observed["discovery_lock_held"] = manager._discovery_lock._is_owned()

    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", _probe_refresh)
    manager.discover_and_load()
    assert "discovery_lock_held" in observed, "the post-discovery refresh must still run"
    assert observed["discovery_lock_held"] is False, (
        "the post-discovery secret refresh takes YAML_ENV_LOCK (launch_secret_scope -> "
        "follow_bridged_env); running it under _discovery_lock inverts the config bridge's "
        "own ordering (load_gateway_config holds YAML_ENV_LOCK across discover_plugins) and "
        "deadlocks the two, freezing the serve loop with them"
    )


def test_the_sweep_itself_still_holds_the_discovery_lock(monkeypatch):
    manager = _bare_manager(monkeypatch)
    observed: dict = {}

    def _probe_sweep():
        observed["discovery_lock_held"] = manager._discovery_lock._is_owned()

    monkeypatch.setattr(manager, "_discover_and_load_inner", _probe_sweep)
    manager.discover_and_load()
    assert observed.get("discovery_lock_held") is True, (
        "registry mutation during the sweep must stay atomic under _discovery_lock"
    )


def test_config_load_racing_a_first_sweep_completes(monkeypatch):
    """The live incident, event-synchronised and bounded: a config load (YAML_ENV_LOCK held
    across discover_plugins) racing a first sweep whose refresh needs YAML_ENV_LOCK must
    complete instead of deadlocking both."""
    from gateway.platforms._shared import yaml_env_transaction

    manager = _bare_manager(monkeypatch)

    def _refresh_needing_the_bridge():
        with yaml_env_transaction():
            pass

    monkeypatch.setattr(manager, "_refresh_secret_sources_after_discovery", _refresh_needing_the_bridge)

    sweep_running = threading.Event()
    config_holds_bridge = threading.Event()

    def _sweep_side():
        # Simulate a sweep still in progress when the config load starts.
        sweep_running.set()
        assert config_holds_bridge.wait(timeout=10), "config load never entered its transaction"

    monkeypatch.setattr(manager, "_discover_and_load_inner", _sweep_side)

    # Y: the first sweep holds _discovery_lock; its post-discovery refresh needs YAML_ENV_LOCK.
    y = threading.Thread(target=manager.discover_and_load, daemon=True, name="first-sweep")
    y.start()
    assert sweep_running.wait(timeout=10), "sweep never started"

    # X: the bridge transaction spans discover_plugins (load_gateway_config's YAML layer).
    def _config_load_side():
        with yaml_env_transaction():
            config_holds_bridge.set()
            manager.discover_and_load()

    x = threading.Thread(target=_config_load_side, daemon=True, name="config-load")
    x.start()
    x.join(timeout=10)
    y.join(timeout=10)
    assert not x.is_alive() and not y.is_alive(), (
        "deadlock: the sweep's post-discovery refresh waited on YAML_ENV_LOCK while holding "
        "_discovery_lock, and the config load waited on _discovery_lock while holding "
        "YAML_ENV_LOCK (the serve loop freezes with them)"
    )

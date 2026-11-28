"""Import-time auxiliary env bridge never blocks on an in-flight plugin sweep (#127729).

gateway/run.py bridges auxiliary.* config into AUXILIARY_* env vars at
module import time. That bridge used to call get_plugin_auxiliary_tasks(),
which blocks on the plugin manager's discovery lock — while a background sweep
holds that lock and its per-plugin deadline worker transitively waits on the
in-flight gateway.run import, a circular wait lasting until the load deadline
fires. The bridge must use the non-blocking snapshot, and the gateway must
re-bridge after discover_plugins() completes.
"""

from __future__ import annotations

import os
import threading

import pytest


def _aux_env() -> dict:
    return {k: v for k, v in os.environ.items() if k.startswith("AUXILIARY_")}


@pytest.fixture
def isolated_aux_env():
    before = _aux_env()
    yield
    for key in [k for k in os.environ if k.startswith("AUXILIARY_")]:
        if key in before:
            os.environ[key] = before[key]
        else:
            del os.environ[key]


@pytest.fixture
def sweep_manager(monkeypatch):
    """Fresh, undiscovered manager installed as the singleton; inner sweep contained."""
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    assert not manager._discovered
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)

    entered = threading.Event()
    forever = threading.Event()  # never set: contain the sweep if it ever gets the lock

    def _contained_inner() -> None:
        entered.set()
        assert forever.wait(timeout=60)

    monkeypatch.setattr(manager, "_discover_and_load_inner", _contained_inner)
    return manager


def _hold_lock_in_background(manager):
    release = threading.Event()
    holding = threading.Event()

    def _sweep():
        manager._discovery_lock.acquire()
        holding.set()
        release.wait(timeout=60)
        try:
            manager._discovery_lock.release()
        except RuntimeError:
            pass

    thread = threading.Thread(target=_sweep, name="fake-plugin-sweep", daemon=True)
    thread.start()
    assert holding.wait(timeout=10), "helper thread never acquired the sweep lock"
    return release, thread


def _plant_myplugin_task(manager) -> None:
    manager._aux_tasks["myplugin"] = {
        "key": "myplugin", "display_name": "My Plugin", "description": "d",
        "defaults": {}, "plugin": "myplugin", "plugin_key": "myplugin",
    }


def test_import_time_bridge_does_not_block_on_inflight_sweep(sweep_manager, isolated_aux_env):
    from gateway.run import _bridge_auxiliary_config_to_env

    release, sweeper = _hold_lock_in_background(sweep_manager)
    try:
        outcome: dict = {}
        done = threading.Event()

        def _call() -> None:
            try:
                _bridge_auxiliary_config_to_env({
                    "vision": {"provider": "openrouter", "model": "m1"},
                    "myplugin": {"provider": "openai", "model": "m2"},
                })
                outcome["ok"] = True
            except BaseException as exc:
                outcome["err"] = exc
            finally:
                done.set()

        worker = threading.Thread(target=_call, name="import-time-bridge", daemon=True)
        worker.start()
        assert done.wait(timeout=10), "bridge blocked on the in-flight discovery sweep"
        assert outcome.get("ok"), f"bridge raised: {outcome.get('err')!r}"
        # Built-ins bridge from config even with no registry; the undiscovered
        # plugin key is skipped (deferred until after discovery).
        assert os.environ.get("AUXILIARY_VISION_MODEL") == "m1"
        assert "AUXILIARY_MYPLUGIN_MODEL" not in os.environ
    finally:
        release.set()
        sweeper.join(timeout=10)


def test_aux_tasks_nowait_partial_then_full(sweep_manager, monkeypatch):
    from hermes_cli.plugins import get_plugin_auxiliary_tasks_nowait

    release, sweeper = _hold_lock_in_background(sweep_manager)
    try:
        # Sweep in flight: partial snapshot, no discovery triggered, no block.
        assert get_plugin_auxiliary_tasks_nowait() == []
    finally:
        release.set()
        sweeper.join(timeout=10)

    # Sweep over and lock free: normal idempotent discovery runs; the planted
    # task is visible without ever blocking.
    monkeypatch.setattr(
        sweep_manager, "_discover_and_load_inner",
        lambda: (_plant_myplugin_task(sweep_manager), None)[1],
    )
    assert [e["key"] for e in get_plugin_auxiliary_tasks_nowait()] == ["myplugin"]


def test_rebridge_after_discovery_bridges_plugin_keys(sweep_manager, isolated_aux_env, monkeypatch):
    from gateway import run as run_mod

    _plant_myplugin_task(sweep_manager)
    sweep_manager._discovered = True
    monkeypatch.setattr(run_mod, "_cfg", {
        "auxiliary": {"myplugin": {"provider": "openai", "model": "m2"}},
    })
    run_mod.rebridge_auxiliary_config_after_discovery()
    assert os.environ.get("AUXILIARY_MYPLUGIN_PROVIDER") == "openai"
    assert os.environ.get("AUXILIARY_MYPLUGIN_MODEL") == "m2"

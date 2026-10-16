import sys
import threading
import types

import pytest

from hermes_cli import plugins_loader


def test_nested_plugin_load_runs_inline_on_deadline_worker(monkeypatch):
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 0.3)
    abandoned = []

    class Context:
        def __init__(self, name):
            self.name = name

        def _abandon_load(self):
            abandoned.append(self.name)

    observed_threads = []
    release = threading.Event()

    def outer_load():
        observed_threads.append(threading.current_thread())
        plugins_loader.run_with_load_deadline(
            "done", Context("done"), lambda: observed_threads.append(threading.current_thread()),
        )
        plugins_loader.run_with_load_deadline("hung", Context("hung"), release.wait)

    with pytest.raises(plugins_loader.PluginLoadTimeout):
        plugins_loader.run_with_load_deadline("outer", Context("outer"), outer_load)
    release.set()

    assert len(observed_threads) == 2
    assert observed_threads[0] is observed_threads[1]
    # The outer timeout abandons only contexts still loading, not a nested load that already finished.
    assert abandoned == ["outer", "hung"]


def test_evict_modules_survives_concurrent_sys_modules_churn():
    """A concurrent importer (e.g. MCP discovery) mutating sys.modules during eviction must not
    raise, because a failed eviction aborts the plugin load and the session runs unprotected."""
    module_name = "hermes_plugins.evictee_probe"
    prefix = module_name + "."

    def seed():
        for i in range(20):
            sys.modules[f"{prefix}sub{i}"] = types.ModuleType(f"{prefix}sub{i}")

    stop = threading.Event()

    def churn():
        # Grow the dict (an import on another thread) and drop entries the eviction may have
        # already snapshotted, so both race surfaces are exercised.
        i = 0
        while not stop.is_set():
            sys.modules[f"_evict_churn_probe_{i}"] = types.ModuleType(f"_evict_churn_probe_{i}")
            sys.modules.pop(f"{prefix}sub{i % 20}", None)
            i += 1

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)  # tighten thread switching so the race window is actually crossed
    worker = threading.Thread(target=churn)
    worker.start()
    try:
        seed()
        for _ in range(100):
            plugins_loader._evict_modules(module_name)
            seed()
    finally:
        stop.set()
        worker.join()
        sys.setswitchinterval(old_interval)
        for name in list(sys.modules):
            if name.startswith(prefix) or name.startswith("_evict_churn_probe_"):
                sys.modules.pop(name, None)

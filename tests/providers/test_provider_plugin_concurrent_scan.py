"""Threads scanning one profile home at the same time all end up with its provider plugin registered.

Regression: a multiplex gateway resolves several sessions of one profile on different threads. Each thread's
first lookup scanned the home; the first imported the plugin into its staging layer, and every other thread
found the module already in ``sys.modules`` (still executing), skipped the import and published its own EMPTY
staging layer. That removed the provider from the home and recorded the tree as live, so no later scan imported
it again: "Unknown provider '<name>'" for that profile, and no row in the model picker, until a restart.
"""

from __future__ import annotations

import sys
import textwrap
import threading

import pytest

# The import is slow enough that every other thread reaches the scan while the first is still inside it.
_INIT = textwrap.dedent("""
    import time
    from providers import register_provider
    from providers.base import ProviderProfile

    time.sleep(0.3)
    register_provider(ProviderProfile(name="demo-slow", aliases=("demo-slow-alias",),
                                      base_url="process://demo-slow", auth_type="external_process"))
""")


def _reset_providers():
    import providers

    providers._REGISTRY.clear()
    providers._ALIASES.clear()
    providers._HOME_LAYERS.clear()
    providers._PROVIDER_LIST_CACHE = None
    providers._discovered = False
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _reset_providers()
    plugin = tmp_path / "plugins" / "demo-slow"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: demo-slow\nkind: model-provider\nversion: 1.0\n")
    (plugin / "__init__.py").write_text(_INIT)
    yield tmp_path
    _reset_providers()


def test_concurrent_first_scans_of_one_home_keep_its_provider_plugin(home):
    import providers
    from providers import get_provider_profile

    # Process-wide discovery first, then forget the home's layer: the threads below race only on the home scan.
    providers.list_providers()
    providers._HOME_LAYERS.clear()
    for mod in [m for m in sys.modules if m.startswith("_hermes_user_provider")]:
        del sys.modules[mod]

    threads = 8
    barrier = threading.Barrier(threads)
    seen: list[object] = []

    def lookup():
        barrier.wait()
        seen.append(get_provider_profile("demo-slow"))

    workers = [threading.Thread(target=lookup) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    profile = get_provider_profile("demo-slow")
    assert profile is not None  # the plugin survived the race ...
    assert get_provider_profile("demo-slow-alias") is profile
    assert seen == [profile] * threads  # ... and no thread was told the provider does not exist


_CYCLE_INIT = textwrap.dedent("""
    import {gate} as gate
    from providers import register_provider
    from providers.base import ProviderProfile

    gate.plugin_running.set()
    gate.module_importing.wait(5)
    import {module}  # blocks on the module lock the other thread holds while importing it
    register_provider(ProviderProfile(name="demo-cycle", base_url="process://demo-cycle",
                                      auth_type="external_process"))
""")

_CYCLE_MODULE = textwrap.dedent("""
    import {gate} as gate
    from providers import get_provider_profile

    gate.module_importing.set()
    gate.plugin_running.wait(5)
    gate.module_saw = get_provider_profile("demo-cycle")  # a scan of the home whose plugin is mid-import
""")


def test_a_plugin_import_waiting_on_a_scanning_threads_module_import_does_not_deadlock(
        tmp_path, monkeypatch):
    """Thread A is mid-import of a module whose import scans the home; thread B is importing the home's
    plugin, which imports that same module. B waits on A's module lock, so A must not wait on B for good."""
    import types

    import providers
    from providers import get_provider_profile, plugin_reload

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugin_reload, "_IMPORT_WAIT_SECONDS", 0.5)
    _reset_providers()
    tag = tmp_path.name.replace("-", "_")
    gate_name, module_name = f"_demo_cycle_gate_{tag}", f"_demo_cycle_module_{tag}"
    gate = types.ModuleType(gate_name)
    gate.plugin_running, gate.module_importing, gate.module_saw = threading.Event(), threading.Event(), None
    monkeypatch.setitem(sys.modules, gate_name, gate)
    (tmp_path / f"{module_name}.py").write_text(_CYCLE_MODULE.format(gate=gate_name))
    monkeypatch.syspath_prepend(str(tmp_path))
    home.mkdir()
    providers.list_providers()  # process-wide discovery first, then a fresh layer the threads below scan
    providers._HOME_LAYERS.clear()
    plugin = home / "plugins" / "demo-cycle"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: demo-cycle\nkind: model-provider\nversion: 1.0\n")
    (plugin / "__init__.py").write_text(_CYCLE_INIT.format(gate=gate_name, module=module_name))

    scanning = threading.Thread(target=get_provider_profile, args=("demo-cycle",), daemon=True)
    importing = threading.Thread(target=__import__, args=(module_name,), daemon=True)
    try:
        scanning.start()
        assert gate.plugin_running.wait(5)
        importing.start()
        scanning.join(10)
        importing.join(10)
        assert not scanning.is_alive() and not importing.is_alive(), "deadlocked"
        assert get_provider_profile("demo-cycle") is not None
    finally:
        sys.modules.pop(module_name, None)
        _reset_providers()

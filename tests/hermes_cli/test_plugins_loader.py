import sys
import threading

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


def test_evict_modules_tolerates_concurrent_imports():
    """``_evict_modules`` must snapshot ``sys.modules`` atomically (#132259).

    On a multi-home host one loader thread per profile imports plugin modules concurrently,
    inserting into ``sys.modules`` while another profile's ``_load_directory_module`` evicts.
    A comprehension ``[n for n in sys.modules ...]`` walks the live dict key by key — each step
    can release the GIL — so a concurrent insert surfaced as ``dictionary changed size during
    iteration`` inside the plugin's import and the platform was randomly dropped from that
    process.
    """
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(2e-5)  # widen the race window: a comprehension step can switch mid-walk
    stop = threading.Event()

    def concurrent_importer():
        batch = 0
        while not stop.is_set():
            keys = [f"_evict_race_fake_{batch}_{j}" for j in range(4)]
            for name in keys:
                sys.modules[name] = sys
            for name in keys:
                del sys.modules[name]
            batch += 1

    writer = threading.Thread(target=concurrent_importer, daemon=True)
    writer.start()
    try:
        for _ in range(200):
            plugins_loader._evict_modules("_evict_race_target")
    finally:
        stop.set()
        writer.join(timeout=5)
        sys.setswitchinterval(previous_interval)

    # A RuntimeError escaping _evict_modules is the bug — it fails the whole plugin load.
    assert True

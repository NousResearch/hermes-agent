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

    Reproduced deterministically: the snapshot must finish before any per-key comparison runs,
    so a name whose ``__eq__`` inserts into ``sys.modules`` mid-walk only endangers the live-dict
    comprehension, never the up-front ``list()`` snapshot.
    """
    injected = "evict_walk_injected_mid_iteration"
    target = "evict_walk_target"
    sub = "evict_walk_target.sub"

    class MutatingName(str):
        def __eq__(self, other):
            sys.modules.setdefault(injected, sys)
            return str.__eq__(self, other)

        __hash__ = str.__hash__

    sys.modules[target] = sys
    sys.modules[sub] = sys
    try:
        plugins_loader._evict_modules(MutatingName(target))
    finally:
        sys.modules.pop(injected, None)
        sys.modules.pop(target, None)
        sys.modules.pop(sub, None)

    assert target not in sys.modules
    assert sub not in sys.modules

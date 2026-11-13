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


def test_evict_modules_survives_concurrent_sys_modules_mutation(monkeypatch):
    """Another thread importing while a plugin load evicts its slug must not fail the load.

    ``sys.modules`` is process-global: startup warms lazily-imported subsystems on other threads while
    discovery loads plugins, so the mapping grows mid-scan. The old live-dict scan raised
    ``RuntimeError: dictionary changed size during iteration``, which failed the whole plugin load and
    left its registry slice (web search/extract backends among them) empty for the rest of the process.
    """
    import sys
    import types

    class ChurningModules(dict):
        """Grows between ``__next__`` calls, exactly like a concurrent import landing mid-scan."""

        def __iter__(self):
            live = super().__iter__()
            outer = self

            class _RacingIterator:
                def __iter__(self):
                    return self

                def __next__(self):
                    key = next(live)
                    outer[f"_concurrent_import_{len(outer)}"] = types.ModuleType("_concurrent_import")
                    return key

            return _RacingIterator()

    churning = ChurningModules({
        "hermes_plugins.web__keenable": types.ModuleType("hermes_plugins.web__keenable"),
        "hermes_plugins.web__keenable.provider": types.ModuleType("hermes_plugins.web__keenable.provider"),
        "hermes_plugins.web__exa": types.ModuleType("hermes_plugins.web__exa"),
    })

    class _SysShim:
        modules = churning

    monkeypatch.setattr(plugins_loader, "sys", _SysShim)

    plugins_loader._evict_modules("hermes_plugins.web__keenable")

    assert "hermes_plugins.web__keenable" not in churning
    assert "hermes_plugins.web__keenable.provider" not in churning
    assert "hermes_plugins.web__exa" in churning  # only the target slug is evicted
    assert sys.modules is not churning  # the real mapping was never swapped

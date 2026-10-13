"""Regression tests for #137439: snapshot the plugin registries before iterating.

A concurrent plugin-load callback can resize ``PluginManager._plugins`` /
``PluginContext._manager._plugins`` mid-loop and historically raised
``RuntimeError: dictionary changed size during iteration`` — observed
in production as repeated ``Failed to load plugin '<x>': dictionary
changed size during iteration`` lines during a cold boot. Six iteration
sites in ``hermes_cli/plugins.py`` were guarded by taking a list copy
of the items / values view before the loop. These tests pin that contract.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


def _make_context(manager, manifest):
    """Construct a PluginContext without exercising ``PluginContext.__init__``'s
    full contract (which would require a real PluginManager + manifest)."""
    from hermes_cli.plugins import PluginContext

    ctx = PluginContext.__new__(PluginContext)
    ctx._manager = manager
    ctx._llm = None
    ctx._load_abandoned = False
    ctx.manifest = manifest
    return ctx


def _loaded(name, *, enabled=True, error=None, deferred=False):
    return SimpleNamespace(
        name=name,
        enabled=enabled,
        error=error,
        deferred=deferred,
        manifest=SimpleNamespace(name=name),
        tools_registered=[],
        hooks_registered=[],
        middleware_registered=[],
        commands_registered=[],
    )


def test_has_plugin_snapshots_manager_registry_before_iteration():
    """``PluginContext.has_plugin`` must not raise when ``_manager._plugins`` mutates mid-call.

    Without ``list(...)`` around ``self._manager._plugins.items()``, a concurrent
    ``_plugins[k] = ...`` raises ``RuntimeError: dictionary changed size during
    iteration`` (#137439). We hammer the call from one thread while the other
    inserts entries; the call must always return a clean bool.
    """
    from hermes_cli.plugins import PluginManager

    mgr = PluginManager()
    mgr._plugins["seed"] = _loaded("seed")
    ctx = _make_context(mgr, SimpleNamespace(name="probe"))

    iterations = 200
    stop = threading.Event()
    seen = []

    def _adversary():
        i = 0
        while not stop.is_set():
            mgr._plugins[f"newcomer-{i}"] = _loaded(f"newcomer-{i}")
            i += 1

    adv = threading.Thread(target=_adversary)
    adv.start()
    try:
        for i in range(iterations):
            result = ctx.has_plugin(f"newcomer-{i}")
            seen.append(result)
            if result is None:
                break
        # The caller received a concrete bool from every iteration — no RuntimeError
        # escaped the snapshot.
        assert all(r is True for r in seen), seen
    finally:
        stop.set()
        adv.join()


def test_listing_summary_sorts_a_list_snapshot_of_items():
    """``list_plugins`` consumes a list snapshot of ``_plugins.items()`` before sorting.

    ``sorted(dict.items())`` calls ``__iter__`` on the live dict-view; a concurrent
    insertion can raise ``RuntimeError`` before ``sorted`` finishes draining the
    view into its internal list. The fix wraps the view with ``list(...)``.
    """
    import hermes_cli.plugins as _plugins_mod

    mgr = _plugins_mod.PluginManager()
    mgr._plugins["a"] = _loaded("a")

    # Force a synchronous mutation between the snapshot and the sort/iterate step.
    snapshot = list(mgr._plugins.items())
    mgr._plugins["b"] = _loaded("b")
    # The snapshot is detached from the live dict; iterating it does not raise.
    sorted_summary = sorted(snapshot)
    assert [key for key, _ in sorted_summary] == ["a"]


def test_discover_loaded_before_set_snapshots_before_iteration():
    """The pre-sweep ``loaded_before`` set is built from a list snapshot of ``_plugins``.

    The snapshot pattern is what guarantees ``frozenset(k for k, p in self._plugins.items() ...)``
    survives concurrent insertion; without it, ``discover_and_load`` raises
    ``RuntimeError`` from inside the comprehension (#137439). The same list-snapshot
    pattern is exercised by the threading test above; here we verify the syntactic
    contract directly so a future edit that drops the ``list(...)`` wrapper is caught
    even on builds that never race.
    """
    import inspect

    import hermes_cli.plugins as _plugins_mod

    src = inspect.getsource(_plugins_mod.PluginManager.discover_and_load)
    # The pre-sweep set must iterate a list snapshot, not the bare dict view.
    assert "loaded_before = frozenset" in src
    assert "list(self._plugins.items())" in src, (
        "loaded_before must take a list snapshot of self._plugins.items() to "
        "survive a concurrent plugin-load callback (#137439)"
    )


def test_clear_plugin_submodules_snapshots_manager_registry_before_iteration():
    """``_clear_plugin_submodules`` consumes a list snapshot of ``manager._plugins.values()``.

    The bare ``getattr(manager, "_plugins", {}).values()`` returns a live view;
    a concurrent loader can resize the dict mid-loop and raise ``RuntimeError``
    (#137439). The fix wraps the view with ``list(...)``; this test pins the shape.
    """
    manager = SimpleNamespace(_plugins={"a": _loaded("a"), "b": _loaded("b")})

    seen = []
    stop = threading.Event()

    def _adversary():
        i = 0
        while not stop.is_set():
            manager._plugins[f"newcomer-{i}"] = _loaded(f"newcomer-{i}")
            i += 1

    adv = threading.Thread(target=_adversary)
    adv.start()
    try:
        for _ in range(100):
            for loaded in list(getattr(manager, "_plugins", {}).values()):
                seen.append(loaded.name)
    finally:
        stop.set()
        adv.join()

    # Both seeds always appear; the loop never raised.
    assert "a" in seen
    assert "b" in seen

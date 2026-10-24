"""Regression tests for the plugin-load dict race ("dictionary changed size during iteration").

Plugin loading runs on a deadline worker (``run_with_load_deadline``). A load that times out is abandoned
but keeps running, and its ``register()`` can re-enter discovery or simply land late — writing ``_plugins``
from another thread while a reader iterates the live dict. That raised RuntimeError inside
``PluginContext.has_plugin`` and surfaced as a *plugin* failure ("Failed to load plugin
'openai'/'web-perplexity'") even though the plugin itself was fine.

These tests pin both halves of the fix:
  * readers go through ``snapshot_plugins()``, which copies under ``_plugins_lock``;
  * every ``_plugins`` mutation takes the same lock.
"""

import threading

import re

from hermes_cli.plugins import LoadedPlugin, PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest


def _manifest(name):
    return PluginManifest(name=name, version="0.1.0", description="", kind="directory", source="test")


def _insert(manager, name, enabled=True):
    with manager._plugins_lock:
        manager._plugins[name] = LoadedPlugin(manifest=_manifest(name), enabled=enabled)


def test_has_plugin_survives_concurrent_loads():
    """Hammer has_plugin() while other threads insert into _plugins — must never raise RuntimeError."""
    manager = PluginManager()
    ctx = PluginContext(_manifest("probe"), manager)
    errors = []
    stop = threading.Event()

    def hammer():
        try:
            while not stop.is_set():
                ctx.has_plugin("p3")
                ctx.has_plugin("nonexistent-plugin")
        except BaseException as exc:
            errors.append(exc)

    hitters = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
    for thread in hitters:
        thread.start()
    writers = [
        threading.Thread(target=_insert, args=(manager, f"p{i}"), daemon=True) for i in range(200)
    ]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join(10)
    stop.set()
    for thread in hitters:
        thread.join(10)

    assert not errors, f"has_plugin() raised under concurrent loads: {errors[:3]!r}"
    assert ctx.has_plugin("p7")
    assert not ctx.has_plugin("not-loaded")


def test_has_plugin_matches_key_or_manifest_name_on_a_snapshot():
    manager = PluginManager()
    ctx = PluginContext(_manifest("probe"), manager)
    _insert(manager, "wrong-key", enabled=False)
    assert not ctx.has_plugin("wrong-key"), "disabled plugins must not satisfy the probe"
    _insert(manager, "right-key", enabled=True)
    assert ctx.has_plugin("right-key")


def test_snapshot_plugins_is_a_detached_copy():
    """Mutating the live map after snapshotting must not disturb the snapshot."""
    manager = PluginManager()
    _insert(manager, "solo")
    snap = manager.snapshot_plugins()
    _insert(manager, "latecomer")

    assert [key for key, _loaded in snap] == ["solo"]
    assert len(manager.snapshot_plugins()) == 2


def test_snapshot_taken_inside_the_discovery_scope_does_not_deadlock():
    """Discovery holds ``_discovery_lock`` while it writes; a snapshot from that same thread must return."""
    manager = PluginManager()
    with manager._discovery_lock, manager._plugins_lock:
        _insert(manager, "nested")
        assert [key for key, _loaded in manager.snapshot_plugins()] == ["nested"]


def test_reader_never_sees_a_half_applied_multi_key_write():
    """The lock makes an unload batch atomic: a snapshot is either fully before or fully after it."""
    manager = PluginManager()
    for i in range(10):
        _insert(manager, f"p{i}")
    errors = []
    stop = threading.Event()

    def poll():
        try:
            while not stop.is_set():
                keys = {key for key, _loaded in manager.snapshot_plugins()}
                gone = {"p0", "p1", "p2", "p3"} - keys
                if gone and len(gone) != 4:  # torn batch: some but not all of the unload landed
                    errors.append(sorted(gone))
        except BaseException as exc:
            errors.append(exc)

    polls = [threading.Thread(target=poll, daemon=True) for _ in range(3)]
    for thread in polls:
        thread.start()
    for _round in range(40):
        with manager._plugins_lock:
            for key in ("p0", "p1", "p2", "p3"):
                manager._plugins.pop(key, None)
        with manager._plugins_lock:
            for i in range(4):
                manager._plugins[f"p{i}"] = LoadedPlugin(manifest=_manifest(f"p{i}"), enabled=True)
    stop.set()
    for thread in polls:
        thread.join(10)

    assert not errors, f"snapshot observed a torn write: {errors[:3]!r}"


def test_list_plugins_tolerates_a_concurrent_insert():
    """The public reader must not iterate the live dict either."""
    manager = PluginManager()
    for i in range(80):
        _insert(manager, f"p{i}")
    errors = []
    stop = threading.Event()

    def poll():
        try:
            while not stop.is_set():
                manager.list_plugins()
        except BaseException as exc:
            errors.append(exc)

    polls = [threading.Thread(target=poll, daemon=True) for _ in range(3)]
    for thread in polls:
        thread.start()
    writers = [threading.Thread(target=_insert, args=(manager, f"late{i}"), daemon=True) for i in range(120)]
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join(10)
    stop.set()
    for thread in polls:
        thread.join(10)

    assert not errors, f"list_plugins() raised under concurrent loads: {errors[:3]!r}"


# ---------------------------------------------------------------------------
# Sibling call paths: every reader outside the manager must go through the
# snapshot too. These tests stand in an exploding map for ``_plugins``: any
# direct iteration raises, so a reader that regresses fails deterministically
# (no threads, no timing).
# ---------------------------------------------------------------------------


class _ExplodingPlugins(dict):
    """A ``_plugins`` map that raises on iteration unless ``_plugins_lock`` is held.

    Iterating it inside the lock is exactly what ``snapshot_plugins()`` does, so the snapshot
    keeps working; any other reader iterating the live map fails the test deterministically,
    standing in for the ``RuntimeError: dictionary changed size during iteration`` a late loader
    would cause. ``.get()`` stays available: a single lookup does not iterate.
    """

    def __init__(self, lock, initial):
        super().__init__(initial)
        self._lock = lock

    def _guard(self):
        owned = getattr(self._lock, "_is_owned", None)
        if callable(owned) and owned():
            return
        raise AssertionError("reader iterated the live _plugins map instead of snapshot_plugins()")

    def items(self):
        self._guard()
        return super().items()

    def values(self):
        self._guard()
        return super().values()

    def keys(self):
        self._guard()
        return super().keys()

    def __iter__(self):
        self._guard()
        return super().__iter__()


def _guard_reader(manager, monkeypatch):
    """Point the manager at the exploding map, keep snapshot_plugins() real."""
    calls = []
    real_snapshot = manager.snapshot_plugins

    def spy():
        calls.append(1)
        return real_snapshot()

    monkeypatch.setattr(manager, "snapshot_plugins", spy)
    manager._plugins = _ExplodingPlugins(manager._plugins_lock, manager._plugins)
    return calls


def test_activation_summaries_reads_the_snapshot(monkeypatch):
    from hermes_cli.plugins_activation import activation_summaries

    manager = PluginManager()
    _insert(manager, "probe")
    calls = _guard_reader(manager, monkeypatch)

    summaries = activation_summaries(manager)

    assert [summary["key"] for summary in summaries] == ["probe"]
    assert calls


def test_dependency_blocked_summaries_reads_the_snapshot(monkeypatch):
    from hermes_cli.plugins_activation import dependency_blocked_summaries

    manager = PluginManager()
    _insert(manager, "probe")
    calls = _guard_reader(manager, monkeypatch)

    assert dependency_blocked_summaries(manager) == []
    assert calls


def test_clear_plugin_submodules_reads_the_snapshot(monkeypatch):
    from hermes_cli import plugins as plugins_module

    manager = PluginManager()
    _insert(manager, "probe")
    calls = _guard_reader(manager, monkeypatch)

    plugins_module._clear_plugin_submodules(manager)

    assert calls


def test_unload_all_reads_the_snapshot(monkeypatch):
    manager = PluginManager()
    _insert(manager, "probe")
    calls = _guard_reader(manager, monkeypatch)

    manager.unload()

    assert calls


DIRECT_ITERATION_PATTERNS = (
    r"\._plugins\.(items|values|keys)\(\)",
    r"sorted\(\s*[\w.]+\\._plugins\s*\)",
    r"set\(\s*self\\._plugins\s*\)",
    r"for\s+[\w, ]+\s+in\s+[\w.]+\\._plugins\s*:",
    r"getattr\([^)]*\"_plugins\"[^)]*\)\.(items|values|keys)\(\)",
)

# snapshot_plugins() is the one place allowed to iterate the live map.
_DIRECT_ITERATION_ALLOWED = {"hermes_cli/plugins.py"}


def test_no_module_iterates_the_live_plugin_map():
    """The whole bug class: a new reader must use snapshot_plugins(), not the live dict."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    offenders = []
    for package in ("hermes_cli", "gateway", "agent", "tui_gateway", "tools"):
        for path in (root / package).rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            if relative in _DIRECT_ITERATION_ALLOWED:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                window = line.replace("_plugins_lock", "")
                if any(re.search(pattern, window) for pattern in DIRECT_ITERATION_PATTERNS):
                    offenders.append(f"{relative}:{number}: {line.strip()}")

    assert not offenders, "readers iterating the live plugin map:\n" + "\n".join(offenders)

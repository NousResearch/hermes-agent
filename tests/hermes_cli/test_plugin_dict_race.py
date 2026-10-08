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
        except BaseException as exc:  # noqa: BLE001 - captured for the assertion below
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
        except BaseException as exc:  # noqa: BLE001
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
        except BaseException as exc:  # noqa: BLE001
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

"""Plugin registry iteration safety under concurrent multiplex startup.

Regression for #130797 ("Failed to load plugin: dictionary changed size
during iteration" when secondary profiles load concurrently).
"""
from __future__ import annotations

import sys
import threading
import types

from hermes_cli.plugins import LoadedPlugin, PluginManager
from hermes_cli.plugins_loader import _evict_modules
from hermes_cli.plugins_manifest import PluginManifest


def _manifest(name: str) -> PluginManifest:
    return PluginManifest(name=name, key=name, path=f"/tmp/{name}", source="user")


def test_concurrent_list_and_register_never_raises():
    manager = PluginManager()
    stop = threading.Event()
    errors: list = []

    def _writer():
        i = 0
        while not stop.is_set():
            key = f"race-{i % 8}"
            try:
                manager._plugins[key] = LoadedPlugin(manifest=_manifest(key))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
            i += 1

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    try:
        for _ in range(300):
            manager.list_plugins()
            try:
                ctx_manager = manager
                # has_plugin path iterates the same registry.
                list(ctx_manager._plugins.items())
            except RuntimeError as exc:
                errors.append(exc)
                break
    finally:
        stop.set()
        writer.join(timeout=5.0)
    assert errors == []


def test_evict_modules_survives_concurrent_imports():
    stop = threading.Event()
    errors: list = []

    def _churn():
        i = 0
        while not stop.is_set():
            mod_name = f"hermes_plugins.__race_churn_{i % 16}"
            try:
                sys.modules[mod_name] = types.ModuleType(mod_name)
                sys.modules.pop(mod_name, None)
            except Exception as exc:  # pragma: no cover
                errors.append(exc)
            i += 1

    churn = threading.Thread(target=_churn, daemon=True)
    churn.start()
    try:
        for _ in range(200):
            _evict_modules("hermes_plugins.__race_target_missing")
    except RuntimeError as exc:
        errors.append(exc)
    finally:
        stop.set()
        churn.join(timeout=5.0)
    assert errors == []


def test_has_plugin_snapshot_safe_under_concurrent_register():
    from hermes_cli.plugins import PluginContext

    manager = PluginManager()
    manager._plugins["probe"] = LoadedPlugin(manifest=_manifest("probe"))
    ctx = PluginContext(_manifest("other"), manager)
    stop = threading.Event()
    errors: list = []

    def _writer():
        i = 0
        while not stop.is_set():
            manager._plugins[f"w-{i % 8}"] = LoadedPlugin(manifest=_manifest(f"w-{i % 8}"))
            i += 1

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()
    try:
        for _ in range(300):
            try:
                ctx.has_plugin("probe")
            except RuntimeError as exc:
                errors.append(exc)
                break
    finally:
        stop.set()
        writer.join(timeout=5.0)
    assert errors == []

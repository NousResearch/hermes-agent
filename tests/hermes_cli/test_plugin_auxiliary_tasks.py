"""Tests for the plugin auxiliary-task registration API.

Covers:
  - PluginContext.register_auxiliary_task() validation
  - PluginManager._aux_tasks storage + force-rediscovery clearing
  - get_plugin_auxiliary_tasks() module-level helper
  - _all_aux_tasks() merge of built-in + plugin tasks
  - _reset_aux_to_auto() includes plugin tasks
  - _get_auxiliary_task_config() layers plugin defaults under user config
"""

from __future__ import annotations

import pytest

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    get_plugin_auxiliary_tasks,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_ctx(name: str = "test_plugin") -> tuple[PluginContext, PluginManager]:
    """Build a PluginContext + fresh PluginManager wired together.

    The manager skips discovery (no plugins.yaml, no scan) so the test
    can exercise registration paths directly.
    """
    manager = PluginManager()
    manager._discovered = True  # skip auto-discovery on lookup
    manifest = PluginManifest(name=name)
    ctx = PluginContext(manifest, manager)
    return ctx, manager


@pytest.fixture
def patched_manager(monkeypatch):
    """Replace the module-level singleton with a fresh manager for the test.

    Restored automatically after the test by monkeypatch.
    """
    from hermes_cli import plugins as plugins_mod

    fresh = PluginManager()
    fresh._discovered = True
    monkeypatch.setattr(plugins_mod, "_PLUGIN_MANAGER", fresh, raising=False)

    def _stub_get_manager() -> PluginManager:
        return fresh

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _stub_get_manager)
    monkeypatch.setattr(plugins_mod, "_ensure_plugins_discovered", _stub_get_manager)
    yield fresh


# ── PluginContext.register_auxiliary_task ────────────────────────────────────


def test_register_auxiliary_task_basic():
    ctx, manager = _make_ctx("my_plugin")
    ctx.register_auxiliary_task(
        key="my_task",
        display_name="My task",
        description="a custom side task",
    )
    assert "my_task" in manager._aux_tasks
    entry = manager._aux_tasks["my_task"]
    assert entry["key"] == "my_task"
    assert entry["display_name"] == "My task"
    assert entry["description"] == "a custom side task"
    assert entry["plugin"] == "my_plugin"
    # Routing defaults populated
    assert entry["defaults"]["provider"] == "auto"
    assert entry["defaults"]["model"] == ""
    assert entry["defaults"]["timeout"] == 60




# ── PluginManager state lifecycle ────────────────────────────────────────────




# ── Module-level helper ──────────────────────────────────────────────────────




# ── _all_aux_tasks merges built-in + plugin ──────────────────────────────────


def test_all_aux_tasks_includes_plugin_registered(patched_manager):
    from hermes_cli.main import _AUX_TASKS, _all_aux_tasks

    manifest = PluginManifest(name="hindsight")
    ctx = PluginContext(manifest, patched_manager)
    ctx.register_auxiliary_task(
        key="memory_retain_filter",
        display_name="Memory retain filter",
        description="hindsight pre-retain dedup/extract",
    )

    merged = _all_aux_tasks()
    keys = [k for k, _, _ in merged]
    # Built-ins preserved (and come first)
    builtin_keys = [k for k, _, _ in _AUX_TASKS]
    assert keys[: len(builtin_keys)] == builtin_keys
    # Plugin task appended
    assert "memory_retain_filter" in keys
    plugin_entry = next(t for t in merged if t[0] == "memory_retain_filter")
    assert plugin_entry == (
        "memory_retain_filter",
        "Memory retain filter",
        "hindsight pre-retain dedup/extract",
    )




# ── _reset_aux_to_auto includes plugin tasks ─────────────────────────────────


def test_reset_aux_to_auto_resets_plugin_tasks(tmp_path, monkeypatch, patched_manager):
    """Plugin task with non-auto config gets reset alongside built-ins."""
    from pathlib import Path
    from hermes_cli.config import load_config, save_config
    from hermes_cli.main import _reset_aux_to_auto

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    manifest = PluginManifest(name="plug")
    ctx = PluginContext(manifest, patched_manager)
    ctx.register_auxiliary_task(
        key="my_aux",
        display_name="My Aux",
        description="d",
    )

    # Manually configure the plugin task to non-auto
    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    aux["my_aux"] = {"provider": "openrouter", "model": "gpt-4o", "base_url": "", "api_key": ""}
    save_config(cfg)

    n = _reset_aux_to_auto()
    assert n >= 1

    cfg = load_config()
    assert cfg["auxiliary"]["my_aux"]["provider"] == "auto"
    assert cfg["auxiliary"]["my_aux"]["model"] == ""


# ── auxiliary_client._get_auxiliary_task_config defaults layering ────────────





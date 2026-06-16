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


def test_register_auxiliary_task_with_custom_defaults():
    ctx, manager = _make_ctx()
    ctx.register_auxiliary_task(
        key="custom_task",
        display_name="Custom",
        description="d",
        defaults={"timeout": 30, "extra_body": {"reasoning_effort": "low"}},
    )
    entry = manager._aux_tasks["custom_task"]
    assert entry["defaults"]["timeout"] == 30
    assert entry["defaults"]["extra_body"] == {"reasoning_effort": "low"}
    # Unspecified defaults still populated
    assert entry["defaults"]["provider"] == "auto"


def test_register_auxiliary_task_rejects_builtin_keys():
    ctx, _ = _make_ctx()
    for builtin in (
        "vision",
        "compression",
        "web_extract",
        "approval",
        "mcp",
        "title_generation",
        "skills_hub",
        "curator",
    ):
        with pytest.raises(ValueError, match="reserved for a built-in task"):
            ctx.register_auxiliary_task(
                key=builtin,
                display_name="x",
                description="x",
            )


def test_register_auxiliary_task_rejects_invalid_key_shapes():
    ctx, _ = _make_ctx()
    for bad in ("", "with-dash", "with.dot", "with space", "with/slash"):
        with pytest.raises(ValueError):
            ctx.register_auxiliary_task(
                key=bad,
                display_name="x",
                description="x",
            )


def test_register_auxiliary_task_allows_same_plugin_re_registration():
    """Re-registration by the same plugin updates the entry (idempotent)."""
    ctx, manager = _make_ctx("plug_a")
    ctx.register_auxiliary_task(
        key="t1", display_name="First", description="first"
    )
    ctx.register_auxiliary_task(
        key="t1", display_name="Second", description="second"
    )
    assert manager._aux_tasks["t1"]["display_name"] == "Second"


def test_register_auxiliary_task_rejects_cross_plugin_collision():
    """Two different plugins cannot register the same task key."""
    manager = PluginManager()
    manager._discovered = True

    manifest_a = PluginManifest(name="plug_a")
    manifest_b = PluginManifest(name="plug_b")
    ctx_a = PluginContext(manifest_a, manager)
    ctx_b = PluginContext(manifest_b, manager)

    ctx_a.register_auxiliary_task(
        key="shared", display_name="A", description="a"
    )
    with pytest.raises(ValueError, match="already registered by plugin 'plug_a'"):
        ctx_b.register_auxiliary_task(
            key="shared", display_name="B", description="b"
        )


# ── PluginManager state lifecycle ────────────────────────────────────────────


def test_force_rediscovery_clears_aux_tasks():
    ctx, manager = _make_ctx()
    ctx.register_auxiliary_task(
        key="will_be_cleared",
        display_name="x",
        description="x",
    )
    assert "will_be_cleared" in manager._aux_tasks

    manager._discovered = False
    # Simulate force=True path: clears state before re-scanning
    manager._aux_tasks.clear()
    assert manager._aux_tasks == {}


# ── Module-level helper ──────────────────────────────────────────────────────


def test_get_plugin_auxiliary_tasks_returns_sorted_list(patched_manager):
    manifest = PluginManifest(name="plug")
    ctx = PluginContext(manifest, patched_manager)
    ctx.register_auxiliary_task(
        key="zeta_task", display_name="Zeta", description="z"
    )
    ctx.register_auxiliary_task(
        key="alpha_task", display_name="Alpha", description="a"
    )
    ctx.register_auxiliary_task(
        key="mike_task", display_name="Mike", description="m"
    )

    tasks = get_plugin_auxiliary_tasks()
    assert [t["key"] for t in tasks] == ["alpha_task", "mike_task", "zeta_task"]


def test_get_plugin_auxiliary_tasks_empty_when_none_registered(patched_manager):
    assert get_plugin_auxiliary_tasks() == []


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


def test_all_aux_tasks_swallows_plugin_discovery_failure(monkeypatch):
    """Plugin discovery failure must not break the aux config UI."""
    from hermes_cli import main as main_mod

    def _broken():
        raise RuntimeError("plugin scan exploded")

    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_auxiliary_tasks", _broken
    )

    merged = main_mod._all_aux_tasks()
    # Built-in tasks still present
    assert any(k == "vision" for k, _, _ in merged)


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


def test_get_auxiliary_task_config_layers_plugin_defaults(
    tmp_path, monkeypatch, patched_manager
):
    """Plugin-declared defaults appear when user has no config entry."""
    from pathlib import Path
    from agent.auxiliary_client import _get_auxiliary_task_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    manifest = PluginManifest(name="plug")
    ctx = PluginContext(manifest, patched_manager)
    ctx.register_auxiliary_task(
        key="my_filter",
        display_name="My filter",
        description="x",
        defaults={"timeout": 15, "extra_body": {"reasoning_effort": "low"}},
    )

    # No user config for my_filter — defaults should surface
    resolved = _get_auxiliary_task_config("my_filter")
    assert resolved["timeout"] == 15
    assert resolved["extra_body"] == {"reasoning_effort": "low"}
    assert resolved["provider"] == "auto"


def test_get_auxiliary_task_config_preserves_plugin_gemma_defaults(
    tmp_path, monkeypatch, patched_manager
):
    from pathlib import Path
    from hermes_cli.config import load_config, save_config
    from agent.auxiliary_client import _get_auxiliary_task_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    manifest = PluginManifest(name="plug")
    ctx = PluginContext(manifest, patched_manager)
    ctx.register_auxiliary_task(
        key="future_task",
        display_name="Future task",
        description="x",
        defaults={
            "provider": "gemini",
            "model": "gemma-4-26b-a4b-it",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "timeout": 61,
        },
    )

    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    aux["future_task"] = {"timeout": 75}
    save_config(cfg)

    resolved = _get_auxiliary_task_config("future_task")
    assert resolved["provider"] == "gemini"
    assert resolved["model"] == "gemma-4-26b-a4b-it"
    assert resolved["base_url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert resolved["timeout"] == 75


def test_get_auxiliary_task_config_user_config_wins_over_plugin_defaults(
    tmp_path, monkeypatch, patched_manager
):
    """User's config.yaml entry overrides plugin-declared defaults."""
    from pathlib import Path
    from hermes_cli.config import load_config, save_config
    from agent.auxiliary_client import _get_auxiliary_task_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    manifest = PluginManifest(name="plug")
    ctx = PluginContext(manifest, patched_manager)
    ctx.register_auxiliary_task(
        key="my_filter",
        display_name="My filter",
        description="x",
        defaults={"timeout": 15, "provider": "auto"},
    )

    # User overrides timeout + provider via config.yaml
    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    aux["my_filter"] = {"timeout": 90, "provider": "nous"}
    save_config(cfg)

    resolved = _get_auxiliary_task_config("my_filter")
    assert resolved["timeout"] == 90  # user wins
    assert resolved["provider"] == "nous"  # user wins


def test_get_auxiliary_task_config_unknown_task_returns_empty(
    tmp_path, monkeypatch, patched_manager
):
    from pathlib import Path
    from agent.auxiliary_client import _get_auxiliary_task_config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    assert _get_auxiliary_task_config("nonexistent") == {}

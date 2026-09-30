"""Plugin-registered auxiliary tasks merge into the built-in task list and ``_reset_aux_to_auto``."""

from __future__ import annotations

import pytest

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


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


# ── _all_aux_tasks merges built-in + plugin ──────────────────────────────────


def test_all_aux_tasks_includes_plugin_registered(patched_manager):
    from hermes_cli.main_provider_setup import _AUX_TASKS, _all_aux_tasks

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
    from hermes_cli.main_provider_setup import _reset_aux_to_auto

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


# ── inherit_from: read-time base + precedence ────────────────────────────────


@pytest.fixture
def aux_home(tmp_path, monkeypatch):
    """Empty HERMES_HOME so DEFAULT_CONFIG is the only source of built-in aux values."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _register(patched_manager, **kwargs):
    ctx = PluginContext(PluginManifest(name="plug"), patched_manager)
    return ctx.register_auxiliary_task(
        key=kwargs.pop("key"), display_name="Plug Aux", description="d", **kwargs)


def test_inherit_from_builtin_uses_builtin_defaults(aux_home, patched_manager):
    """An inherited built-in supplies the shape, so the plugin's 60 s default never applies."""
    from agent.auxiliary_client import _get_auxiliary_task_config

    _register(patched_manager, key="plug_aux", inherit_from="mcp")

    cfg = _get_auxiliary_task_config("plug_aux")
    assert cfg["timeout"] == 30          # DEFAULT_CONFIG["auxiliary"]["mcp"], not 60
    assert cfg["provider"] == "auto"
    assert cfg["extra_body"] == {}
    # Keys only the built-in block has survive the merge.
    assert cfg["reasoning_effort"] == ""


def test_plugin_defaults_override_inherited_base(aux_home, patched_manager):
    from agent.auxiliary_client import _get_auxiliary_task_config

    _register(patched_manager, key="plug_aux", inherit_from="mcp",
              defaults={"timeout": 90, "model": "vendor/plugin"})

    cfg = _get_auxiliary_task_config("plug_aux")
    assert cfg["timeout"] == 90
    assert cfg["model"] == "vendor/plugin"
    assert cfg["provider"] == "auto"     # untouched keys still come from the base


def test_user_config_overrides_base_and_plugin_defaults(aux_home, patched_manager):
    from hermes_cli.config import load_config, save_config
    from agent.auxiliary_client import _get_auxiliary_task_config

    _register(patched_manager, key="plug_aux", inherit_from="mcp",
              defaults={"timeout": 90, "model": "vendor/plugin"})

    cfg = load_config()
    cfg.setdefault("auxiliary", {})["plug_aux"] = {"timeout": 5, "model": "vendor/user"}
    save_config(cfg)

    merged = _get_auxiliary_task_config("plug_aux")
    assert merged["timeout"] == 5        # user beats plugin default and the base
    assert merged["model"] == "vendor/user"
    assert merged["provider"] == "auto"  # base still fills what nobody set


def test_inherit_from_another_plugin_task(aux_home, patched_manager):
    from agent.auxiliary_client import _get_auxiliary_task_config

    _register(patched_manager, key="plug_base", inherit_from="mcp")
    _register(patched_manager, key="plug_child", inherit_from="plug_base",
              defaults={"timeout": 45})

    assert _get_auxiliary_task_config("plug_child")["timeout"] == 45
    assert _get_auxiliary_task_config("plug_child")["reasoning_effort"] == ""


def test_inheritance_cycle_does_not_recurse(aux_home, patched_manager, caplog):
    """A hand-edited registry with a cycle must warn and fall back, not blow the stack."""
    import logging

    from agent.auxiliary_client import _get_auxiliary_task_config

    _register(patched_manager, key="plug_a", inherit_from="mcp", defaults={"timeout": 90})
    _register(patched_manager, key="plug_b", inherit_from="plug_a")
    patched_manager._aux_tasks["plug_a"]["inherit_from"] = "plug_b"  # force the cycle

    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
        merged = _get_auxiliary_task_config("plug_a")

    # Cycle cut at plug_a: its own defaults, no inherited base.
    assert merged["timeout"] == 90
    assert "reasoning_effort" not in merged
    assert "circular inherit_from" in caplog.text


# ── inherit_from validation ─────────────────────────────────────────────────


def test_unknown_inherit_from_raises(patched_manager):
    with pytest.raises(ValueError, match="unknown task"):
        _register(patched_manager, key="plug_aux", inherit_from="not_a_task")


def test_self_inherit_raises(patched_manager):
    with pytest.raises(ValueError, match="itself"):
        _register(patched_manager, key="plug_aux", inherit_from="plug_aux")


def test_empty_inherit_from_raises(patched_manager):
    with pytest.raises(ValueError, match="inherit_from"):
        _register(patched_manager, key="plug_aux", inherit_from="")


def test_registration_without_inherit_from_keeps_fixed_shape(aux_home, patched_manager):
    """No inherit_from → today's fixed-shape defaults and no inheritance at read time."""
    from agent.auxiliary_client import _get_auxiliary_task_config

    reg = _register(patched_manager, key="plug_aux")
    entry = patched_manager._aux_tasks[reg.key]
    assert entry["inherit_from"] is None
    assert entry["defaults"] == {"provider": "auto", "model": "", "base_url": "", "api_key": "",
                                 "timeout": 60, "extra_body": {}}
    assert _get_auxiliary_task_config("plug_aux") == entry["defaults"]

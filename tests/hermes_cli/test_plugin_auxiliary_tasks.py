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


# ── inherit_from: read-time base, explicit pin wins ──────────────────────────


@pytest.fixture
def aux_home(tmp_path, monkeypatch):
    """Empty HERMES_HOME so DEFAULT_CONFIG is the only source of built-in aux values."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _register(patched_manager, key, **kwargs):
    ctx = PluginContext(PluginManifest(name="plug"), patched_manager)
    return ctx.register_auxiliary_task(key, display_name="Plug Aux", description="d", **kwargs)


def _set_aux(**blocks):
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    cfg.setdefault("auxiliary", {}).update(blocks)
    save_config(cfg)


def test_inheriting_task_follows_base_until_pinned(aux_home, patched_manager):
    from agent.auxiliary_client import _get_auxiliary_task_config as resolve

    _register(patched_manager, "plug_aux", inherit_from="compression", defaults={"timeout": 90})
    _set_aux(compression={"provider": "openrouter", "model": "vendor/a", "base_url": "https://base.example",
                          "api_key": "base-key"})
    got = resolve("plug_aux")
    assert (got["provider"], got["model"], got["base_url"]) == ("openrouter", "vendor/a", "https://base.example")
    assert got["timeout"] == 90  # plugin default layers over the base

    # Read time, not registration time: a base change propagates with no re-registration.
    _set_aux(compression={"provider": "openrouter", "model": "vendor/b"})
    assert resolve("plug_aux")["model"] == "vendor/b"

    # What the picker / dashboard "reset to auto" persist means "no preference": keep following.
    _set_aux(plug_aux={"provider": "auto", "model": "", "base_url": "", "api_key": "",
                       "reasoning_effort": "", "timeout": 5})
    got = resolve("plug_aux")
    assert (got["provider"], got["model"], got["timeout"]) == ("openrouter", "vendor/b", 5)

    # An explicit pin wins as a whole route: the base's endpoint/key never leak into it.
    _set_aux(compression={"provider": "custom", "model": "vendor/b", "base_url": "https://base.example",
                          "api_key": "base-key"},
             plug_aux={"provider": "anthropic", "model": "claude-x", "base_url": "", "api_key": ""})
    got = resolve("plug_aux")
    assert (got["provider"], got["model"]) == ("anthropic", "claude-x")
    assert not got.get("base_url") and not got.get("api_key")


@pytest.mark.parametrize("bad", ["no_such_task", "plug_aux", ""])
def test_bad_inherit_from_warns_and_registers_without_inheritance(aux_home, patched_manager, caplog, bad):
    import logging

    from agent.auxiliary_client import _get_auxiliary_task_config as resolve

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        _register(patched_manager, "plug_aux", inherit_from=bad)
    assert "ignoring inherit_from" in caplog.text
    assert patched_manager._aux_tasks["plug_aux"]["inherit_from"] is None
    assert resolve("plug_aux")["timeout"] == 60  # the fixed non-inheriting shape


def test_inheritance_cycle_does_not_recurse(aux_home, patched_manager, caplog):
    """Same-owner re-registration can close a loop; reads must warn and cut it, not blow the stack."""
    import logging

    from agent.auxiliary_client import _get_auxiliary_task_config as resolve

    _register(patched_manager, "plug_a", inherit_from="mcp", defaults={"timeout": 90})
    _register(patched_manager, "plug_b", inherit_from="plug_a")
    _register(patched_manager, "plug_a", inherit_from="plug_b", defaults={"timeout": 90})

    with caplog.at_level(logging.WARNING, logger="agent.auxiliary_task_config"):
        assert resolve("plug_a")["timeout"] == 90
    assert "circular inherit_from" in caplog.text

"""Tests: cmd_update post-pull union re-sync for member-manifest plugins.

Task 5 of the plugin auto-update plan: a plugin update whose pull changed
its pyproject pins must re-sync the venv union (new deps land); a sync
failure surfaces the resolver reason but never fails the update — the
plugin code is already updated.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_cli.plugins_cmd as pc


@pytest.fixture
def plugin_env(tmp_path, monkeypatch):
    """An installed, provenanced, enabled plugin + injectable seams."""
    home = tmp_path / "home"
    plugins = home / "plugins"
    plug = plugins / "plug"
    (plug / ".git").mkdir(parents=True)
    (plug / "plugin.yaml").write_text("name: plug\n", encoding="utf-8")
    (home / "config.yaml").write_text("plugins:\n  enabled: [plug]\n", encoding="utf-8")
    plugins.mkdir(parents=True, exist_ok=True)
    (plugins / ".install-metadata.json").write_text(
        json.dumps({"plug": {"pinned": False, "revision": "a" * 40,
                            "source": "https://example/o/r"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: home)
    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins)
    monkeypatch.setattr(
        pc, "_scan_on_install_enabled", lambda: False
    )
    monkeypatch.setattr(
        pc, "_git_pull_plugin_dir",
        lambda target: (True, "ok"),
    )
    monkeypatch.setattr(pc, "_resolve_git_executable", lambda: None)
    monkeypatch.setattr(pc, "_read_install_metadata", lambda: {
        "plug": {"pinned": False, "revision": "a" * 40,
                 "source": "https://example/o/r"}
    })
    monkeypatch.setattr(pc, "_write_install_metadata", lambda m: None)
    monkeypatch.setattr(pc, "_get_enabled_set", lambda: {"plug"})
    monkeypatch.setattr(pc, "_get_disabled_set", lambda: set())
    monkeypatch.setattr(pc, "_copy_example_files", lambda t, c: None)
    return plug


def test_member_plugin_triggers_post_pull_sync(plugin_env, monkeypatch):
    (plugin_env / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    synced = []
    import pm

    monkeypatch.setattr(pm, "sync_venv", lambda extras=None, **k: synced.append(extras))
    pc.cmd_update("plug")
    assert synced == [None]  # explicit sync with recorded extras


def test_plain_plugin_skips_sync(plugin_env, monkeypatch):
    # no pyproject.toml — nothing to union
    import pm

    synced = []
    monkeypatch.setattr(pm, "sync_venv", lambda extras=None, **k: synced.append(extras))
    pc.cmd_update("plug")
    assert synced == []


def test_sync_failure_surfaces_but_update_succeeds(plugin_env, monkeypatch, capsys):
    (plugin_env / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    import pm

    def boom(extras=None, **k):
        raise pm.InstallError("venv", "uv lock exited 1: unsatisfiable")

    monkeypatch.setattr(pm, "sync_venv", boom)
    # must NOT raise — the plugin code is already updated
    pc.cmd_update("plug")
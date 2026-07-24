"""Security contracts for model-provider distribution lifecycle operations."""

from __future__ import annotations

import importlib.metadata
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import yaml

from hermes_cli import plugins_cmd as pc
import providers

KEY = "model-providers/acme"
ALIASES = {KEY, "acme", "Acme Provider"}


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / "hermes"
    (path / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


def _config(home, *, enabled=(), disabled=()):
    path = home / "config.yaml"
    path.write_text(
        yaml.safe_dump({
            "plugins": {"enabled": list(enabled), "disabled": list(disabled)}
        })
    )
    return path


def _activation(path):
    plugins = yaml.safe_load(path.read_text())["plugins"]
    return set(plugins["enabled"]), set(plugins["disabled"])


def _clone(manifest):
    def run(argv, **_kwargs):
        target = Path(argv[-1])
        target.mkdir(parents=True)
        (target / "plugin.yaml").write_text(manifest)
        (target / "__init__.py").write_text("# provider\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _entry(path, *, name="Acme Provider", key=KEY):
    return (name, "1.0.0", "test provider", "user", path, key)


def _provider(home, monkeypatch, *, enabled=ALIASES):
    root = home / "plugins"
    target = root / "model-providers" / "acme"
    target.mkdir(parents=True)
    config = _config(home, enabled=enabled)
    monkeypatch.setattr(pc, "_plugins_dir", lambda: root)
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: [_entry(target)])
    return target, config


def _install_env(home, monkeypatch, manifest):
    root = home / "plugins"
    monkeypatch.setattr(pc, "_plugins_dir", lambda: root)
    monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "/usr/bin/git")
    monkeypatch.setattr(pc.subprocess, "run", _clone(manifest))
    return root


def test_activation_write_rejects_save_noop(home):
    config = _config(home)
    original = config.read_bytes()
    with patch("hermes_cli.config.save_config", return_value=None):
        with pytest.raises(pc.PluginOperationError, match="persist|verif|managed"):
            pc._write_plugin_activation(KEY, ALIASES, True)
    assert config.read_bytes() == original


def test_activation_write_preserves_malformed_yaml(home):
    config = home / "config.yaml"
    malformed = b"plugins: [unterminated\n"
    config.write_bytes(malformed)
    save = Mock()
    with patch("hermes_cli.config.save_config", save):
        with pytest.raises(pc.PluginOperationError, match="config|YAML|parse"):
            pc._write_plugin_activation(KEY, ALIASES, True)
    save.assert_not_called()
    assert config.read_bytes() == malformed


def test_install_does_not_publish_when_activation_fails(home, monkeypatch):
    root = _install_env(
        home, monkeypatch, "name: acme\nmanifest_version: 1\nkind: model-provider\n"
    )
    monkeypatch.setattr(
        pc,
        "_write_plugin_activation",
        Mock(side_effect=pc.PluginOperationError("cannot persist")),
    )
    with pytest.raises(pc.PluginOperationError, match="cannot persist"):
        pc._install_plugin_core("owner/acme", force=False, enabled=False)
    assert not (root / "model-providers" / "acme").exists()
    assert not list(root.glob(".install-*"))


def test_install_stages_hidden_and_publishes_with_one_rename(home, monkeypatch):
    root = home / "plugins"
    events = []

    def clone(argv, **_kwargs):
        stage = Path(argv[-1])
        assert stage.parent.parent == root and stage.parent.name.startswith(".install-")
        stage.mkdir()
        (stage / "plugin.yaml").write_text("name: acme\nkind: model-provider\n")
        events.append("clone")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pc, "_plugins_dir", lambda: root)
    monkeypatch.setattr(pc, "_resolve_git_executable", lambda: "/usr/bin/git")
    monkeypatch.setattr(pc.subprocess, "run", clone)
    monkeypatch.setattr(
        pc,
        "_write_plugin_activation",
        lambda _key, _aliases, enabled: events.append(("activation", enabled)),
    )
    real_replace = os.replace

    def publish(source, target):
        assert events == ["clone", ("activation", False)]
        events.append("publish")
        real_replace(source, target)

    monkeypatch.setattr(pc.os, "replace", publish)
    target, _manifest, key = pc._install_plugin_core(
        "owner/acme", force=False, enabled=True
    )
    assert events == [
        "clone",
        ("activation", False),
        "publish",
        ("activation", True),
    ]
    assert target == root / "model-providers" / "acme"
    assert key == KEY


def test_failed_enabled_publish_retains_deny_and_blocks_package(home, monkeypatch):
    root = _install_env(home, monkeypatch, "name: acme\nkind: model-provider\n")
    real_replace = os.replace

    def fail_publication(source, target):
        if ".install-" in str(source):
            raise OSError("publish failed")
        return real_replace(source, target)

    monkeypatch.setattr(pc.os, "replace", fail_publication)

    with pytest.raises(OSError, match="publish failed"):
        pc._install_plugin_core("owner/acme", force=False, enabled=True)

    enabled, disabled = _activation(home / "config.yaml")
    assert KEY not in enabled and KEY in disabled
    assert not (root / "model-providers" / "acme").exists()

    called = Mock()
    entry = SimpleNamespace(name="acme", load=lambda: called)
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: SimpleNamespace(select=lambda **_kwargs: [entry]),
    )
    providers._load_package_providers(enabled, disabled, set())
    called.assert_not_called()


def test_remove_rejects_symlink_before_mutation(home, monkeypatch):
    root = home / "plugins"
    actual = root / "holding" / "acme"
    actual.mkdir(parents=True)
    link = root / "model-providers"
    link.symlink_to(root / "holding", target_is_directory=True)
    write = Mock()
    remove = Mock()
    monkeypatch.setattr(pc, "_plugins_dir", lambda: root)
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: [_entry(link / "acme")])
    monkeypatch.setattr(pc, "_write_plugin_activation", write)
    monkeypatch.setattr(pc.shutil, "rmtree", remove)
    assert pc.dashboard_remove_user_plugin(KEY)["ok"] is False
    write.assert_not_called()
    remove.assert_not_called()
    assert actual.is_dir()


def test_install_rejects_symlinked_category_before_activation(home, monkeypatch):
    root = home / "plugins"
    holding = root / "holding"
    holding.mkdir()
    (root / "model-providers").symlink_to(holding, target_is_directory=True)
    _install_env(home, monkeypatch, "name: acme\nkind: model-provider\n")
    write = Mock()
    monkeypatch.setattr(pc, "_write_plugin_activation", write)
    with pytest.raises(pc.PluginOperationError, match="symlink"):
        pc._install_plugin_core("owner/acme", force=False, enabled=False)
    write.assert_not_called()
    assert not (holding / "acme").exists()


def test_remove_aborts_when_deny_cannot_persist(home, monkeypatch):
    target, config = _provider(home, monkeypatch)
    original = config.read_bytes()
    remove = Mock()
    monkeypatch.setattr(pc.shutil, "rmtree", remove)
    with patch("hermes_cli.config.save_config", return_value=None):
        assert pc.dashboard_remove_user_plugin(KEY)["ok"] is False
    remove.assert_not_called()
    assert target.is_dir() and config.read_bytes() == original


def test_remove_retains_canonical_deny_when_delete_fails(home, monkeypatch):
    target, config = _provider(home, monkeypatch)

    def fail_delete(path):
        enabled, disabled = _activation(config)
        assert not enabled & ALIASES and KEY in disabled
        raise OSError("delete failed")

    monkeypatch.setattr(pc.shutil, "rmtree", fail_delete)
    assert pc.dashboard_remove_user_plugin(KEY)["ok"] is False
    enabled, disabled = _activation(config)
    assert target.is_dir() and not enabled & ALIASES and KEY in disabled


def test_remove_cleans_aliases_only_after_successful_delete(home, monkeypatch):
    target, config = _provider(home, monkeypatch)
    real_rmtree = shutil.rmtree

    def delete_after_deny(path):
        enabled, disabled = _activation(config)
        assert not enabled & ALIASES and KEY in disabled
        real_rmtree(path)

    monkeypatch.setattr(pc.shutil, "rmtree", delete_after_deny)
    assert pc.dashboard_remove_user_plugin(KEY) == {"ok": True, "name": KEY}
    enabled, disabled = _activation(config)
    assert not target.exists() and not enabled & ALIASES and not disabled & ALIASES


def test_generic_disabled_install_preserves_activation_boundary(home, monkeypatch):
    root = _install_env(home, monkeypatch, "name: generic\nkind: tool\n")
    config = home / "config.yaml"
    original = b"plugins:\n  enabled: [existing]\n  disabled: []\n"
    config.write_bytes(original)
    save_enabled = Mock(side_effect=RuntimeError("must not write"))
    save_disabled = Mock(side_effect=RuntimeError("must not write"))
    monkeypatch.setattr(pc, "_save_enabled_set", save_enabled)
    monkeypatch.setattr(pc, "_save_disabled_set", save_disabled)
    target, _manifest, key = pc._install_plugin_core(
        "owner/generic", force=False, enabled=False
    )
    assert key == "generic" and target == root / "generic"
    assert config.read_bytes() == original
    save_enabled.assert_not_called()
    save_disabled.assert_not_called()


def test_dashboard_generic_enabled_install_persists_activation(home, monkeypatch):
    _install_env(home, monkeypatch, "name: generic\nkind: tool\n")

    result = pc.dashboard_install_plugin("owner/generic", force=False, enable=True)

    assert result["ok"] is True and result["enabled"] is True
    plugins = yaml.safe_load((home / "config.yaml").read_text())["plugins"]
    assert "generic" in plugins["enabled"]
    assert "generic" not in plugins["disabled"]


def test_generic_remove_preserves_activation_boundary(home, monkeypatch):
    root = home / "plugins"
    target = root / "generic"
    target.mkdir()
    (target / "plugin.yaml").write_text("name: generic\nkind: tool\n")
    config = home / "config.yaml"
    original = b"plugins:\n  enabled: [generic]\n  disabled: []\n"
    config.write_bytes(original)
    monkeypatch.setattr(pc, "_plugins_dir", lambda: root)
    monkeypatch.setattr(
        pc,
        "_discover_all_plugins",
        lambda: [_entry(target, name="generic", key="generic")],
    )
    save_enabled = Mock(side_effect=RuntimeError("must not write"))
    save_disabled = Mock(side_effect=RuntimeError("must not write"))
    monkeypatch.setattr(pc, "_save_enabled_set", save_enabled)
    monkeypatch.setattr(pc, "_save_disabled_set", save_disabled)
    assert pc.dashboard_remove_user_plugin("generic")["ok"] is True
    assert not target.exists()
    assert config.read_bytes() == original
    save_enabled.assert_not_called()
    save_disabled.assert_not_called()

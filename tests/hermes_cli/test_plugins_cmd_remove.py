"""Removal lifecycle tests for ``hermes plugins``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hermes_cli.plugins_cmd import cmd_remove, dashboard_remove_user_plugin


@pytest.fixture
def hermes_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "hermes"
    (home / "plugins").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _install_plugin(
    hermes_home: Path,
    directory_name: str,
    *,
    manifest_name: str | None = None,
) -> Path:
    plugin_dir = hermes_home / "plugins" / directory_name
    plugin_dir.mkdir()
    if manifest_name is not None:
        (plugin_dir / "plugin.yaml").write_text(
            f"name: {manifest_name}\nversion: 1.0.0\n",
            encoding="utf-8",
        )
    return plugin_dir


def _write_config(hermes_home: Path, plugins: dict) -> Path:
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"plugins": plugins}, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _managed_home_dirs(hermes_home: Path) -> None:
    for subdir in ("cron", "sessions", "logs", "memories"):
        (hermes_home / subdir).mkdir(parents=True, exist_ok=True)


def test_remove_clears_all_config_aliases(hermes_home):
    plugin_dir = _install_plugin(
        hermes_home,
        "directory-name",
        manifest_name="manifest-name",
    )
    config_path = _write_config(
        hermes_home,
        {
            "enabled": ["directory-name", "manifest-name", "other"],
            "disabled": ["directory-name", "manifest-name", "other-disabled"],
            "entries": {
                "directory-name": {"allow_tool_override": True},
                "manifest-name": {"allow_tool_override": False},
                "other": {"allow_tool_override": True},
            },
        },
    )

    cmd_remove("directory-name")

    assert not plugin_dir.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["plugins"]["enabled"] == ["other"]
    assert config["plugins"]["disabled"] == ["other-disabled"]
    assert config["plugins"]["entries"] == {"other": {"allow_tool_override": True}}


def test_remove_absent_plugin_does_not_mutate_config(hermes_home):
    config_path = hermes_home / "config.yaml"
    original = yaml.safe_dump(
        {
            "plugins": {
                "enabled": ["absent-plugin", "other"],
                "disabled": ["absent-plugin"],
                "entries": {"absent-plugin": {"allow_tool_override": True}},
            }
        },
        sort_keys=False,
    )
    config_path.write_text(original, encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        cmd_remove("absent-plugin")

    assert exc_info.value.code == 1
    assert config_path.read_text(encoding="utf-8") == original


def test_remove_keeps_plugin_when_config_save_fails(hermes_home):
    plugin_dir = _install_plugin(hermes_home, "test-plugin")
    _write_config(hermes_home, {"enabled": ["test-plugin"]})

    with patch(
        "hermes_cli.config.save_config",
        side_effect=RuntimeError("config unavailable"),
    ):
        with pytest.raises(RuntimeError, match="config unavailable"):
            cmd_remove("test-plugin")

    assert plugin_dir.is_dir()


def test_remove_keeps_plugin_when_managed(hermes_home):
    plugin_dir = _install_plugin(hermes_home, "test-plugin")
    _managed_home_dirs(hermes_home)
    config_path = _write_config(hermes_home, {"enabled": ["test-plugin"]})
    original = config_path.read_text(encoding="utf-8")

    with patch("hermes_cli.config.is_managed", return_value=True):
        with pytest.raises(RuntimeError, match="managed mode"):
            cmd_remove("test-plugin")

    assert plugin_dir.is_dir()
    assert config_path.read_text(encoding="utf-8") == original


def test_dashboard_remove_keeps_plugin_when_managed(hermes_home):
    plugin_dir = _install_plugin(hermes_home, "test-plugin")
    _managed_home_dirs(hermes_home)
    config_path = _write_config(
        hermes_home,
        {
            "enabled": ["test-plugin", "other"],
            "entries": {"test-plugin": {"allow_tool_override": True}},
        },
    )
    original = config_path.read_text(encoding="utf-8")

    with (
        patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=[]),
        patch("hermes_cli.config.is_managed", return_value=True),
    ):
        result = dashboard_remove_user_plugin("test-plugin")

    assert result["ok"] is False
    assert "managed mode" in result["error"]
    assert plugin_dir.is_dir()
    assert config_path.read_text(encoding="utf-8") == original


def test_dashboard_remove_clears_plugin_config_state(hermes_home):
    plugin_dir = _install_plugin(hermes_home, "test-plugin")
    config_path = _write_config(
        hermes_home,
        {
            "enabled": ["test-plugin", "other"],
            "disabled": ["test-plugin"],
            "entries": {"test-plugin": {"allow_tool_override": True}},
        },
    )

    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=[]):
        result = dashboard_remove_user_plugin("test-plugin")

    assert result["ok"] is True
    assert not plugin_dir.exists()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["plugins"]["enabled"] == ["other"]
    assert config["plugins"]["disabled"] == []
    assert config["plugins"]["entries"] == {}

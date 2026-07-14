"""CLI enable/disable must keep platform_toolsets in sync.

The dashboard path already calls ``_toggle_plugin_toolset``; the CLI
``cmd_enable`` / ``cmd_disable`` used to only flip the allow/deny lists,
leaving stale toolset entries after partial configs or dashboard/CLI drift.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


def _make_plugin_dir(parent: Path, name: str, manifest: dict) -> Path:
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
    (d / "__init__.py").write_text("def register(ctx): pass\n", encoding="utf-8")
    return d


@pytest.fixture
def tool_plugin_env(tmp_path):
    """A flat user plugin that reports tools (via mocked toolset key)."""
    _make_plugin_dir(
        tmp_path,
        "tooly",
        {
            "name": "tooly",
            "version": "1.0.0",
            "description": "provides tools",
            "provides_tools": ["tooly_do"],
        },
    )
    return tmp_path


def _patch_env(mock_user, mock_bundled, env):
    mock_user.return_value = env
    mock_bundled.return_value = env / "nonexistent"


class TestEnableDisableSyncsPlatformToolsets:
    @patch("hermes_cli.plugins_cmd._get_plugin_toolset_key", return_value="plugin:tooly")
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_enable_adds_toolset_to_platform_toolsets(
        self,
        mock_en,
        mock_dis,
        mock_save_en,
        mock_save_dis,
        mock_user,
        mock_bundled,
        mock_ts_key,
        tool_plugin_env,
        tmp_path,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        _patch_env(mock_user, mock_bundled, tool_plugin_env)

        cfg = {
            "platform_toolsets": {
                "cli": ["hermes-cli"],
                "telegram": ["hermes-telegram"],
            }
        }
        saved = {}

        def _load():
            return cfg

        def _save(c):
            saved["config"] = c

        monkeypatch.setattr("hermes_cli.config.load_config", _load)
        monkeypatch.setattr("hermes_cli.config.save_config", _save)

        plugins_cmd.cmd_enable("tooly", allow_tool_override=False)

        assert "plugin:tooly" in saved["config"]["platform_toolsets"]["cli"]
        assert "plugin:tooly" in saved["config"]["platform_toolsets"]["telegram"]
        mock_save_en.assert_called_once()

    @patch("hermes_cli.plugins_cmd._get_plugin_toolset_key", return_value="plugin:tooly")
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set")
    def test_already_enabled_still_reconciles_stale_toolset(
        self,
        mock_en,
        mock_dis,
        mock_save_en,
        mock_save_dis,
        mock_user,
        mock_bundled,
        mock_ts_key,
        tool_plugin_env,
        monkeypatch,
    ):
        """Re-running enable must repair platform_toolsets even when already enabled."""
        from hermes_cli import plugins_cmd

        _patch_env(mock_user, mock_bundled, tool_plugin_env)
        mock_en.return_value = {"tooly"}  # already enabled
        mock_dis.return_value = set()

        cfg = {"platform_toolsets": {"cli": ["hermes-cli"]}}  # missing plugin toolset
        saved = {}
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
        monkeypatch.setattr("hermes_cli.config.save_config", lambda c: saved.update(config=c))

        plugins_cmd.cmd_enable("tooly", allow_tool_override=False)

        assert "plugin:tooly" in saved["config"]["platform_toolsets"]["cli"]
        # Enable set already had the key — no need to rewrite allow-list.
        mock_save_en.assert_not_called()

    @patch("hermes_cli.plugins_cmd._get_plugin_toolset_key", return_value="plugin:tooly")
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set())
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value={"tooly"})
    def test_disable_removes_toolset_from_platform_toolsets(
        self,
        mock_en,
        mock_dis,
        mock_save_en,
        mock_save_dis,
        mock_user,
        mock_bundled,
        mock_ts_key,
        tool_plugin_env,
        monkeypatch,
    ):
        from hermes_cli import plugins_cmd

        _patch_env(mock_user, mock_bundled, tool_plugin_env)

        cfg = {
            "platform_toolsets": {
                "cli": ["hermes-cli", "plugin:tooly"],
                "telegram": ["plugin:tooly"],
            }
        }
        saved = {}
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
        monkeypatch.setattr("hermes_cli.config.save_config", lambda c: saved.update(config=c))

        plugins_cmd.cmd_disable("tooly")

        assert "plugin:tooly" not in saved["config"]["platform_toolsets"]["cli"]
        assert "plugin:tooly" not in saved["config"]["platform_toolsets"]["telegram"]
        mock_save_dis.assert_called_once()

    @patch("hermes_cli.plugins_cmd._get_plugin_toolset_key", return_value="plugin:tooly")
    @patch("hermes_cli.plugins.get_bundled_plugins_dir")
    @patch("hermes_cli.plugins_cmd._plugins_dir")
    @patch("hermes_cli.plugins_cmd._save_disabled_set")
    @patch("hermes_cli.plugins_cmd._save_enabled_set")
    @patch("hermes_cli.plugins_cmd._get_disabled_set", return_value={"tooly"})
    @patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set())
    def test_already_disabled_still_reconciles_stale_toolset(
        self,
        mock_en,
        mock_dis,
        mock_save_en,
        mock_save_dis,
        mock_user,
        mock_bundled,
        mock_ts_key,
        tool_plugin_env,
        monkeypatch,
    ):
        """Re-running disable must strip stale toolset entries."""
        from hermes_cli import plugins_cmd

        _patch_env(mock_user, mock_bundled, tool_plugin_env)

        cfg = {"platform_toolsets": {"cli": ["hermes-cli", "plugin:tooly"]}}
        saved = {}
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
        monkeypatch.setattr("hermes_cli.config.save_config", lambda c: saved.update(config=c))

        plugins_cmd.cmd_disable("tooly")

        assert "plugin:tooly" not in saved["config"]["platform_toolsets"]["cli"]
        mock_save_dis.assert_not_called()

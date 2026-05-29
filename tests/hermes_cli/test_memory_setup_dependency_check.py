"""Tests for memory_setup._install_dependencies return value and setup abort behavior.

Regression tests for:
    https://github.com/NousResearch/hermes-agent/issues/25086
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _install_dependencies return value
# ---------------------------------------------------------------------------

class TestInstallDependenciesReturn:
    """Verify _install_dependencies returns True/False correctly."""

    def test_returns_true_when_no_plugin_dir(self, tmp_path: Path) -> None:
        """No plugin directory -> no deps to check -> True."""
        from hermes_cli.memory_setup import _install_dependencies

        with patch("plugins.memory.find_provider_dir", return_value=None):
            assert _install_dependencies("nonexistent") is True

    def test_returns_true_when_no_plugin_yaml(self, tmp_path: Path) -> None:
        """Plugin dir exists but no plugin.yaml -> True."""
        from hermes_cli.memory_setup import _install_dependencies

        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        with patch("plugins.memory.find_provider_dir", return_value=plugin_dir):
            assert _install_dependencies("test_plugin") is True

    def test_returns_true_when_no_pip_deps(self, tmp_path: Path) -> None:
        """plugin.yaml has no pip_dependencies -> True."""
        from hermes_cli.memory_setup import _install_dependencies

        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text("name: test\nversion: 1.0\n")
        with patch("plugins.memory.find_provider_dir", return_value=plugin_dir):
            assert _install_dependencies("test_plugin") is True

    def test_returns_true_when_all_deps_importable(self, tmp_path: Path) -> None:
        """All pip_dependencies already installed -> True without installing."""
        from hermes_cli.memory_setup import _install_dependencies

        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: test\npip_dependencies:\n  - json\n"  # json is always available
        )
        with patch("plugins.memory.find_provider_dir", return_value=plugin_dir):
            assert _install_dependencies("test_plugin") is True

    def test_returns_false_when_uv_not_found(self, tmp_path: Path) -> None:
        """Missing dep + no uv -> False."""
        from hermes_cli.memory_setup import _install_dependencies

        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: test\npip_dependencies:\n  - nonexistent-fake-pkg-xyz\n"
        )
        with (
            patch("plugins.memory.find_provider_dir", return_value=plugin_dir),
            patch("shutil.which", return_value=None),
        ):
            assert _install_dependencies("test_plugin") is False

    def test_returns_false_when_install_fails(self, tmp_path: Path) -> None:
        """Missing dep + uv install fails -> False."""
        import subprocess

        from hermes_cli.memory_setup import _install_dependencies

        plugin_dir = tmp_path / "test_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.yaml").write_text(
            "name: test\npip_dependencies:\n  - nonexistent-fake-pkg-xyz\n"
        )
        with (
            patch("plugins.memory.find_provider_dir", return_value=plugin_dir),
            patch("shutil.which", return_value="/usr/bin/uv"),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "uv", stderr=b"fail"),
            ),
        ):
            assert _install_dependencies("test_plugin") is False


# ---------------------------------------------------------------------------
# cmd_setup aborts when deps are missing
# ---------------------------------------------------------------------------

class TestCmdSetupAbortOnMissingDeps:
    """Verify cmd_setup / cmd_setup_provider abort when deps are missing."""

    def test_cmd_setup_aborts_when_install_fails(self, tmp_path: Path, capsys) -> None:
        """cmd_setup should abort with message when _install_dependencies returns False."""
        from hermes_cli.memory_setup import cmd_setup

        mock_provider = MagicMock()
        mock_provider.get_config_schema.return_value = []

        with (
            patch("hermes_cli.memory_setup._get_available_providers",
                  return_value=[("mem0", "requires API key", mock_provider)]),
            patch("hermes_cli.memory_setup._curses_select", return_value=0),
            patch("hermes_cli.memory_setup._install_dependencies", return_value=False),
        ):
            cmd_setup(None)

        captured = capsys.readouterr()
        assert "Cannot configure" in captured.out
        assert "dependencies are missing" in captured.out

    def test_cmd_setup_provider_aborts_when_install_fails(self, tmp_path: Path, capsys) -> None:
        """cmd_setup_provider should abort when _install_dependencies returns False."""
        from hermes_cli.memory_setup import cmd_setup_provider

        mock_provider = MagicMock()

        with (
            patch("hermes_cli.memory_setup._get_available_providers",
                  return_value=[("mem0", "requires API key", mock_provider)]),
            patch("hermes_cli.memory_setup._install_dependencies", return_value=False),
        ):
            cmd_setup_provider("mem0")

        captured = capsys.readouterr()
        assert "Cannot configure" in captured.out
        assert "dependencies are missing" in captured.out

    def test_cmd_setup_proceeds_when_deps_ok(self, tmp_path: Path) -> None:
        """cmd_setup should proceed normally when _install_dependencies returns True."""
        from hermes_cli.memory_setup import cmd_setup

        mock_provider = MagicMock()
        # Mock has auto-created post_setup, so cmd_setup delegates to it

        with (
            patch("hermes_cli.memory_setup._get_available_providers",
                  return_value=[("test", "local", mock_provider)]),
            patch("hermes_cli.memory_setup._curses_select", return_value=0),
            patch("hermes_cli.memory_setup._install_dependencies", return_value=True),
            patch("hermes_cli.config.load_config", return_value={}),
            patch("hermes_cli.config.save_config"),
            patch("hermes_constants.get_hermes_home", return_value=tmp_path),
        ):
            cmd_setup(None)

        # post_setup should have been called (setup proceeded past dep check)
        mock_provider.post_setup.assert_called_once()

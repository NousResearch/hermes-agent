"""Tests for hermes desktop launcher install/uninstall."""

from __future__ import annotations

import platform
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import pytest

# Skip all tests on non-macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="macOS-only launcher tests",
)


class TestInstallLauncher(TestCase):
    """Test install_launcher function."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

        # Create mock hermes-agent directory structure
        self.hermes_agent_dir = self.tmp_path / "hermes-agent"
        self.desktop_dir = self.hermes_agent_dir / "apps" / "desktop"
        self.release_dir = self.desktop_dir / "release" / "mac-arm64"
        self.app_bundle = self.release_dir / "Hermes.app"
        self.app_contents = self.app_bundle / "Contents"
        self.app_resources = self.app_contents / "Resources"
        self.app_macos = self.app_contents / "MacOS"

        # Create the structure
        self.app_resources.mkdir(parents=True, exist_ok=True)
        self.app_macos.mkdir(parents=True, exist_ok=True)

        # Create mock icon
        self.icon_path = self.app_resources / "icon.icns"
        self.icon_path.write_bytes(b"mock icon data")

        # Create mock executable
        (self.app_macos / "Hermes").write_text("#!/bin/bash\nexit 0")
        (self.app_macos / "Hermes").chmod(0o755)

        # Mock Desktop and Applications directories
        self.desktop_dir_mock = self.tmp_path / "Desktop"
        self.app_dir_mock = self.tmp_path / "Applications"
        self.desktop_dir_mock.mkdir(parents=True, exist_ok=True)
        self.app_dir_mock.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    @mock.patch("hermes_cli.subcommands.gui_launcher.Path.home")
    def test_install_launcher_creates_files(self, mock_home, mock_system):
        """Test that install_launcher creates .command and .app files."""
        from hermes_cli.subcommands.gui_launcher import install_launcher

        mock_system.return_value = "Darwin"
        mock_home.return_value = self.tmp_path

        cwd = self.tmp_path / "Projects" / "example"
        cwd.mkdir(parents=True, exist_ok=True)

        result = install_launcher(
            cwd=cwd,
            name="Test Launcher",
            hermes_agent_dir=self.hermes_agent_dir,
        )

        self.assertTrue(result)

        # Check .command file
        command_file = self.desktop_dir_mock / "Test Launcher.command"
        self.assertTrue(command_file.exists())
        self.assertIn("hermes desktop --skip-build", command_file.read_text())

        # Check .app bundle
        app_bundle = self.app_dir_mock / "Test Launcher.app"
        self.assertTrue(app_bundle.exists())
        self.assertTrue((app_bundle / "Contents" / "Info.plist").exists())
        self.assertTrue((app_bundle / "Contents" / "PkgInfo").exists())
        self.assertTrue((app_bundle / "Contents" / "MacOS" / "Test Launcher").exists())
        self.assertTrue((app_bundle / "Contents" / "Resources" / "icon.icns").exists())

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    @mock.patch("hermes_cli.subcommands.gui_launcher.Path.home")
    def test_install_launcher_default_cwd(self, mock_home, mock_system):
        """Test that install_launcher defaults to $HOME when cwd is None."""
        from hermes_cli.subcommands.gui_launcher import install_launcher

        mock_system.return_value = "Darwin"
        mock_home.return_value = self.tmp_path

        result = install_launcher(
            name="Test Launcher",
            hermes_agent_dir=self.hermes_agent_dir,
        )

        self.assertTrue(result)

        # Check that launcher uses $HOME
        command_file = self.desktop_dir_mock / "Test Launcher.command"
        self.assertIn(str(self.tmp_path), command_file.read_text())

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    @mock.patch("hermes_cli.subcommands.gui_launcher.Path.home")
    def test_install_launcher_replaces_existing(self, mock_home, mock_system):
        """Test that install_launcher replaces existing launcher files."""
        from hermes_cli.subcommands.gui_launcher import install_launcher

        mock_system.return_value = "Darwin"
        mock_home.return_value = self.tmp_path

        # Create existing launcher
        existing_command = self.desktop_dir_mock / "Test Launcher.command"
        existing_command.write_text("old content")

        existing_app = self.app_dir_mock / "Test Launcher.app"
        existing_app.mkdir(parents=True, exist_ok=True)
        (existing_app / "old_file.txt").write_text("old")

        cwd = self.tmp_path / "Projects" / "example"
        cwd.mkdir(parents=True, exist_ok=True)

        result = install_launcher(
            cwd=cwd,
            name="Test Launcher",
            hermes_agent_dir=self.hermes_agent_dir,
        )

        self.assertTrue(result)

        # Verify old content is replaced
        self.assertNotIn("old content", existing_command.read_text())
        self.assertFalse((existing_app / "old_file.txt").exists())

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    @mock.patch("hermes_cli.subcommands.gui_launcher.Path.home")
    def test_install_launcher_no_bundle_builds(self, mock_home, mock_system):
        """Test that install_launcher attempts to build when no bundle exists."""
        from hermes_cli.subcommands.gui_launcher import install_launcher

        mock_system.return_value = "Darwin"
        mock_home.return_value = self.tmp_path

        # Remove the app bundle
        shutil.rmtree(self.app_bundle)

        # Create a fresh bundle after "build"
        def mock_run(*args, **kwargs):
            self.app_resources.mkdir(parents=True, exist_ok=True)
            self.app_macos.mkdir(parents=True, exist_ok=True)
            self.icon_path.write_bytes(b"mock icon data")
            (self.app_macos / "Hermes").write_text("#!/bin/bash\nexit 0")
            return mock.MagicMock(returncode=0)

        with mock.patch("hermes_cli.subcommands.gui_launcher.subprocess.run", side_effect=mock_run):
            cwd = self.tmp_path / "Projects" / "example"
            cwd.mkdir(parents=True, exist_ok=True)

            result = install_launcher(
                cwd=cwd,
                name="Test Launcher",
                hermes_agent_dir=self.hermes_agent_dir,
            )

            self.assertTrue(result)

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    def test_install_launcher_non_macos_fails(self, mock_system):
        """Test that install_launcher fails on non-macOS."""
        from hermes_cli.subcommands.gui_launcher import install_launcher

        mock_system.return_value = "Linux"

        result = install_launcher(
            cwd=Path.home(),
            name="Test Launcher",
            hermes_agent_dir=self.hermes_agent_dir,
        )

        self.assertFalse(result)

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    @mock.patch("hermes_cli.subcommands.gui_launcher.Path.home")
    def test_install_launcher_nonexistent_cwd_fails(self, mock_home, mock_system):
        """Test that install_launcher fails when cwd doesn't exist."""
        from hermes_cli.subcommands.gui_launcher import install_launcher

        mock_system.return_value = "Darwin"
        mock_home.return_value = self.tmp_path

        cwd = self.tmp_path / "nonexistent" / "path"

        result = install_launcher(
            cwd=cwd,
            name="Test Launcher",
            hermes_agent_dir=self.hermes_agent_dir,
        )

        self.assertFalse(result)


class TestUninstallLauncher(TestCase):
    """Test uninstall_launcher function."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_path = Path(self.tmp_dir)

        # Mock Desktop and Applications directories
        self.desktop_dir_mock = self.tmp_path / "Desktop"
        self.app_dir_mock = self.tmp_path / "Applications"
        self.desktop_dir_mock.mkdir(parents=True, exist_ok=True)
        self.app_dir_mock.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    @mock.patch("hermes_cli.subcommands.gui_launcher.Path.home")
    def test_uninstall_launcher_removes_files(self, mock_home, mock_system):
        """Test that uninstall_launcher removes launcher files."""
        from hermes_cli.subcommands.gui_launcher import uninstall_launcher

        mock_system.return_value = "Darwin"
        mock_home.return_value = self.tmp_path

        # Create launcher files
        command_file = self.desktop_dir_mock / "Test Launcher.command"
        command_file.write_text("#!/bin/bash\necho test")

        app_bundle = self.app_dir_mock / "Test Launcher.app"
        app_contents = app_bundle / "Contents"
        app_contents.mkdir(parents=True, exist_ok=True)
        (app_contents / "Info.plist").write_text("<plist></plist>")

        result = uninstall_launcher(name="Test Launcher")

        self.assertTrue(result)
        self.assertFalse(command_file.exists())
        self.assertFalse(app_bundle.exists())

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    @mock.patch("hermes_cli.subcommands.gui_launcher.Path.home")
    def test_uninstall_launcher_no_files(self, mock_home, mock_system):
        """Test that uninstall_launcher handles missing files gracefully."""
        from hermes_cli.subcommands.gui_launcher import uninstall_launcher

        mock_system.return_value = "Darwin"
        mock_home.return_value = self.tmp_path

        result = uninstall_launcher(name="Nonexistent Launcher")

        # Should return False when no files found
        self.assertFalse(result)

    @mock.patch("hermes_cli.subcommands.gui_launcher.platform.system")
    def test_uninstall_launcher_non_macos_fails(self, mock_system):
        """Test that uninstall_launcher fails on non-macOS."""
        from hermes_cli.subcommands.gui_launcher import uninstall_launcher

        mock_system.return_value = "Linux"

        result = uninstall_launcher(name="Test Launcher")

        self.assertFalse(result)


class TestHelperFunctions(TestCase):
    """Test helper functions."""

    def test_generate_info_plist(self):
        """Test _generate_info_plist returns valid plist."""
        from hermes_cli.subcommands.gui_launcher import _generate_info_plist

        plist = _generate_info_plist("Test App")

        self.assertIn("CFBundleDisplayName", plist)
        self.assertIn("Test App", plist)
        self.assertIn("com.nousresearch.hermes-desktop-launcher", plist)
        self.assertIn("APPL", plist)

    def test_generate_launcher_script(self):
        """Test _generate_launcher_script returns valid script."""
        from hermes_cli.subcommands.gui_launcher import _generate_launcher_script

        cwd = Path("/Users/test/Projects")
        hermes_dir = Path("/Users/test/.hermes/hermes-agent")

        script = _generate_launcher_script(cwd, hermes_dir)

        self.assertIn("#!/usr/bin/env bash", script)
        self.assertIn(str(cwd), script)
        self.assertIn(str(hermes_dir), script)
        self.assertIn("hermes desktop --skip-build", script)
        self.assertIn("pgrep", script)

    def test_generate_app_launcher_script(self):
        """Test _generate_app_launcher_script returns valid script."""
        from hermes_cli.subcommands.gui_launcher import _generate_app_launcher_script

        cwd = Path("/Users/test/Projects")
        hermes_dir = Path("/Users/test/.hermes/hermes-agent")

        script = _generate_app_launcher_script(cwd, hermes_dir)

        self.assertIn("#!/usr/bin/env bash", script)
        self.assertIn(str(cwd), script)
        self.assertIn(str(hermes_dir), script)
        self.assertIn("nohup hermes desktop --skip-build", script)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

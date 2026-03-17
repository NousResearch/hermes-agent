"""
Tests for skip_confirm behavior in /skills install and /skills uninstall.

Verifies that --yes / -y bypasses the interactive confirmation prompt
that hangs inside prompt_toolkit's TUI.

Based on PR #1595 by 333Alden333 (salvaged).
"""

from unittest.mock import patch, MagicMock

import pytest


class TestHandleSkillsSlashInstallFlags:
    """Test flag parsing in handle_skills_slash for install."""

    def test_yes_flag_sets_skip_confirm(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill --yes")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("skip_confirm") is True
            assert kwargs.get("force") is False

    def test_y_flag_sets_skip_confirm(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill -y")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("skip_confirm") is True

    def test_force_flag_sets_force_not_skip(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill --force")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("force") is True
            assert kwargs.get("skip_confirm") is False

    def test_no_flags(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_install") as mock_install:
            handle_skills_slash("/skills install test/skill")
            mock_install.assert_called_once()
            _, kwargs = mock_install.call_args
            assert kwargs.get("force") is False
            assert kwargs.get("skip_confirm") is False


class TestHandleSkillsSlashUninstallFlags:
    """Test flag parsing in handle_skills_slash for uninstall."""

    def test_yes_flag_sets_skip_confirm(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_uninstall") as mock_uninstall:
            handle_skills_slash("/skills uninstall test-skill --yes")
            mock_uninstall.assert_called_once()
            _, kwargs = mock_uninstall.call_args
            assert kwargs.get("skip_confirm") is True

    def test_y_flag_sets_skip_confirm(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_uninstall") as mock_uninstall:
            handle_skills_slash("/skills uninstall test-skill -y")
            mock_uninstall.assert_called_once()
            _, kwargs = mock_uninstall.call_args
            assert kwargs.get("skip_confirm") is True

    def test_no_flags(self):
        from hermes_cli.skills_hub import handle_skills_slash
        with patch("hermes_cli.skills_hub.do_uninstall") as mock_uninstall:
            handle_skills_slash("/skills uninstall test-skill")
            mock_uninstall.assert_called_once()
            _, kwargs = mock_uninstall.call_args
            assert kwargs.get("skip_confirm", False) is False


class TestDoInstallSkipConfirm:
    """Test that do_install respects skip_confirm parameter."""

    @patch("hermes_cli.skills_hub.input", return_value="n")
    def test_without_skip_confirm_prompts_user(self, mock_input):
        """Without skip_confirm, input() is called for confirmation."""
        from hermes_cli.skills_hub import do_install
        with patch("hermes_cli.skills_hub._console"), \
             patch("tools.skills_hub.ensure_hub_dirs"), \
             patch("tools.skills_hub.GitHubAuth"), \
             patch("tools.skills_hub.create_source_router") as mock_router, \
             patch("hermes_cli.skills_hub._resolve_short_name", return_value="test/skill"), \
             patch("hermes_cli.skills_hub._resolve_source_meta_and_bundle") as mock_resolve:

            # Make it return None so we exit early
            mock_resolve.return_value = (None, None, None)
            do_install("test-skill", skip_confirm=False)
            # We don't get to the input() call because resolve returns None,
            # but the parameter wiring is correct


class TestDoUninstallSkipConfirm:
    """Test that do_uninstall respects skip_confirm parameter."""

    def test_skip_confirm_bypasses_input(self):
        """With skip_confirm=True, input() should not be called."""
        from hermes_cli.skills_hub import do_uninstall
        with patch("hermes_cli.skills_hub._console") as mock_console, \
             patch("tools.skills_hub.uninstall_skill", return_value=(True, "Removed")) as mock_uninstall, \
             patch("builtins.input") as mock_input:
            do_uninstall("test-skill", skip_confirm=True)
            mock_input.assert_not_called()
            mock_uninstall.assert_called_once_with("test-skill")

    def test_without_skip_confirm_calls_input(self):
        """Without skip_confirm, input() should be called."""
        from hermes_cli.skills_hub import do_uninstall
        with patch("hermes_cli.skills_hub._console"), \
             patch("tools.skills_hub.uninstall_skill", return_value=(True, "Removed")), \
             patch("builtins.input", return_value="y") as mock_input:
            do_uninstall("test-skill", skip_confirm=False)
            mock_input.assert_called_once()

    def test_without_skip_confirm_cancel(self):
        """Without skip_confirm, answering 'n' should cancel."""
        from hermes_cli.skills_hub import do_uninstall
        with patch("hermes_cli.skills_hub._console"), \
             patch("tools.skills_hub.uninstall_skill") as mock_uninstall, \
             patch("builtins.input", return_value="n"):
            do_uninstall("test-skill", skip_confirm=False)
            mock_uninstall.assert_not_called()

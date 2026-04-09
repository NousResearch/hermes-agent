"""Tests for non-interactive setup and first-run headless behavior."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest


def _make_setup_args(**overrides):
    return Namespace(
        non_interactive=overrides.get("non_interactive", False),
        section=overrides.get("section", None),
        reset=overrides.get("reset", False),
    )


def _make_chat_args(**overrides):
    return Namespace(
        continue_last=overrides.get("continue_last", None),
        resume=overrides.get("resume", None),
        model=overrides.get("model", None),
        provider=overrides.get("provider", None),
        toolsets=overrides.get("toolsets", None),
        verbose=overrides.get("verbose", False),
        query=overrides.get("query", None),
        worktree=overrides.get("worktree", False),
        yolo=overrides.get("yolo", False),
        pass_session_id=overrides.get("pass_session_id", False),
        quiet=overrides.get("quiet", False),
        checkpoints=overrides.get("checkpoints", False),
    )


class TestNonInteractiveSetup:
    """Verify setup paths exit cleanly in headless/non-interactive environments."""

    def test_non_interactive_flag_skips_wizard(self, capsys):
        """--non-interactive should print guidance and not enter the wizard."""
        from hermes_cli.setup import run_setup_wizard

        args = _make_setup_args(non_interactive=True)

        with (
            patch("hermes_cli.setup.ensure_hermes_home"),
            patch("hermes_cli.setup.load_config", return_value={}),
            patch("hermes_cli.setup.get_hermes_home", return_value="/tmp/.hermes"),
            patch("hermes_cli.auth.get_active_provider", side_effect=AssertionError("wizard continued")),
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            run_setup_wizard(args)

        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out

    def test_no_tty_skips_wizard(self, capsys):
        """When stdin has no TTY, the setup wizard should print guidance and return."""
        from hermes_cli.setup import run_setup_wizard

        args = _make_setup_args(non_interactive=False)

        with (
            patch("hermes_cli.setup.ensure_hermes_home"),
            patch("hermes_cli.setup.load_config", return_value={}),
            patch("hermes_cli.setup.get_hermes_home", return_value="/tmp/.hermes"),
            patch("hermes_cli.auth.get_active_provider", side_effect=AssertionError("wizard continued")),
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            mock_stdin.isatty.return_value = False
            run_setup_wizard(args)

        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out

    def test_chat_first_run_headless_skips_setup_prompt(self, capsys):
        """Bare `hermes` should not prompt for input when no provider exists and stdin is headless."""
        from hermes_cli.main import cmd_chat

        args = _make_chat_args()

        with (
            patch("hermes_cli.main._has_any_provider_configured", return_value=False),
            patch("hermes_cli.main.cmd_setup") as mock_setup,
            patch("sys.stdin") as mock_stdin,
            patch("builtins.input", side_effect=AssertionError("input should not be called")),
        ):
            mock_stdin.isatty.return_value = False
            with pytest.raises(SystemExit) as exc:
                cmd_chat(args)

        assert exc.value.code == 1
        mock_setup.assert_not_called()
        out = capsys.readouterr().out
        assert "hermes config set model.provider custom" in out

    def test_returning_user_terminal_menu_choice_dispatches_terminal_section(self, tmp_path):
        """Returning-user menu should map Terminal Backend to the terminal setup, not TTS."""
        from hermes_cli import setup as setup_mod

        args = _make_setup_args()
        config = {}
        model_section = MagicMock()
        tts_section = MagicMock()
        terminal_section = MagicMock()
        gateway_section = MagicMock()
        tools_section = MagicMock()
        agent_section = MagicMock()

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value=config),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch.object(
                setup_mod,
                "get_env_value",
                side_effect=lambda key: "sk-test" if key == "OPENROUTER_API_KEY" else "",
            ),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch.object(setup_mod, "prompt_choice", return_value=4),
            patch.object(
                setup_mod,
                "SETUP_SECTIONS",
                [
                    ("model", "Model & Provider", model_section),
                    ("tts", "Text-to-Speech", tts_section),
                    ("terminal", "Terminal Backend", terminal_section),
                    ("gateway", "Messaging Platforms (Gateway)", gateway_section),
                    ("tools", "Tools", tools_section),
                    ("agent", "Agent Settings", agent_section),
                ],
            ),
            patch.object(setup_mod, "save_config"),
            patch.object(setup_mod, "_print_setup_summary"),
        ):
            setup_mod.run_setup_wizard(args)

        terminal_section.assert_called_once_with(config)
        tts_section.assert_not_called()

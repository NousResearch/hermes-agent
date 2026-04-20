"""Tests for non-interactive setup and first-run headless behavior."""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest
from hermes_cli.config import DEFAULT_CONFIG, load_config, save_config


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

    def test_cmd_setup_allows_noninteractive_flag_without_tty(self):
        """The CLI entrypoint should not block --non-interactive before setup.py handles it."""
        from hermes_cli.main import cmd_setup

        args = _make_setup_args(non_interactive=True)

        with (
            patch("hermes_cli.setup.run_setup_wizard") as mock_run_setup,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            cmd_setup(args)

        mock_run_setup.assert_called_once_with(args)

    def test_cmd_setup_defers_no_tty_handling_to_setup_wizard(self):
        """Bare `hermes setup` should reach setup.py, which prints headless guidance."""
        from hermes_cli.main import cmd_setup

        args = _make_setup_args(non_interactive=False)

        with (
            patch("hermes_cli.setup.run_setup_wizard") as mock_run_setup,
            patch("sys.stdin") as mock_stdin,
        ):
            mock_stdin.isatty.return_value = False
            cmd_setup(args)

        mock_run_setup.assert_called_once_with(args)

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

    def test_reset_flag_rewrites_config_before_noninteractive_exit(self, tmp_path, monkeypatch, capsys):
        """--reset should rewrite config.yaml even when the wizard cannot run interactively."""
        from hermes_cli.setup import run_setup_wizard

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = load_config()
        cfg["model"] = {"provider": "custom", "base_url": "http://localhost:8080/v1", "default": "llama3"}
        cfg["agent"]["max_turns"] = 12
        save_config(cfg)

        args = _make_setup_args(non_interactive=True, reset=True)

        run_setup_wizard(args)

        reloaded = load_config()
        assert reloaded["model"] == DEFAULT_CONFIG["model"]
        assert reloaded["agent"]["max_turns"] == DEFAULT_CONFIG["agent"]["max_turns"]
        out = capsys.readouterr().out
        assert "Configuration reset to defaults." in out

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
            patch.object(setup_mod, "prompt_choice", return_value=3),
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

    def test_returning_user_menu_does_not_show_separator_rows(self, tmp_path):
        """Returning-user menu should only show selectable actions."""
        from hermes_cli import setup as setup_mod

        args = _make_setup_args()
        captured = {}

        def fake_prompt_choice(question, choices, default=0):
            captured["question"] = question
            captured["choices"] = list(choices)
            return len(choices) - 1

        with (
            patch.object(setup_mod, "ensure_hermes_home"),
            patch.object(setup_mod, "load_config", return_value={}),
            patch.object(setup_mod, "get_hermes_home", return_value=tmp_path),
            patch.object(setup_mod, "is_interactive_stdin", return_value=True),
            patch.object(
                setup_mod,
                "get_env_value",
                side_effect=lambda key: "sk-test" if key == "OPENROUTER_API_KEY" else "",
            ),
            patch("hermes_cli.auth.get_active_provider", return_value=None),
            patch.object(setup_mod, "prompt_choice", side_effect=fake_prompt_choice),
        ):
            setup_mod.run_setup_wizard(args)

        assert captured["question"] == "What would you like to do?"
        assert "---" not in captured["choices"]
        assert captured["choices"] == [
            "Quick Setup - configure missing items only",
            "Full Setup - reconfigure everything",
            "Model & Provider",
            "Terminal Backend",
            "Messaging Platforms (Gateway)",
            "Tools",
            "Agent Settings",
            "Exit",
        ]

    def test_main_accepts_tts_setup_section(self, monkeypatch):
        """`hermes setup tts` should parse and dispatch like other setup sections."""
        from hermes_cli import main as main_mod

        received = {}

        def fake_cmd_setup(args):
            received["section"] = args.section

        monkeypatch.setattr(main_mod, "cmd_setup", fake_cmd_setup)
        monkeypatch.setattr("sys.argv", ["hermes", "setup", "tts"])

        main_mod.main()

        assert received["section"] == "tts"

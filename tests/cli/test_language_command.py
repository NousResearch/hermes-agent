"""Tests for the /language CLI command and language config persistence."""

import unittest
from unittest.mock import MagicMock, patch

import cli
from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {"display": {"language": "en"}}
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = None
    cli_obj._pending_input = MagicMock()
    return cli_obj


class TestLanguageDispatch(unittest.TestCase):
    def test_language_dispatches_to_handler(self):
        cli_obj = _make_cli()
        with patch.object(cli_obj, "_handle_language_command") as mock_handler:
            result = cli_obj.process_command("/language zh")
        mock_handler.assert_called_once_with("/language zh")
        self.assertTrue(result)


class TestHandleLanguageCommand(unittest.TestCase):
    def test_no_arg_shows_current_and_supported(self):
        cli_obj = _make_cli()
        with (
            patch.object(cli, "_cprint") as mock_cprint,
            patch.object(cli, "save_config_value") as mock_save,
        ):
            cli_obj._handle_language_command("/language")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("en", printed)

    def test_status_arg_shows_current_and_supported(self):
        cli_obj = _make_cli()
        with (
            patch.object(cli, "_cprint") as mock_cprint,
            patch.object(cli, "save_config_value") as mock_save,
        ):
            cli_obj._handle_language_command("/language status")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("en", printed)

    def test_valid_code_persists_and_calls_reset_cache(self):
        cli_obj = _make_cli()
        events = []

        def _save(key, value):
            events.append(("save", key, value))
            return True

        def _reset():
            events.append(("reset",))

        with (
            patch.object(cli, "_cprint"),
            patch.object(cli, "save_config_value", side_effect=_save),
            patch("agent.i18n.reset_language_cache", side_effect=_reset),
        ):
            cli_obj._handle_language_command("/language zh")

        # persist BEFORE cache reset, so the reset re-reads the freshly saved config
        self.assertEqual([("save", "display.language", "zh"), ("reset",)], events)
        self.assertEqual(cli_obj.config["display"]["language"], "zh")

    def test_regional_tag_resolves_to_canonical(self):
        cli_obj = _make_cli()
        with (
            patch.object(cli, "_cprint"),
            patch.object(cli, "save_config_value", return_value=True) as mock_save,
        ):
            with patch("agent.i18n.reset_language_cache") as mock_reset:
                cli_obj._handle_language_command("/language zh-CN")

        mock_save.assert_called_once_with("display.language", "zh")
        mock_reset.assert_called_once()
        self.assertEqual(cli_obj.config["display"]["language"], "zh")

    def test_alias_chinese_resolves_to_zh(self):
        cli_obj = _make_cli()
        with (
            patch.object(cli, "_cprint"),
            patch.object(cli, "save_config_value", return_value=True) as mock_save,
        ):
            with patch("agent.i18n.reset_language_cache") as mock_reset:
                cli_obj._handle_language_command("/language chinese")

        mock_save.assert_called_once_with("display.language", "zh")
        mock_reset.assert_called_once()
        self.assertEqual(cli_obj.config["display"]["language"], "zh")

    def test_hermes_language_env_override_prints_note(self):
        cli_obj = _make_cli()
        with (
            patch.object(cli, "_cprint") as mock_cprint,
            patch.object(cli, "save_config_value", return_value=True),
            patch.dict("os.environ", {"HERMES_LANGUAGE": "ja"}),
            patch("agent.i18n.reset_language_cache"),
        ):
            cli_obj._handle_language_command("/language zh")

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("HERMES_LANGUAGE", printed)

    def test_no_env_override_prints_no_note(self):
        cli_obj = _make_cli()
        with (
            patch.object(cli, "_cprint") as mock_cprint,
            patch.object(cli, "save_config_value", return_value=True),
            patch.dict("os.environ", {}, clear=True),
            patch("agent.i18n.reset_language_cache"),
        ):
            cli_obj._handle_language_command("/language zh")

        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertNotIn("HERMES_LANGUAGE", printed)

    def test_invalid_code_prints_error_and_does_not_save(self):
        cli_obj = _make_cli()
        with (
            patch.object(cli, "_cprint") as mock_cprint,
            patch.object(cli, "save_config_value") as mock_save,
        ):
            cli_obj._handle_language_command("/language klingon")

        mock_save.assert_not_called()
        printed = " ".join(str(c) for c in mock_cprint.call_args_list)
        self.assertIn("Usage: /language", printed)


class TestLanguageRegistry(unittest.TestCase):
    def test_language_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        names = [c.name for c in COMMAND_REGISTRY]
        self.assertIn("language", names)

    def test_language_subcommands_match_handler(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        from agent.i18n import SUPPORTED_LANGUAGES

        language = next(c for c in COMMAND_REGISTRY if c.name == "language")
        self.assertEqual(language.category, "Configuration")
        self.assertEqual(set(language.subcommands), set(SUPPORTED_LANGUAGES) | {"status"})


if __name__ == "__main__":
    unittest.main()

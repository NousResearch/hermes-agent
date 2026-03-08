"""Regression tests for the `/model` slash command in the interactive CLI."""

from unittest.mock import patch

from cli import HermesCLI


class TestModelCommand:
    def _make_cli(self):
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.model = "anthropic/claude-opus-4.6"
        cli_obj.agent = object()
        cli_obj.provider = "openrouter"
        cli_obj.requested_provider = "openrouter"
        cli_obj.base_url = "https://openrouter.ai/api/v1"
        cli_obj.api_key = "test-key"
        cli_obj._explicit_api_key = None
        cli_obj._explicit_base_url = None
        return cli_obj

    def test_valid_model_from_api_saved_to_config(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.fetch_api_models",
                   return_value=["anthropic/claude-sonnet-4.5", "openai/gpt-5.4"]), \
             patch("cli.save_config_value", return_value=True) as save_mock:
            cli_obj.process_command("/model anthropic/claude-sonnet-4.5")

        output = capsys.readouterr().out
        assert "saved to config" in output
        assert cli_obj.model == "anthropic/claude-sonnet-4.5"
        assert cli_obj.agent is None
        save_mock.assert_called_once_with("model.default", "anthropic/claude-sonnet-4.5")

    def test_invalid_model_from_api_is_rejected(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.fetch_api_models",
                   return_value=["anthropic/claude-opus-4.6"]), \
             patch("cli.save_config_value") as save_mock:
            cli_obj.process_command("/model anthropic/fake-model")

        output = capsys.readouterr().out
        assert "not a valid model" in output
        assert cli_obj.model == "anthropic/claude-opus-4.6"  # unchanged
        assert cli_obj.agent is not None  # not reset
        save_mock.assert_not_called()

    def test_model_when_api_unreachable_falls_back_session_only(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch("cli.save_config_value") as save_mock:
            cli_obj.process_command("/model anthropic/claude-sonnet-next")

        output = capsys.readouterr().out
        assert "session only" in output
        assert cli_obj.model == "anthropic/claude-sonnet-next"
        assert cli_obj.agent is None
        save_mock.assert_not_called()

    def test_bad_format_rejected_without_api_call(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.fetch_api_models") as fetch_mock, \
             patch("cli.save_config_value") as save_mock:
            cli_obj.process_command("/model invalid-no-slash")

        output = capsys.readouterr().out
        assert "provider/model" in output
        assert cli_obj.model == "anthropic/claude-opus-4.6"  # unchanged
        fetch_mock.assert_not_called()  # no API call for format errors
        save_mock.assert_not_called()

    def test_validation_crash_falls_back_to_save(self, capsys):
        """If validate_requested_model throws, /model should still work (old behavior)."""
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.validate_requested_model",
                   side_effect=RuntimeError("boom")), \
             patch("cli.save_config_value", return_value=True) as save_mock:
            cli_obj.process_command("/model anthropic/claude-sonnet-4.5")

        output = capsys.readouterr().out
        assert "saved to config" in output
        assert cli_obj.model == "anthropic/claude-sonnet-4.5"
        save_mock.assert_called_once()

    def test_show_model_when_no_argument(self, capsys):
        cli_obj = self._make_cli()
        cli_obj.process_command("/model")

        output = capsys.readouterr().out
        assert "anthropic/claude-opus-4.6" in output
        assert "Usage" in output

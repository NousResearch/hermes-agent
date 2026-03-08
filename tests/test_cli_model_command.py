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
        cli_obj._explicit_api_key = None
        cli_obj._explicit_base_url = None
        return cli_obj

    def test_invalid_model_does_not_change_current_model(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": False,
                 "persist": False,
                 "recognized": False,
                 "message": "OpenRouter model IDs should use the `provider/model` format.",
             }), \
             patch("cli.save_config_value") as save_mock:
            cli_obj.process_command("/model invalid-model")

        output = capsys.readouterr().out
        assert "Current model unchanged" in output
        assert cli_obj.model == "anthropic/claude-opus-4.6"
        assert cli_obj.agent is not None
        save_mock.assert_not_called()

    def test_unknown_model_stays_session_only(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": True,
                 "persist": False,
                 "recognized": False,
                 "message": "Using it for this session only; config unchanged.",
             }), \
             patch("cli.save_config_value") as save_mock:
            cli_obj.process_command("/model anthropic/claude-sonnet-next")

        output = capsys.readouterr().out
        assert "session only" in output
        assert cli_obj.model == "anthropic/claude-sonnet-next"
        assert cli_obj.agent is None
        save_mock.assert_not_called()

    def test_known_model_is_saved_to_config(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": True,
                 "persist": True,
                 "recognized": True,
                 "message": None,
             }), \
             patch("cli.save_config_value", return_value=True) as save_mock:
            cli_obj.process_command("/model anthropic/claude-sonnet-4.5")

        output = capsys.readouterr().out
        assert "saved to config" in output
        assert cli_obj.model == "anthropic/claude-sonnet-4.5"
        assert cli_obj.agent is None
        save_mock.assert_called_once_with("model.default", "anthropic/claude-sonnet-4.5")

    def test_validation_crash_falls_back_to_save(self, capsys):
        """If validate_requested_model throws, /model should still work (old behavior)."""
        cli_obj = self._make_cli()

        with patch("hermes_cli.auth.resolve_provider", return_value="openrouter"), \
             patch("hermes_cli.models.validate_requested_model", side_effect=RuntimeError("boom")), \
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

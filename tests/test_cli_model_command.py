"""Regression tests for the `/model` slash command in the interactive CLI."""

from unittest.mock import patch, MagicMock

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

        with patch("hermes_cli.models.fetch_api_models",
                   return_value=["anthropic/claude-sonnet-4.5", "openai/gpt-5.4"]), \
             patch("cli.save_config_value", return_value=True) as save_mock:
            cli_obj.process_command("/model anthropic/claude-sonnet-4.5")

        output = capsys.readouterr().out
        assert "saved to config" in output
        assert cli_obj.model == "anthropic/claude-sonnet-4.5"
        save_mock.assert_called_once_with("model.default", "anthropic/claude-sonnet-4.5")

    def test_invalid_model_from_api_is_rejected(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.models.fetch_api_models",
                   return_value=["anthropic/claude-opus-4.6"]), \
             patch("cli.save_config_value") as save_mock:
            cli_obj.process_command("/model anthropic/fake-model")

        output = capsys.readouterr().out
        assert "not a valid model" in output
        assert "Model unchanged" in output
        assert cli_obj.model == "anthropic/claude-opus-4.6"
        save_mock.assert_not_called()

    def test_api_unreachable_falls_back_session_only(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch("cli.save_config_value") as save_mock:
            cli_obj.process_command("/model anthropic/claude-sonnet-next")

        output = capsys.readouterr().out
        assert "session only" in output
        assert "will revert on restart" in output
        assert cli_obj.model == "anthropic/claude-sonnet-next"
        save_mock.assert_not_called()

    def test_no_slash_model_probes_api_and_rejects(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.models.fetch_api_models",
                   return_value=["openai/gpt-5.4"]) as fetch_mock, \
             patch("cli.save_config_value") as save_mock:
            cli_obj.process_command("/model gpt-5.4")

        output = capsys.readouterr().out
        assert "not a valid model" in output
        assert "Model unchanged" in output
        assert cli_obj.model == "anthropic/claude-opus-4.6"  # unchanged
        assert cli_obj.agent is not None  # not reset
        save_mock.assert_not_called()

    def test_validation_crash_falls_back_to_save(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.models.validate_requested_model",
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
        assert "OpenRouter" in output
        assert "Available models" in output
        assert "provider:model-name" in output

    # -- provider switching tests -------------------------------------------

    def test_provider_colon_model_switches_provider(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "provider": "zai",
                 "api_key": "zai-key",
                 "base_url": "https://api.z.ai/api/paas/v4",
             }), \
             patch("hermes_cli.models.fetch_api_models",
                   return_value=["glm-5", "glm-4.7"]), \
             patch("cli.save_config_value", return_value=True) as save_mock:
            cli_obj.process_command("/model zai:glm-5")

        output = capsys.readouterr().out
        assert "glm-5" in output
        assert "provider:" in output.lower() or "Z.AI" in output
        assert cli_obj.model == "glm-5"
        assert cli_obj.provider == "zai"
        assert cli_obj.base_url == "https://api.z.ai/api/paas/v4"
        # Both model and provider should be saved
        assert save_mock.call_count == 2

    def test_provider_switch_fails_on_bad_credentials(self, capsys):
        cli_obj = self._make_cli()

        with patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   side_effect=Exception("No API key found")):
            cli_obj.process_command("/model nous:hermes-3")

        output = capsys.readouterr().out
        assert "Could not resolve credentials" in output
        assert cli_obj.model == "anthropic/claude-opus-4.6"  # unchanged
        assert cli_obj.provider == "openrouter"  # unchanged

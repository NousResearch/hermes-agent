"""Regression tests for #94881: Copilot per-model api_mode routing must win
over a persisted model.api_mode.

Copilot is a multi-endpoint provider (GPT-5+ -> /responses, Claude ->
/v1/messages, rest -> /chat/completions). A persisted api_mode was only
correct for the model selected when it was written; honouring it for a
different model sends Responses-only models to /chat/completions with a
misleading HTTP 400.
"""

from unittest.mock import patch


def _cfg(api_mode="chat_completions", default="gpt-5.4"):
    return {"provider": "copilot", "api_mode": api_mode, "default": default}


class TestCopilotPerModelRoutingWins:
    def test_persisted_chat_completions_ignored_for_responses_model(self):
        """A persisted chat_completions must NOT pin a Responses-only model."""
        from hermes_cli.runtime_provider import _copilot_runtime_api_mode

        with patch("hermes_cli.models.copilot_model_api_mode",
                   return_value="codex_responses"):
            mode = _copilot_runtime_api_mode(_cfg(), "key", target_model="gpt-5.6-sol")
        assert mode == "codex_responses"

    def test_persisted_mode_ignored_for_claude(self):
        from hermes_cli.runtime_provider import _copilot_runtime_api_mode

        with patch("hermes_cli.models.copilot_model_api_mode",
                   return_value="anthropic_messages"):
            mode = _copilot_runtime_api_mode(_cfg(), "key", target_model="claude-sonnet-4")
        assert mode == "anthropic_messages"

    def test_falls_back_to_persisted_when_catalog_lookup_fails(self):
        """Model unresolvable -> persisted explicit mode still honoured."""
        from hermes_cli.runtime_provider import _copilot_runtime_api_mode

        with patch("hermes_cli.models.copilot_model_api_mode",
                   side_effect=RuntimeError("catalog unreachable")):
            mode = _copilot_runtime_api_mode(_cfg(), "key", target_model="mystery-model")
        assert mode == "chat_completions"

    def test_no_model_at_all_uses_persisted(self):
        from hermes_cli.runtime_provider import _copilot_runtime_api_mode

        cfg = {"provider": "copilot", "api_mode": "codex_responses", "default": ""}
        with patch("hermes_cli.models.copilot_model_api_mode") as m:
            mode = _copilot_runtime_api_mode(cfg, "key", target_model=None)
        m.assert_not_called()
        assert mode == "codex_responses"

    def test_no_model_no_persisted_defaults_to_chat(self):
        from hermes_cli.runtime_provider import _copilot_runtime_api_mode

        cfg = {"provider": "copilot", "default": ""}
        mode = _copilot_runtime_api_mode(cfg, "key", target_model=None)
        assert mode == "chat_completions"

    def test_empty_resolution_falls_through_to_persisted(self):
        """copilot_model_api_mode returning '' falls back to persisted mode."""
        from hermes_cli.runtime_provider import _copilot_runtime_api_mode

        with patch("hermes_cli.models.copilot_model_api_mode", return_value=""):
            mode = _copilot_runtime_api_mode(_cfg(), "key", target_model="gpt-5.6-sol")
        assert mode == "chat_completions"

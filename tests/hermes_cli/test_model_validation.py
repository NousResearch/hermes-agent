"""Tests for provider-aware `/model` validation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hermes_cli.models import validate_requested_model


class TestValidateRequestedModel:
    def test_known_openrouter_model_can_be_saved(self):
        result = validate_requested_model("anthropic/claude-opus-4.6", "openrouter")

        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_openrouter_requires_provider_model_format(self):
        result = validate_requested_model("claude-opus-4.6", "openrouter")

        assert result["accepted"] is False
        assert result["persist"] is False
        assert "provider/model" in result["message"]

    def test_unknown_codex_model_is_session_only(self):
        result = validate_requested_model("totally-made-up", "openai-codex")

        assert result["accepted"] is True
        assert result["persist"] is False
        assert "OpenAI Codex" in result["message"]

    def test_custom_endpoint_allows_plain_model_ids(self):
        result = validate_requested_model(
            "gpt-4",
            "openrouter",
            base_url="http://localhost:11434/v1",
        )

        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["message"] is None

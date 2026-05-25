"""Test for OpenAI encrypted_content support check.

Verifies that openai_supports_encrypted_content() correctly identifies
models that support the `reasoning.encrypted_content` include parameter
on the Responses API.
"""

import pytest

from agent.model_metadata import openai_supports_encrypted_content


class TestOpenAIEncryptedContentSupport:
    """Test openai_supports_encrypted_content function."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5.4",
            "gpt-5.5-preview",
            "o1",
            "o1-preview",
            "o1-mini",
            "o3",
            "o3-mini",
            "openai/gpt-5",
            "openai/o3",
            "openrouter/openai/gpt-5.4",
        ],
    )
    def test_supported_models(self, model: str) -> None:
        """Models that should support encrypted_content."""
        assert openai_supports_encrypted_content(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-08-06",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openrouter/openai/gpt-4o",
            "claude-sonnet-4.6",
            "grok-4.3",
            "",
            None,
        ],
    )
    def test_unsupported_models(self, model: str) -> None:
        """Models that should NOT support encrypted_content."""
        assert openai_supports_encrypted_content(model) is False

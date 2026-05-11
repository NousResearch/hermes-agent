"""Tests for the redaction module."""

import pytest

from gateway.daimon.redaction import redact_response


class TestRedactionPatterns:
    """Each key pattern is correctly caught."""

    def test_openai_project_key(self):
        text = "My key is sk-proj-abc123DEF456_ghi789-jklmnopqrst"
        result = redact_response(text)
        assert "[REDACTED_OPENAI_KEY]" in result
        assert "sk-proj-" not in result

    def test_openai_generic_key(self):
        text = "Key: sk-abcdefghijklmnopqrstuvwxyz"
        result = redact_response(text)
        assert "[REDACTED_OPENAI_KEY]" in result
        assert "sk-abcdef" not in result

    def test_anthropic_key(self):
        text = "Token: sk-ant-abcdef1234567890-ghijk"
        result = redact_response(text)
        assert "[REDACTED_ANTHROPIC_KEY]" in result
        assert "sk-ant-" not in result

    def test_github_personal_access_token(self):
        text = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_response(text)
        assert "[REDACTED_GITHUB_TOKEN]" in result
        assert "ghp_" not in result

    def test_github_oauth_token(self):
        text = "gho_abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_response(text)
        assert "[REDACTED_GITHUB_TOKEN]" in result
        assert "gho_" not in result

    def test_github_pat_token(self):
        text = "github_pat_abc123DEF456_ghijklmnop"
        result = redact_response(text)
        assert "[REDACTED_GITHUB_TOKEN]" in result
        assert "github_pat_" not in result

    def test_xai_key(self):
        text = "xai-abcdefghijklmnopqrstuvwxyz"
        result = redact_response(text)
        assert "[REDACTED_XAI_KEY]" in result
        assert "xai-" not in result

    def test_google_api_key(self):
        text = "AIzaSyB-abcDEF123456789_ghijklmnopqrst"
        result = redact_response(text)
        assert "[REDACTED_GOOGLE_KEY]" in result
        assert "AIza" not in result

    def test_aws_access_key(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        result = redact_response(text)
        assert "[REDACTED_AWS_KEY]" in result
        assert "AKIA" not in result

    def test_bot_token(self):
        token = "Bot " + "a" * 60
        text = f"Authorization: {token}"
        result = redact_response(text)
        assert "[REDACTED_BOT_TOKEN]" in result
        assert "Bot " + "a" * 10 not in result


class TestRedactionEdgeCases:
    """Edge cases and combined scenarios."""

    def test_normal_text_unchanged(self):
        text = "Hello, this is a normal response with no secrets."
        assert redact_response(text) == text

    def test_multiple_keys_in_one_response(self):
        text = (
            "OpenAI: sk-proj-abcdefghijklmnopqrstuvwx\n"
            "GitHub: ghp_abcdefghijklmnopqrstuvwxyz1234567890\n"
            "AWS: AKIAIOSFODNN7EXAMPLE\n"
        )
        result = redact_response(text)
        assert "[REDACTED_OPENAI_KEY]" in result
        assert "[REDACTED_GITHUB_TOKEN]" in result
        assert "[REDACTED_AWS_KEY]" in result
        assert "sk-proj-" not in result
        assert "ghp_" not in result
        assert "AKIA" not in result

    def test_sk_proj_not_eaten_by_generic_sk(self):
        """sk-proj- should be redacted as OPENAI_KEY, not by the generic sk- pattern."""
        text = "sk-proj-abcdefghijklmnopqrstuvwx"
        result = redact_response(text)
        # Should have exactly one redaction marker
        assert result.count("[REDACTED_OPENAI_KEY]") == 1

    def test_short_prefix_not_matched(self):
        """Short strings that happen to start with a prefix should not be redacted."""
        # sk- followed by less than 20 chars
        text = "sk-short"
        assert redact_response(text) == text

    def test_empty_string(self):
        assert redact_response("") == ""

    def test_partial_match_preserved(self):
        """Text containing 'sk-' in a normal context shouldn't be redacted."""
        text = "I used sk-learn for ML"  # sk-learn is < 20 chars after sk-
        assert redact_response(text) == text

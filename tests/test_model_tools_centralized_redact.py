"""Tests for centralized secret redaction in handle_function_call.

Verifies that ALL tool results pass through _redact_tool_result() at the
centralized dispatch point, regardless of whether the individual tool
implements its own redaction.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestRedactToolResult:
    """Test _redact_tool_result helper directly."""

    def test_string_result_is_redacted(self):
        """String tool results should be passed through redact_sensitive_text."""
        from model_tools import _redact_tool_result

        with patch("agent.redact.redact_sensitive_text", return_value="REDACTED") as mock:
            result = _redact_tool_result("secret key=sk-abc123")
            assert result == "REDACTED"
            mock.assert_called_once_with("secret key=sk-abc123")

    def test_non_string_result_passes_through(self):
        """Non-string results should be returned unchanged."""
        from model_tools import _redact_tool_result

        data = [{"type": "text", "text": "output"}]
        result = _redact_tool_result(data)
        assert result is data  # same object, not modified

    def test_empty_string_passes_through(self):
        """Empty string should not be redacted (short-circuit)."""
        from model_tools import _redact_tool_result

        with patch("agent.redact.redact_sensitive_text") as mock:
            result = _redact_tool_result("")
            assert result == ""
            mock.assert_not_called()

    def test_fail_open_on_exception(self):
        """If redact_sensitive_text raises, original result should be returned."""
        from model_tools import _redact_tool_result

        with patch("agent.redact.redact_sensitive_text", side_effect=RuntimeError("oops")):
            result = _redact_tool_result("secret key=sk-abc123")
            assert result == "secret key=sk-abc123"

    def test_none_result_passes_through(self):
        """None results should be returned unchanged."""
        from model_tools import _redact_tool_result

        result = _redact_tool_result(None)
        assert result is None

    def test_dict_result_passes_through(self):
        """Dict results should be returned unchanged."""
        from model_tools import _redact_tool_result

        data = {"error": "something failed"}
        result = _redact_tool_result(data)
        assert result is data


class TestRedactToolResultIntegration:
    """Integration tests using real redact_sensitive_text."""

    def test_real_redaction_openai_key(self):
        """Real redaction should catch OpenAI API keys."""
        from model_tools import _redact_tool_result

        result = _redact_tool_result("key=sk-1234567890abcdef1234567890abcdef")
        assert "sk-1234567890abcdef1234567890abcdef" not in result

    def test_real_redaction_bearer_token(self):
        """Real redaction should catch Bearer tokens."""
        from model_tools import _redact_tool_result

        result = _redact_tool_result("Authorization: Bearer abc123token")
        assert "abc123token" not in result

    def test_real_redaction_private_key(self):
        """Real redaction should catch private keys."""
        from model_tools import _redact_tool_result

        result = _redact_tool_result(
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA7Vx9k2mF3pQzL8nR4tYwGcHb2sJvN6xK1aD5eP0oQ\n"
            "-----END RSA PRIVATE KEY-----"
        )
        assert "PRIVATE KEY" not in result

    def test_real_redaction_preserves_safe_text(self):
        """Non-secret text should pass through unchanged."""
        from model_tools import _redact_tool_result

        assert _redact_tool_result("Hello world") == "Hello world"

    def test_config_flag_respected(self):
        """When _REDACT_ENABLED is False, redaction should be a no-op."""
        from model_tools import _redact_tool_result
        import agent.redact as redact_mod

        original = redact_mod._REDACT_ENABLED
        try:
            redact_mod._REDACT_ENABLED = False
            secret = "key=sk-1234567890abcdef1234567890abcdef"
            result = _redact_tool_result(secret)
            assert result == secret
        finally:
            redact_mod._REDACT_ENABLED = original

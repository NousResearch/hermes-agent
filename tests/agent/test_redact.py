"""Tests for agent.redact -- secret masking in logs and output."""

import logging
import os

import pytest

from agent.redact import redact_sensitive_text, RedactingFormatter


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure HERMES_REDACT_SECRETS is not disabled by prior test imports."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    # Also patch the module-level snapshot so it reflects the cleared env var
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


class TestKnownPrefixes:
    def test_openai_sk_key(self):
        text = "Using key sk-proj-abc123def456ghi789jkl012"
        result = redact_sensitive_text(text)
        assert "sk-pro" in result
        assert "abc123def456" not in result
        assert "..." in result

    def test_openrouter_sk_key(self):
        text = "OPENROUTER_API_KEY=sk-or-v1-abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_sensitive_text(text)
        assert "abcdefghijklmnop" not in result

    def test_github_pat_classic(self):
        result = redact_sensitive_text("token: ghp_abc123def456ghi789jkl")
        assert "abc123def456" not in result

    def test_github_pat_fine_grained(self):
        result = redact_sensitive_text("github_pat_abc123def456ghi789jklmno")
        assert "abc123def456" not in result

    def test_slack_token(self):
        token = "xoxb-" + "0" * 12 + "-" + "a" * 14
        result = redact_sensitive_text(token)
        assert "a" * 14 not in result

    def test_google_api_key(self):
        result = redact_sensitive_text("AIzaSyB-abc123def456ghi789jklmno012345")
        assert "abc123def456" not in result

    def test_perplexity_key(self):
        result = redact_sensitive_text("pplx-abcdef123456789012345")
        assert "abcdef12345" not in result

    def test_fal_key(self):
        result = redact_sensitive_text("fal_abc123def456ghi789jkl")
        assert "abc123def456" not in result

    def test_short_token_fully_masked(self):
        result = redact_sensitive_text("key=sk-short1234567")
        assert "***" in result


class TestEnvAssignments:
    def test_export_api_key(self):
        text = "export OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012"
        result = redact_sensitive_text(text)
        assert "OPENAI_API_KEY=" in result
        assert "abc123def456" not in result

    def test_quoted_value(self):
        text = 'MY_SECRET_TOKEN="supersecretvalue123456789"'
        result = redact_sensitive_text(text)
        assert "MY_SECRET_TOKEN=" in result
        assert "supersecretvalue" not in result

    def test_non_secret_env_unchanged(self):
        text = "HOME=/home/user"
        result = redact_sensitive_text(text)
        assert result == text

    def test_path_unchanged(self):
        text = "PATH=/usr/local/bin:/usr/bin"
        result = redact_sensitive_text(text)
        assert result == text

    def test_lowercase_python_variable_token_unchanged(self):
        # Regression: #4367 — lowercase 'token' assignment must not be redacted
        text = "before_tokens = response.usage.prompt_tokens"
        result = redact_sensitive_text(text)
        assert result == text

    def test_lowercase_python_variable_api_key_unchanged(self):
        # Regression: #4367 — lowercase 'api_key' must not be redacted
        text = "api_key = config.get('api_key')"
        result = redact_sensitive_text(text)
        assert result == text

    def test_typescript_await_token_unchanged(self):
        # Regression: #4367 — 'await' keyword must not be redacted as a secret value
        text = "const token = await getToken();"
        result = redact_sensitive_text(text)
        assert result == text

    def test_typescript_await_secret_unchanged(self):
        # Regression: #4367 — similar pattern with 'secret' variable
        text = "const secret = await fetchSecret();"
        result = redact_sensitive_text(text)
        assert result == text

    def test_export_whitespace_preserved(self):
        # Regression: #4367 — whitespace before uppercase env var must be preserved
        text = "export SECRET_TOKEN=mypassword"
        result = redact_sensitive_text(text)
        assert result.startswith("export ")
        assert "SECRET_TOKEN=" in result
        assert "mypassword" not in result


class TestJsonFields:
    def test_json_api_key(self):
        text = '{"apiKey": "sk-proj-abc123def456ghi789jkl012"}'
        result = redact_sensitive_text(text)
        assert "abc123def456" not in result

    def test_json_token(self):
        text = '{"access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.longtoken.here"}'
        result = redact_sensitive_text(text)
        assert "eyJhbGciOiJSUzI1NiIs" not in result

    def test_json_non_secret_unchanged(self):
        text = '{"name": "John", "model": "gpt-4"}'
        result = redact_sensitive_text(text)
        assert result == text


class TestAuthHeaders:
    def test_bearer_token(self):
        text = "Authorization: Bearer sk-proj-abc123def456ghi789jkl012"
        result = redact_sensitive_text(text)
        assert "Authorization: Bearer" in result
        assert "abc123def456" not in result

    def test_case_insensitive(self):
        text = "authorization: bearer mytoken123456789012345678"
        result = redact_sensitive_text(text)
        assert "mytoken12345" not in result


class TestTelegramTokens:
    def test_bot_token(self):
        text = "bot123456789:ABCDEfghij-KLMNopqrst_UVWXyz12345"
        result = redact_sensitive_text(text)
        assert "ABCDEfghij" not in result
        assert "123456789:***" in result

    def test_raw_token(self):
        text = "12345678901:ABCDEfghijKLMNopqrstUVWXyz1234567890"
        result = redact_sensitive_text(text)
        assert "ABCDEfghij" not in result


class TestPassthrough:
    def test_empty_string(self):
        assert redact_sensitive_text("") == ""

    def test_none_returns_none(self):
        assert redact_sensitive_text(None) is None

    def test_non_string_input_int_coerced(self):
        assert redact_sensitive_text(12345) == "12345"

    def test_non_string_input_dict_coerced_and_redacted(self):
        result = redact_sensitive_text({"token": "sk-proj-abc123def456ghi789jkl012"})
        assert "abc123def456" not in result

    def test_normal_text_unchanged(self):
        text = "Hello world, this is a normal log message with no secrets."
        assert redact_sensitive_text(text) == text

    def test_code_unchanged(self):
        text = "def main():\n    print('hello')\n    return 42"
        assert redact_sensitive_text(text) == text

    def test_url_without_key_unchanged(self):
        text = "Connecting to https://api.openai.com/v1/chat/completions"
        assert redact_sensitive_text(text) == text


class TestRedactingFormatter:
    def test_formats_and_redacts(self):
        formatter = RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Key is sk-proj-abc123def456ghi789jkl012",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "abc123def456" not in result
        assert "sk-pro" in result


class TestPrintenvSimulation:
    """Simulate what happens when the agent runs `env` or `printenv`."""

    def test_full_env_dump(self):
        env_dump = """HOME=/home/user
PATH=/usr/local/bin:/usr/bin
OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012mno345
OPENROUTER_API_KEY=sk-or-v1-reallyLongSecretKeyValue12345678
FIRECRAWL_API_KEY=fc-shortkey123456789012
TELEGRAM_BOT_TOKEN=bot987654321:ABCDEfghij-KLMNopqrst_UVWXyz12345
SHELL=/bin/bash
USER=teknium"""
        result = redact_sensitive_text(env_dump)
        # Secrets should be masked
        assert "abc123def456" not in result
        assert "reallyLongSecretKey" not in result
        assert "ABCDEfghij" not in result
        # Non-secrets should survive
        assert "HOME=/home/user" in result
        assert "SHELL=/bin/bash" in result
        assert "USER=teknium" in result


class TestSecretCapturePayloadRedaction:
    def test_secret_value_field_redacted(self):
        text = '{"success": true, "secret_value": "sk-test-secret-1234567890"}'
        result = redact_sensitive_text(text)
        assert "sk-test-secret-1234567890" not in result

    def test_raw_secret_field_redacted(self):
        text = '{"raw_secret": "ghp_abc123def456ghi789jkl"}'
        result = redact_sensitive_text(text)
        assert "abc123def456" not in result


class TestElevenLabsTavilyExaKeys:
    """Regression tests for ElevenLabs (sk_), Tavily (tvly-), and Exa (exa_) keys."""

    def test_elevenlabs_key_redacted(self):
        text = "ELEVENLABS_API_KEY=sk_abc123def456ghi789jklmnopqrstu"
        result = redact_sensitive_text(text)
        assert "abc123def456ghi" not in result

    def test_elevenlabs_key_in_log_line(self):
        text = "Connecting to ElevenLabs with key sk_abc123def456ghi789jklmnopqrstu"
        result = redact_sensitive_text(text)
        assert "abc123def456ghi" not in result

    def test_tavily_key_redacted(self):
        text = "TAVILY_API_KEY=tvly-ABCdef123456789GHIJKL0000"
        result = redact_sensitive_text(text)
        assert "ABCdef123456789" not in result

    def test_tavily_key_in_log_line(self):
        text = "Initialising Tavily client with tvly-ABCdef123456789GHIJKL0000"
        result = redact_sensitive_text(text)
        assert "ABCdef123456789" not in result

    def test_exa_key_redacted(self):
        text = "EXA_API_KEY=exa_XYZ789abcdef000000000000000"
        result = redact_sensitive_text(text)
        assert "XYZ789abcdef" not in result

    def test_exa_key_in_log_line(self):
        text = "Using Exa client with key exa_XYZ789abcdef000000000000000"
        result = redact_sensitive_text(text)
        assert "XYZ789abcdef" not in result

    def test_all_three_in_env_dump(self):
        env_dump = (
            "HOME=/home/user\n"
            "ELEVENLABS_API_KEY=sk_abc123def456ghi789jklmnopqrstu\n"
            "TAVILY_API_KEY=tvly-ABCdef123456789GHIJKL0000\n"
            "EXA_API_KEY=exa_XYZ789abcdef000000000000000\n"
            "SHELL=/bin/bash\n"
        )
        result = redact_sensitive_text(env_dump)
        assert "abc123def456ghi" not in result
        assert "ABCdef123456789" not in result
        assert "XYZ789abcdef" not in result
        assert "HOME=/home/user" in result
        assert "SHELL=/bin/bash" in result


class TestJWTTokens:
    """JWT tokens start with eyJ (base64 for '{') and have dot-separated parts."""

    def test_full_3part_jwt(self):
        text = (
            "Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJpc3MiOiI0MjNiZDJkYjg4MjI0MDAwIn0"
            ".Gxgv0rru-_kS-I_60EJ7CENTnBh9UeuL3QhkMoQ-VnM"
        )
        result = redact_sensitive_text(text)
        assert "Token:" in result
        # Payload and signature must not survive
        assert "eyJpc3Mi" not in result
        assert "Gxgv0rru" not in result

    def test_2part_jwt(self):
        text = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        result = redact_sensitive_text(text)
        assert "eyJzdWIi" not in result

    def test_standalone_jwt_header(self):
        text = "leaked header: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 here"
        result = redact_sensitive_text(text)
        assert "IkpXVCJ9" not in result
        assert "leaked header:" in result

    def test_jwt_with_base64_padding(self):
        text = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0=.abc123def456ghij"
        result = redact_sensitive_text(text)
        assert "abc123def456" not in result

    def test_short_eyj_not_matched(self):
        """eyJ followed by fewer than 10 base64 chars should not match."""
        text = "eyJust a normal word"
        assert redact_sensitive_text(text) == text

    def test_jwt_preserves_surrounding_text(self):
        text = "before eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0 after"
        result = redact_sensitive_text(text)
        assert result.startswith("before ")
        assert result.endswith(" after")

    def test_home_assistant_jwt_in_memory(self):
        """Real-world pattern: HA token stored in agent memory block."""
        text = (
            "Home Assistant API Token: "
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJpc3MiOiJhYmNkZWYiLCJleHAiOjE3NzQ5NTcxMDN9"
            ".Gxgv0rru-_kS-I_60EJ7CENTnBh9UeuL3QhkMoQ-VnM"
        )
        result = redact_sensitive_text(text)
        assert "Home Assistant API Token:" in result
        assert "Gxgv0rru" not in result
        assert "..." in result


class TestDiscordMentions:
    """Discord snowflake IDs in <@ID> or <@!ID> format."""

    def test_normal_mention(self):
        result = redact_sensitive_text("Hello <@222589316709220353>")
        assert "222589316709220353" not in result
        assert "<@***>" in result

    def test_nickname_mention(self):
        result = redact_sensitive_text("Ping <@!1331549159177846844>")
        assert "1331549159177846844" not in result
        assert "<@!***>" in result

    def test_multiple_mentions(self):
        text = "<@111111111111111111> and <@222222222222222222>"
        result = redact_sensitive_text(text)
        assert "111111111111111111" not in result
        assert "222222222222222222" not in result

    def test_short_id_not_matched(self):
        """IDs shorter than 17 digits are not Discord snowflakes."""
        text = "<@12345>"
        assert redact_sensitive_text(text) == text

    def test_slack_mention_not_matched(self):
        """Slack mentions use letters, not pure digits."""
        text = "<@U024BE7LH>"
        assert redact_sensitive_text(text) == text

    def test_preserves_surrounding_text(self):
        text = "User <@222589316709220353> said hello"
        result = redact_sensitive_text(text)
        assert result.startswith("User ")
        assert result.endswith(" said hello")

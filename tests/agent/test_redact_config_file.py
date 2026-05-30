"""Tests for config_file parameter in redact_sensitive_text.

Issue #35519: redact_sensitive_text corrupts API keys in config files when
read via read_file/search_files, causing 401 errors.

The config_file parameter skips prefix-based masking so config files
(.yaml, .json, .env, .toml, .conf) return actual API key values to the agent.
"""

import pytest
from agent.redact import redact_sensitive_text


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    """Ensure HERMES_REDACT_SECRETS is not disabled by prior test imports."""
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


class TestConfigFileSkipsPrefixMasking:
    """config_file=True should skip prefix-based secret masking for config files."""

    def test_config_file_preserves_openai_key(self):
        """Config files should NOT mask sk-... prefix keys."""
        key = "sk-" + "proj-" + "a" * 40
        text = "api_key: " + key
        result = redact_sensitive_text(text, config_file=True)
        assert key in result, f"Key should be preserved but got: {result}"

    def test_config_file_preserves_github_pat(self):
        """Config files should NOT mask ghp_... tokens."""
        key = "ghp_" + "a" * 36
        text = "GITHUB_TOKEN: " + key
        result = redact_sensitive_text(text, config_file=True)
        assert key in result, f"PAT should be preserved but got: {result}"

    def test_config_file_preserves_xai_key(self):
        """Config files should NOT mask xai-... keys."""
        key = "xai-" + "a" * 40
        text = "api_key: " + key
        result = redact_sensitive_text(text, config_file=True)
        assert key in result, f"xai key should be preserved but got: {result}"

    def test_config_file_preserves_openrouter_key(self):
        """Config files should NOT mask sk-or-... keys."""
        key = "sk-or-" + "a" * 40
        text = "api_key: " + key
        result = redact_sensitive_text(text, config_file=True)
        assert key in result, f"OpenRouter key should be preserved but got: {result}"

    def test_config_file_preserves_perplexity_key(self):
        """Config files should NOT mask pplx-... keys."""
        key = "pplx-" + "a" * 40
        text = "api_key: " + key
        result = redact_sensitive_text(text, config_file=True)
        assert key in result, f"Perplexity key should be preserved but got: {result}"


class TestConfigFileImpliesCodeFile:
    """config_file=True should also skip ENV-assignment and JSON-field patterns."""

    def test_config_file_skips_env_pattern(self):
        """Config files should NOT mask ENV-style assignments."""
        key = "sk-" + "a" * 40
        text = "OPENAI_API_KEY=" + key
        result = redact_sensitive_text(text, config_file=True)
        assert key in result, f"ENV value should be preserved but got: {result}"

    def test_config_file_skips_json_field_pattern(self):
        """Config files should NOT mask JSON-style key fields."""
        key = "sk-" + "a" * 40
        text = '{"apiKey": "' + key + '"}'
        result = redact_sensitive_text(text, config_file=True)
        assert key in result, f"JSON value should be preserved but got: {result}"


class TestNonConfigFilesStillMasked:
    """Non-config files should still get prefix masking (existing behavior)."""

    def test_default_still_masks_prefix(self):
        """Default mode should still mask prefix keys."""
        key = "sk-" + "proj-" + "a" * 40
        text = "Using key " + key + " here"
        result = redact_sensitive_text(text)
        assert key not in result, f"Key should be masked but got: {result}"

    def test_code_file_still_masks_prefix(self):
        """code_file=True should still mask prefix keys."""
        key = "sk-" + "proj-" + "a" * 40
        text = "Using key " + key + " here"
        result = redact_sensitive_text(text, code_file=True)
        assert key not in result, f"Key should be masked but got: {result}"


class TestConfigFileStillMasksOtherPatterns:
    """config_file=True should still mask auth headers, private keys, DB connstrings."""

    def test_auth_header_still_masked(self):
        """Authorization headers should still be masked in config mode."""
        key = "sk-" + "a" * 40
        text = "Authorization: Bearer " + key
        result = redact_sensitive_text(text, config_file=True)
        assert key not in result, f"Auth header should still be masked but got: {result}"

    def test_db_connstring_still_masked(self):
        """DB connection string passwords should still be masked in config mode."""
        text = "postgres://user:secretpass@host:5432/db"
        result = redact_sensitive_text(text, config_file=True)
        assert "secretpass" not in result, f"DB password should be masked but got: {result}"

    def test_telegram_token_still_masked(self):
        """Telegram bot tokens should still be masked in config mode."""
        text = "bot_token: 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz012345"
        result = redact_sensitive_text(text, config_file=True)
        assert "ABCdefGHIjklMNOpqrsTUVwxyz012345" not in result, f"TG token should be masked but got: {result}"

    def test_private_key_still_masked(self):
        """Private key blocks should still be masked in config mode."""
        text = "-----BEGIN RSA PRIVATE KEY-----\ndata\n-----END RSA PRIVATE KEY-----"
        result = redact_sensitive_text(text, config_file=True)
        assert "REDACTED PRIVATE KEY" in result, f"Private key should be masked but got: {result}"

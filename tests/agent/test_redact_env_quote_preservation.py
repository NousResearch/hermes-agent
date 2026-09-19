"""Tests for env-assignment redaction quote preservation (#103894).

The fix changes _ENV_ASSIGN_RE and _ENV_ASSIGN_LOWER_RE from (\S+) to
([^'"\s]+) so that closing quotes in shell commands survive redaction.
"""

from __future__ import annotations

import pytest

from agent.redact import redact_sensitive_text


class TestEnvAssignmentQuotePreservation:
    """Regression tests: closing quotes must survive env-assignment redaction."""

    def test_printf_quote_preserved(self):
        """printf 'KEY=val\n' >> file → quote preserved, value masked."""
        text = "printf 'ANTHROPIC_API_KEY=sk-pro-abc123\n' >> ~/.hermes/.env"
        result = redact_sensitive_text(text)
        # The closing single-quote after the value must survive.
        # \n is part of the value (not whitespace/quote), so the regex
        # matches sk-pro-abc123\n and the closing ' survives after it.
        assert "' >> ~/.hermes/.env" in result
        # The secret value must be masked
        assert "sk-pro-abc123" not in result

    def test_grep_quote_preserved_empty_value(self):
        """grep -c '^KEY=' file → no secret, quotes intact."""
        text = "grep -c '^ANTHROPIC_API_KEY=' ~/.hermes/.env"
        result = redact_sensitive_text(text)
        # No value to mask, quotes must be intact
        assert "'^ANTHROPIC_API_KEY='" in result or "ANTHROPIC_API_KEY=" in result

    def test_export_quoted_value_preserves_quotes(self):
        """export KEY="secret" → value masked, double-quotes preserved."""
        text = 'export OPENAI_API_KEY="sk-pro-abc123def456"'
        result = redact_sensitive_text(text)
        assert '"' in result
        assert "sk-pro-abc123def456" not in result

    def test_export_single_quoted_value(self):
        """export KEY='secret' → value masked, single-quotes preserved."""
        text = "export ANTHROPIC_API_KEY='sk-ant-abc123def456'"
        result = redact_sensitive_text(text)
        assert "'" in result
        assert "sk-ant-abc123def456" not in result

    def test_unquoted_value_still_masked(self):
        """export KEY=secret → value masked (unquoted path unchanged)."""
        text = "export OPENAI_API_KEY=sk-pro-abc123"
        result = redact_sensitive_text(text)
        assert "sk-pro-abc123" not in result
        assert "OPENAI_API_KEY=" in result

    def test_lowercase_env_name_quoted(self):
        """openai_key='secret' in prose → masked, quotes preserved."""
        text = "set openai_key='sk-pro-abc123' in config"
        result = redact_sensitive_text(text)
        assert "'" in result
        assert "sk-pro-abc123" not in result

    def test_multiple_env_assignments_in_line(self):
        """Multiple KEY=val in one line: each masked independently."""
        text = "export OPENAI_API_KEY=sk-abc ANTHROPIC_API_KEY=sk-def"
        result = redact_sensitive_text(text)
        assert "sk-abc" not in result
        assert "sk-def" not in result

    def test_heredoc_quote_preserved(self):
        """cat <<'EOF' > file with KEY=val → closing delimiter intact."""
        text = "cat <<'EOF' > .env\nANTHROPIC_API_KEY=sk-ant-abc\nEOF"
        result = redact_sensitive_text(text)
        assert "sk-ant-abc" not in result
        # The heredoc marker must survive
        assert "EOF" in result

    def test_non_secret_env_unchanged(self):
        """HOME=/home/user must not be masked."""
        text = "HOME=/home/user"
        result = redact_sensitive_text(text)
        assert result == text

    def test_short_value_still_masked(self):
        """Short secret values must still be masked."""
        text = "export OPENAI_API_KEY=abc123"
        result = redact_sensitive_text(text)
        assert "abc123" not in result

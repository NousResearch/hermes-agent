"""
Security tests for credential pool operations.

Tests SSRF prevention, path traversal blocking, API key leak prevention,
and strategy injection blocking.
"""

import pytest
from hermes_cli.credential_security import (
    validate_base_url_safe,
    validate_provider_name,
    validate_pool_strategy,
    SUPPORTED_POOL_STRATEGIES,
)


# ━━━━ SSRF Prevention ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSSRFPrevention:
    """Verify that validate_base_url_safe blocks SSRF vectors."""

    def test_blocks_aws_metadata_ipv4(self):
        """AWS metadata endpoint 169.254.169.254 must be blocked."""
        with pytest.raises(ValueError, match="Blocked"):
            validate_base_url_safe("http://169.254.169.254/latest/meta-data/")

    def test_blocks_gcp_metadata(self):
        """GCP metadata endpoint metadata.google.internal must be blocked."""
        with pytest.raises(ValueError, match="Blocked"):
            validate_base_url_safe("http://metadata.google.internal/computeMetadata/")

    def test_blocks_link_local_prefix(self):
        """Any 169.254.x.x address must be blocked (link-local range)."""
        with pytest.raises(ValueError, match="Blocked"):
            validate_base_url_safe("http://169.254.1.1/v1/chat")

    def test_blocks_file_scheme(self):
        """file:// scheme must be blocked."""
        with pytest.raises(ValueError, match="Blocked scheme"):
            validate_base_url_safe("file:///etc/passwd")

    def test_blocks_gopher_scheme(self):
        """gopher:// scheme must be blocked."""
        with pytest.raises(ValueError, match="Blocked scheme"):
            validate_base_url_safe("gopher://attacker.com/x")

    def test_blocks_dict_scheme(self):
        """dict:// scheme must be blocked."""
        with pytest.raises(ValueError, match="Blocked scheme"):
            validate_base_url_safe("dict://attacker.com/x")

    def test_blocks_null_byte(self):
        """Null byte in URL must be blocked."""
        with pytest.raises(ValueError, match="Null byte"):
            validate_base_url_safe("https://api.z.ai\x00.evil.com/")

    def test_allows_valid_https(self):
        """Valid HTTPS URL must pass through unchanged."""
        url = "https://api.z.ai/api/coding/paas/v4"
        assert validate_base_url_safe(url) == url

    def test_allows_empty_url(self):
        """Empty URL must be returned as-is (caller decides fallback)."""
        assert validate_base_url_safe("") == ""
        assert validate_base_url_safe("   ") == "   "

    def test_allows_localhost(self):
        """Localhost must NOT be blocked (for local LLM servers like LM Studio)."""
        url = "http://localhost:1234/v1"
        assert validate_base_url_safe(url) == url

    def test_allows_127_loopback(self):
        """127.0.0.1 must NOT be blocked (for local LLM servers)."""
        url = "http://127.0.0.1:1234/v1"
        assert validate_base_url_safe(url) == url


# ━━━━ Path Traversal Prevention ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPathTraversal:
    """Verify that validate_provider_name blocks path traversal."""

    def test_blocks_dotdot(self):
        """../ in provider name must be blocked."""
        with pytest.raises(ValueError):
            validate_provider_name("../etc/passwd")

    def test_blocks_forward_slash(self):
        """Forward slash in provider name must be blocked."""
        with pytest.raises(ValueError):
            validate_provider_name("anthropic/../../../etc/passwd")

    def test_blocks_backslash(self):
        """Backslash in provider name must be blocked."""
        with pytest.raises(ValueError):
            validate_provider_name("zai\\..\\..")

    def test_blocks_null_byte(self):
        """Null byte in provider name must be blocked."""
        with pytest.raises(ValueError):
            validate_provider_name("zai\x00../../")

    def test_blocks_special_chars(self):
        """Shell metacharacters must be blocked."""
        for c in [";", "|", "$", "!", "`", "(", ")"]:
            with pytest.raises(ValueError):
                validate_provider_name(f"provider{c}evil")

    def test_allows_valid_names(self):
        """Valid provider names must pass through."""
        for name in ["zai", "minimax-cn", "openai-codex", "deepseek", "anthropic"]:
            assert validate_provider_name(name) == name

    def test_blocks_too_long(self):
        """Provider names >64 chars must be blocked."""
        with pytest.raises(ValueError, match="too long"):
            validate_provider_name("a" * 65)


# ━━━━ Strategy Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStrategyValidation:
    """Verify that validate_pool_strategy blocks injection."""

    def test_allows_valid_strategies(self):
        """Known strategies must pass."""
        for s in SUPPORTED_POOL_STRATEGIES:
            assert validate_pool_strategy(s) == s

    def test_blocks_unknown_strategy(self):
        """Unknown strategy must be blocked."""
        with pytest.raises(ValueError, match="Unknown"):
            validate_pool_strategy("evil_strategy")

    def test_blocks_sql_injection(self):
        """SQL-like strings must be blocked."""
        with pytest.raises(ValueError):
            validate_pool_strategy("'; DROP TABLE pools; --")

    def test_blocks_empty(self):
        """Empty strategy must be blocked."""
        with pytest.raises(ValueError):
            validate_pool_strategy("")

    def test_normalizes_case_and_whitespace(self):
        """Strategy is normalized to lowercase + stripped."""
        assert validate_pool_strategy("  Round_Robin  ") == "round_robin"

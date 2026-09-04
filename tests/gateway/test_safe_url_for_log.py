"""Security regression coverage for log-safe URLs."""

from gateway.platforms.base import safe_url_for_log


def test_malformed_url_fails_closed():
    result = safe_url_for_log("http://agent-vault-token:hermes@[bad")
    assert result == "<invalid-url>"
    assert "agent-vault-token" not in result

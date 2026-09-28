"""Regression tests for the OAuth User-Agent header in anthropic_adapter.py.

Two DIFFERENT Anthropic endpoints impose OPPOSITE User-Agent requirements:

- Inference (``/v1/messages`` via build_anthropic_client): requires the
  ``claude-code/`` UA + ``x-app: cli`` fingerprint, or requests get
  intermittent 500s. (issue #48534: ``claude-cli/`` is 404'd here.)
- OAuth token endpoint (``/v1/oauth/token`` login exchange + refresh):
  Anthropic now RATE-LIMITS (HTTP 429) any UA whose prefix is ``claude-code/``
  (or ``Mozilla/``). Verified empirically against platform.claude.com:
  ``claude-code/2.1.200`` -> 429; ``axios/*`` / ``node`` -> 400 (reached code
  validation). The token endpoint must therefore use a non-``claude-code/`` UA
  (we send ``axios/*``, matching the real Claude Code CLI's exchange client).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch



class TestOAuthUserAgentPrefix:
    """Inference uses ``claude-code/``; the OAuth token endpoint must NOT."""

    def test_build_anthropic_client_oauth_ua(self):
        """build_anthropic_client (INFERENCE) with OAuth token must use claude-code UA."""
        from agent.anthropic_adapter import build_anthropic_client

        mock_sdk = MagicMock()
        with patch("agent.anthropic_adapter._get_anthropic_sdk", return_value=mock_sdk):
            build_anthropic_client("sk-ant-oauth-abc123", "https://api.anthropic.com")

        # Inspect the kwargs passed to Anthropic()
        call_kwargs = mock_sdk.Anthropic.call_args[1]
        headers = call_kwargs.get("default_headers", {})
        ua = headers.get("user-agent", "") or headers.get("User-Agent", "")

        assert "claude-code/" in ua, f"Expected claude-code/ in UA, got: {ua}"
        assert "claude-cli/" not in ua, f"Must not use claude-cli/ prefix: {ua}"




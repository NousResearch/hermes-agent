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

import re
from unittest.mock import MagicMock, patch

import pytest


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



    def test_token_refresh_ua_not_throttled(self):
        """refresh_anthropic_oauth_pure must NOT send a throttled token-endpoint UA."""
        import inspect
        import agent.anthropic_adapter as mod

        func = getattr(mod, "refresh_anthropic_oauth_pure", None)
        if func is None or not callable(func):
            pytest.skip("refresh_anthropic_oauth_pure not found")
        source = inspect.getsource(func)

        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.strip()
            if ("User-Agent" in stripped or "user-agent" in stripped) and (
                "claude-cli/" in stripped or "claude-code/" in stripped
            ):
                pytest.fail(
                    f"Line {i}: throttled UA in refresh header: {stripped}"
                )
        assert "_OAUTH_TOKEN_USER_AGENT" in source, (
            "refresh_anthropic_oauth_pure should send the shared "
            "_OAUTH_TOKEN_USER_AGENT (non-claude-code) on the token endpoint"
        )

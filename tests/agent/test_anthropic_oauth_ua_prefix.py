"""Regression tests for the OAuth User-Agent header in anthropic_adapter.py.

Anthropic now 404s the OAuth token endpoint for any ``claude-cli/`` UA prefix
(issue #48534). The adapter must use ``claude-code/`` instead.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest


class TestOAuthUserAgentPrefix:
    """All OAuth-related HTTP requests must use ``claude-code/`` UA, not ``claude-cli/``."""

    def test_build_anthropic_client_oauth_ua(self):
        """build_anthropic_client with OAuth token must use claude-code UA."""
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

    def test_no_claude_cli_in_source(self):
        """Source file must not contain claude-cli/ UA pattern (blocks OAuth)."""
        import inspect
        import agent.anthropic_adapter as mod

        source = inspect.getsource(mod)
        # Allow claude-cli in comments/strings that reference the old behavior
        # but not in actual header assignments
        lines = source.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "claude-cli/" in stripped and ("User-Agent" in stripped or "user-agent" in stripped):
                pytest.fail(
                    f"Line {i}: claude-cli/ still used in User-Agent header: {stripped}"
                )

    def test_token_exchange_ua_prefix(self):
        """run_hermes_oauth_login_pure must not send claude-cli/ UA."""
        import inspect
        import agent.anthropic_adapter as mod

        # Get the source of the exchange function
        try:
            source = inspect.getsource(mod.run_hermes_oauth_login_pure)
        except AttributeError:
            pytest.skip("run_hermes_oauth_login_pure not found")

        # Only fail on claude-cli/ in an actual User-Agent header line — a
        # comment that references the old behavior (e.g. "Anthropic blocks
        # claude-cli/ on the OAuth endpoint") is allowed. Mirrors the
        # header-scoped check in test_no_claude_cli_in_source.
        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.strip()
            if "claude-cli/" in stripped and ("User-Agent" in stripped or "user-agent" in stripped):
                pytest.fail(
                    f"Line {i}: run_hermes_oauth_login_pure still uses claude-cli/ UA header: {stripped}"
                )
        assert "claude-code/" in source, (
            "run_hermes_oauth_login_pure should use claude-code/ UA"
        )

    def test_token_refresh_ua_prefix(self):
        """_refresh_oauth_token_raw must not send claude-cli/ UA."""
        import inspect
        import agent.anthropic_adapter as mod

        # Find the function that does the actual refresh HTTP call
        for name in ("_refresh_oauth_token_raw", "_do_token_refresh", "_refresh_oauth_token"):
            func = getattr(mod, name, None)
            if func and callable(func):
                source = inspect.getsource(func)
                if "urllib.request.Request" in source:
                    assert "claude-cli/" not in source, (
                        f"{name} still uses claude-cli/ UA"
                    )
                    break

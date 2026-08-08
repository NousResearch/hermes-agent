"""Regression tests for the OAuth User-Agent header in anthropic_adapter.py.

Two DIFFERENT Anthropic endpoints impose distinct User-Agent requirements:

- Inference (``/v1/messages`` via build_anthropic_client) matches the current
  Claude Agent SDK fingerprint: ``claude-cli/... (external, sdk-cli)``.
- OAuth token endpoint (``/v1/oauth/token`` login exchange + refresh):
  Anthropic now RATE-LIMITS (HTTP 429) any UA whose prefix is ``claude-code/``
  (or ``Mozilla/``). Verified empirically against platform.claude.com:
  ``claude-code/2.1.200`` -> 429; ``axios/*`` / ``node`` -> 400 (reached code
  validation). The token endpoint must therefore use a non-``claude-code/`` UA
  (we send ``axios/*``, matching the real Claude Code CLI's exchange client).
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import httpx
import pytest


class TestOAuthUserAgentPrefix:
    """Inference uses the SDK CLI UA; the OAuth token endpoint must not."""

    def test_build_anthropic_client_oauth_ua(self):
        """OAuth inference must match the current Claude Agent SDK UA."""
        from agent.anthropic_adapter import build_anthropic_client

        mock_sdk = MagicMock()
        with patch("agent.anthropic_adapter._get_anthropic_sdk", return_value=mock_sdk):
            build_anthropic_client("sk-ant-oauth-abc123", "https://api.anthropic.com")

        # Inspect the kwargs passed to Anthropic()
        call_kwargs = mock_sdk.Anthropic.call_args[1]
        headers = call_kwargs.get("default_headers", {})
        ua = headers.get("user-agent", "") or headers.get("User-Agent", "")

        assert "claude-cli/" in ua, f"Expected claude-cli/ in UA, got: {ua}"
        assert "(external, sdk-cli)" in ua
        betas = headers["anthropic-beta"]
        assert "oauth-2025-04-20" in betas
        assert "claude-code-20250219" in betas
        assert "fine-grained-tool-streaming-2025-05-14" not in betas

    def test_oauth_request_hook_rewrites_final_wire_headers(self):
        """The request hook must replace Python-SDK headers on the wire."""
        from agent.anthropic_adapter import _build_oauth_impersonation_http_client

        client = _build_oauth_impersonation_http_client(
            "oauth-token",
            timeout=30,
            anthropic_beta="beta-one,beta-two",
        )
        try:
            hook = client.event_hooks["request"][0]
            request = httpx.Request(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "user-agent": "anthropic-python/1.0",
                    "x-stainless-lang": "python",
                    "x-api-key": "must-be-removed",
                },
                json={"model": "claude-sonnet-4-6"},
            )

            hook(request)

            assert request.headers["user-agent"].startswith("claude-cli/")
            assert "(external, sdk-cli)" in request.headers["user-agent"]
            assert request.headers["x-stainless-lang"] == "js"
            assert request.headers["x-stainless-runtime"] == "node"
            assert request.headers["x-app"] == "cli"
            assert request.headers["anthropic-beta"] == "beta-one,beta-two"
            assert request.headers["authorization"] == "Bearer oauth-token"
            assert "x-api-key" not in request.headers
        finally:
            client.close()

    def test_oauth_tool_choice_none_omits_tools_and_tool_choice(self):
        """Disabling tools must remain effective when OAuth omits tool_choice."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }],
            max_tokens=128,
            reasoning_config=None,
            tool_choice="none",
            is_oauth=True,
        )

        assert "tool_choice" not in kwargs
        assert "tools" not in kwargs

    def test_oauth_body_fields_are_built_at_runtime(self):
        """OAuth body metadata is present and thinking edits are conditional."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        common = {
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": None,
            "max_tokens": 128,
            "is_oauth": True,
        }
        without_thinking = build_anthropic_kwargs(reasoning_config=None, **common)
        with_thinking = build_anthropic_kwargs(
            reasoning_config={"enabled": True, "effort": "medium"},
            **common,
        )

        body = without_thinking["extra_body"]
        identity = json.loads(body["metadata"]["user_id"])
        assert set(identity) == {"device_id", "account_uuid", "session_id"}
        assert body["diagnostics"] == {"previous_message_id": None}
        assert "context_management" not in body
        assert with_thinking["extra_body"]["context_management"] == {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}],
        }

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

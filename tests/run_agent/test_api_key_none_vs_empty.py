"""Tests for api_key None vs empty-string distinction.

The invariant: AIAgent must distinguish "not provided" (None) from "explicitly
empty" (""). Previously both were treated identically — any falsy value triggered
the env fallback, so api_key="" would silently leak the ANTHROPIC_TOKEN env var
to third-party providers (MiniMax, OpenRouter, etc.) that don't use Anthropic auth.

Fixed behaviour (proved by tests below):
  - api_key=None  + native anthropic  → resolve_anthropic_token() env fallback ✅
  - api_key=""    + native anthropic  → use '' as-is (explicit intent) ✅
  - api_key="x"   + any provider       → use 'x' as-is ✅

The third-party api_key=None case hits the chat_completions (OpenAI) branch,
not the anthropic_messages branch where the fix lives — testing it requires
bypassing the init provider-credential guard which is out of scope for this fix.
"""
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


class TestApiKeyNoneVsEmpty:
    """Verify None and '' are handled differently for all providers.

    Tests the effective_key logic at run_agent.py ~1539 inside
    api_mode == 'anthropic_messages' — the branch that resolves api_key
    before building the Anthropic client.
    """

    def test_none_key_native_anthropic_uses_env_fallback(self):
        """api_key=None on native Anthropic → resolve_anthropic_token() is called."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value="env-anthropic-token-sk-ant-1234") as mock_token,
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            a = AIAgent(
                api_key=None,
                provider="anthropic",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert a.api_key == "env-anthropic-token-sk-ant-1234"
        mock_token.assert_called_once()

    def test_empty_string_key_native_anthropic_does_not_use_env_fallback(self):
        """api_key='' on native Anthropic → env token NOT used; '' is explicit intent."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value="env-anthropic-token"),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            a = AIAgent(
                api_key="",
                provider="anthropic",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert a.api_key == ""
        # Explicit empty string should not trigger env fallback

    def test_explicit_key_used_as_is(self):
        """api_key='my-secret-key' on any provider → use the key verbatim."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="my-secret-key-xyz",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
        assert a.api_key == "my-secret-key-xyz"

"""Regression tests for auth retry / paid fallback guard.

Persistent OAuth 401s should get one last primary-provider retry before
failover, and they should not silently jump from ChatGPT/Codex-style OAuth to
paid Anthropic/Bedrock fallbacks after refresh recovery has failed.
"""

from types import SimpleNamespace

from agent.chat_completion_helpers import (
    _should_block_paid_oauth_auth_fallback,
    try_activate_fallback,
)
from agent.conversation_loop import _should_retry_primary_auth_before_fallback
from agent.error_classifier import FailoverReason


def test_codex_401_retries_primary_once_before_pending_fallback():
    agent = SimpleNamespace(
        provider="openai-codex",
        _fallback_chain=[{"provider": "anthropic", "model": "claude-opus-4-8"}],
        _fallback_index=0,
    )
    agent._has_pending_fallback = lambda: True
    classified = SimpleNamespace(is_auth=True)

    assert _should_retry_primary_auth_before_fallback(
        agent,
        classified,
        status_code=401,
        already_attempted=False,
    )
    assert not _should_retry_primary_auth_before_fallback(
        agent,
        classified,
        status_code=401,
        already_attempted=True,
    )


def test_auth_retry_before_fallback_is_limited_to_401_oauthish_providers():
    classified = SimpleNamespace(is_auth=True)
    unsupported_provider = SimpleNamespace(
        provider="openai-api",
        _fallback_chain=[{"provider": "anthropic", "model": "claude-opus-4-8"}],
        _fallback_index=0,
    )
    unsupported_provider._has_pending_fallback = lambda: True

    assert not _should_retry_primary_auth_before_fallback(
        unsupported_provider,
        classified,
        status_code=401,
        already_attempted=False,
    )

    codex = SimpleNamespace(
        provider="openai-codex",
        _fallback_chain=[{"provider": "anthropic", "model": "claude-opus-4-8"}],
        _fallback_index=0,
    )
    codex._has_pending_fallback = lambda: True
    assert not _should_retry_primary_auth_before_fallback(
        codex,
        classified,
        status_code=403,
        already_attempted=False,
    )


def test_paid_oauth_auth_fallback_guard_blocks_codex_to_anthropic():
    agent = SimpleNamespace()

    assert _should_block_paid_oauth_auth_fallback(
        agent,
        reason=FailoverReason.auth,
        current_provider="openai-codex",
        fallback_provider="anthropic",
    )
    assert _should_block_paid_oauth_auth_fallback(
        agent,
        reason=FailoverReason.auth,
        current_provider="xai-oauth",
        fallback_provider="bedrock",
    )


def test_paid_oauth_auth_fallback_guard_does_not_block_rate_limit_or_openrouter():
    agent = SimpleNamespace()

    assert not _should_block_paid_oauth_auth_fallback(
        agent,
        reason=FailoverReason.rate_limit,
        current_provider="openai-codex",
        fallback_provider="anthropic",
    )
    assert not _should_block_paid_oauth_auth_fallback(
        agent,
        reason=FailoverReason.auth,
        current_provider="openai-codex",
        fallback_provider="openrouter",
    )


def test_try_activate_fallback_does_not_consume_paid_fallback_on_codex_auth():
    class Agent(SimpleNamespace):
        def _try_activate_fallback(self, reason=None):
            return try_activate_fallback(self, reason=reason)

    agent = Agent(
        provider="openai-codex",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        _fallback_chain=[{"provider": "anthropic", "model": "claude-opus-4-8"}],
        _fallback_index=0,
        _fallback_activated=False,
        _primary_runtime={"provider": "openai-codex"},
    )

    assert try_activate_fallback(agent, reason=FailoverReason.auth) is False
    assert agent._fallback_index == 0
    assert agent.provider == "openai-codex"

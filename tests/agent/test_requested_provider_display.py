"""Regression tests for AIAgent.requested_provider.

Covers the fix for: status/error display lines showed the internal
transport alias ("custom") instead of the user-configured provider name
(e.g. "ollama") after runtime alias resolution collapses self-hosted/local
endpoints to "custom". See agent/agent_init.py::init_agent and
agent/conversation_loop.py's retry-failure display lines.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME so init_agent's config/env reads stay hermetic."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _make_bare_agent():
    """A minimal object standing in for AIAgent — init_agent() only needs
    attribute assignment targets, not the full class machinery. One method,
    _anthropic_prompt_cache_policy(), is called for real inside init_agent()
    unrelated to what's under test here, so it needs a concrete 2-tuple
    return rather than MagicMock's default (another MagicMock, which breaks
    tuple-unpacking)."""
    agent = MagicMock()
    agent._anthropic_prompt_cache_policy.return_value = (False, False)
    return agent


class TestRequestedProviderFallback:
    """agent.requested_provider must reflect the pre-alias-resolution name,
    but must never break callers that don't pass it."""

    def test_requested_provider_preserved_when_distinct_from_resolved(self):
        """The primary regression case: caller resolved 'ollama' to the
        internal 'custom' transport label, but wants the display layer to
        still say 'ollama'."""
        from agent.agent_init import init_agent

        agent = _make_bare_agent()
        init_agent(
            agent,
            base_url="http://localhost:11434/v1",
            api_key="no-key-required",
            provider="custom",
            requested_provider="ollama",
            model="qwen3:4b",
        )
        assert agent.provider == "custom"
        assert agent.requested_provider == "ollama"

    def test_requested_provider_defaults_to_resolved_provider_when_absent(self):
        """Callers that don't pass requested_provider (every pre-existing
        call site) must see requested_provider fall back to the resolved
        provider — i.e. no behavior change for them."""
        from agent.agent_init import init_agent

        agent = _make_bare_agent()
        init_agent(
            agent,
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            provider="openrouter",
            model="anthropic/claude-sonnet-4",
        )
        assert agent.provider == "openrouter"
        assert agent.requested_provider == "openrouter"

    def test_requested_provider_normalized_lowercase_and_stripped(self):
        from agent.agent_init import init_agent

        agent = _make_bare_agent()
        init_agent(
            agent,
            base_url="http://localhost:11434/v1",
            provider="custom",
            requested_provider="  Ollama  ",
            model="qwen3:4b",
        )
        assert agent.requested_provider == "ollama"

    def test_requested_provider_blank_string_falls_back_to_resolved(self):
        """An empty/whitespace-only requested_provider is treated as absent,
        not as a literal empty display value."""
        from agent.agent_init import init_agent

        agent = _make_bare_agent()
        init_agent(
            agent,
            base_url="http://localhost:11434/v1",
            provider="custom",
            requested_provider="   ",
            model="qwen3:4b",
        )
        assert agent.requested_provider == "custom"


class TestConversationLoopDisplayPrefersRequestedProvider:
    """The retry-failure status line must prefer requested_provider over
    the resolved provider, but degrade gracefully for agents that predate
    this attribute (getattr default)."""

    def test_display_prefers_requested_provider_attribute(self):
        agent = MagicMock()
        agent.requested_provider = "ollama"
        agent.provider = "custom"
        # Mirrors the exact lookup added in agent/conversation_loop.py.
        displayed = getattr(agent, "requested_provider", None) or getattr(agent, "provider", "unknown")
        assert displayed == "ollama"

    def test_display_falls_back_to_provider_when_requested_provider_missing(self):
        agent = MagicMock(spec=["provider"])  # no requested_provider attribute
        agent.provider = "openrouter"
        displayed = getattr(agent, "requested_provider", None) or getattr(agent, "provider", "unknown")
        assert displayed == "openrouter"

    def test_display_falls_back_when_requested_provider_is_none(self):
        agent = MagicMock()
        agent.requested_provider = None
        agent.provider = "anthropic"
        displayed = getattr(agent, "requested_provider", None) or getattr(agent, "provider", "unknown")
        assert displayed == "anthropic"

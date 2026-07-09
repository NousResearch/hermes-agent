"""Tests for Nous fallback local-availability suppression.

Blocker if Nous token material is missing locally: the fallback chain
should not repeatedly attempt Nous resolution; it must skip and continue
to the next provider.
"""

from __future__ import annotations

from unittest.mock import patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = None
        return agent


def _mock_client(base_url="https://chatgpt.com/backend-api/codex", api_key="fb-key"):
    mock = type("Client", (), {})()
    mock.base_url = base_url
    mock.api_key = api_key
    mock.chat = type("Chat", (), {})()
    mock.chat.completions = type("Completions", (), {})()
    mock.chat.completions.create = lambda *args, **kwargs: None
    return mock


class TestNousFallbackLocalAvailability:
    def test_missing_nous_token_is_skipped_once(self):
        """Nous fallback is skipped when no access/refresh token is stored."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
                {"provider": "openai-codex", "model": "gpt-5.5"},
            ]
        )
        with patch(
            "hermes_cli.auth.get_provider_auth_state",
            return_value={},
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(api_key="fb"), "gpt-5.5"),
        ):
            activated = agent._try_activate_fallback(None)
        assert activated is True
        assert agent.model == "gpt-5.5"

    def test_nous_unavailable_not_retried_in_same_session(self):
        """After Nous is skipped once, subsequent activations continue further.

        #60761 Bug 2: the old permanent set was replaced by a
        time-bounded cache. The skip still happens (Nous has no token
        material) but the cache holds an expiry timestamp rather than a
        permanent flag. The TTL default is 300s — long enough to ride
        out the rest of this test session.
        """
        agent = _make_agent(
            fallback_model=[
                {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
                {"provider": "openai-codex", "model": "gpt-5.5"},
            ]
        )
        with patch(
            "hermes_cli.auth.get_provider_auth_state",
            return_value={},
        ):
            agent._try_activate_fallback(None)
        # Time-bounded cache holds an expiry timestamp for the nous entry.
        cache = getattr(agent, "_fallback_unavailable_until", {})
        key = "nous/anthropic/claude-sonnet-4.6"
        assert key in cache
        # Expiry is in the future (within default 300s TTL).
        import time as _time
        assert cache[key] > _time.monotonic()

    def test_present_nous_token_allows_activation(self):
        """Nous is considered when token material exists."""
        agent = _make_agent(
            fallback_model=[
                {"provider": "nous", "model": "anthropic/claude-sonnet-4.6"},
                {"provider": "openai-codex", "model": "gpt-5.5"},
            ]
        )
        with patch(
            "hermes_cli.auth.get_provider_auth_state",
            return_value={"access_token": "abc", "refresh_token": "xyz"},
        ), patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(api_key="fb"), "anthropic/claude-sonnet-4.6"),
        ):
            activated = agent._try_activate_fallback(None)
        assert activated is True
        assert agent.provider == "nous"

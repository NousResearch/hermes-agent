"""Regression test — anthropic fallback must not commit agent state before
the native Anthropic client is successfully built.

``try_activate_fallback`` used to swap ``agent.model`` / ``agent.provider`` /
``agent.base_url`` / ``agent.api_mode`` *before* calling
``build_anthropic_client()`` for the ``anthropic_messages`` branch. When that
call raised (e.g. ``ImportError`` because the ``anthropic`` package isn't
installed), the agent was left half-switched: pointing at the fallback
provider with no working client. See fix in this same PR.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://api.anthropic.com/v1", api_key="sk-ant-test"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Client build failure must not mutate agent state ───────────────────────


class TestAnthropicFallbackClientBuildFailure:
    def test_anthropic_fallback_does_not_mutate_agent_state_on_client_build_failure(self):
        fbs = [{"provider": "anthropic", "model": "claude-sonnet-4-6"}]
        agent = _make_agent(fallback_model=fbs)

        original_model = agent.model
        original_provider = agent.provider
        original_base_url = agent.base_url
        original_api_mode = agent.api_mode
        original_client = agent.client

        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), None),
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=ImportError("anthropic package missing"),
            ),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is False
        assert agent.model == original_model
        assert agent.provider == original_provider
        assert agent.base_url == original_base_url
        assert agent.api_mode == original_api_mode
        assert agent.client is original_client
        assert agent._anthropic_client is None
        assert agent._fallback_activated is False

    def test_broken_entry_then_working_entry_still_activates(self):
        fbs = [
            {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        ]
        agent = _make_agent(fallback_model=fbs)

        def _client_for(provider, **kwargs):
            if provider == "anthropic":
                return (_mock_client(), None)
            return (
                _mock_client(base_url="https://openrouter.ai/api/v1", api_key="sk-or-test"),
                None,
            )

        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                side_effect=lambda provider, **kw: _client_for(provider, **kw),
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=ImportError("anthropic package missing"),
            ),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.api_mode == "chat_completions"
        assert agent.provider == "openrouter"
        assert agent._anthropic_client is None
        assert agent.client is not None

    def test_successful_anthropic_fallback_unchanged_behavior(self):
        fbs = [{"provider": "anthropic", "model": "claude-sonnet-4-6"}]
        agent = _make_agent(fallback_model=fbs)
        built_client = MagicMock()

        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(), None),
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=built_client,
            ) as mock_build,
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.api_mode == "anthropic_messages"
        assert agent._anthropic_client is built_client
        assert agent.client is None
        assert agent._fallback_activated is True
        mock_build.assert_called_once()

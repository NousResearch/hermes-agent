"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
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
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None

    def test_single_dict_backwards_compat(self):
        fb = {"provider": "openai", "model": "gpt-4o"}
        agent = _make_agent(fallback_model=fb)
        assert agent._fallback_chain == [fb]
        assert agent._fallback_model == fb

    def test_list_of_providers(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 2
        assert agent._fallback_model == fbs[0]

    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"

    def test_empty_list(self):
        agent = _make_agent(fallback_model=[])
        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    def test_second_fallback_works(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._try_activate_fallback() is True
            assert agent.model == "glm-4.7"
            assert agent._fallback_index == 2

    def test_all_exhausted_returns_false(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False

    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"


class TestSwitchModelFallbackRefresh:
    def test_switch_model_replaces_runtime_fallback_chain(self):
        agent = _make_agent(fallback_model={"provider": "zai", "model": "glm-5"})
        agent.context_compressor = None
        agent._create_openai_client = MagicMock(return_value=_mock_client(
            base_url="https://api.openai.com/v1",
            api_key="new-key",
        ))

        new_chain = [{"provider": "anthropic", "model": "claude-sonnet-4-6"}]
        agent.switch_model(
            new_model="gpt-5.4",
            new_provider="openai",
            api_key="new-key",
            base_url="https://api.openai.com/v1",
            api_mode="chat_completions",
            fallback_model=new_chain,
        )

        assert agent._fallback_chain == new_chain
        assert agent._fallback_model == new_chain[0]

    def test_switch_model_clears_runtime_fallback_chain_when_given_empty_list(self):
        agent = _make_agent(fallback_model={"provider": "zai", "model": "glm-5"})
        agent.context_compressor = None
        agent._create_openai_client = MagicMock(return_value=_mock_client(
            base_url="https://api.openai.com/v1",
            api_key="new-key",
        ))

        agent.switch_model(
            new_model="gpt-5.4",
            new_provider="openai",
            api_key="new-key",
            base_url="https://api.openai.com/v1",
            api_mode="chat_completions",
            fallback_model=[],
        )

        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_switch_model_uses_replaced_chain_when_fallback_activates(self):
        agent = _make_agent(fallback_model=[{"provider": "zai", "model": "glm-5"}])
        agent.context_compressor = None
        agent._create_openai_client = MagicMock(return_value=_mock_client(
            base_url="https://api.openai.com/v1",
            api_key="new-key",
        ))

        new_chain = [{"provider": "anthropic", "model": "claude-sonnet-4-6"}]
        agent.switch_model(
            new_model="gpt-5.4",
            new_provider="openai",
            api_key="new-key",
            base_url="https://api.openai.com/v1",
            api_mode="chat_completions",
            fallback_model=new_chain,
        )

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(base_url="https://api.anthropic.com", api_key="fb-key"), "claude-sonnet-4-6"),
        ) as mock_rpc:
            assert agent._try_activate_fallback() is True

        mock_rpc.assert_called_once()
        args, kwargs = mock_rpc.call_args
        assert args == ("anthropic",)
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert agent.model == "claude-sonnet-4-6"
        assert agent.provider == "anthropic"
        assert agent._fallback_index == 1
        assert agent._fallback_activated is True

    def test_switch_model_preserves_existing_chain_when_fallback_model_is_none(self):
        original_chain = [{"provider": "zai", "model": "glm-5"}]
        agent = _make_agent(fallback_model=original_chain)
        agent.context_compressor = None
        agent._create_openai_client = MagicMock(return_value=_mock_client(
            base_url="https://api.openai.com/v1",
            api_key="new-key",
        ))

        agent._fallback_index = 1
        agent._fallback_activated = True
        agent.switch_model(
            new_model="gpt-5.4",
            new_provider="openai",
            api_key="new-key",
            base_url="https://api.openai.com/v1",
            api_mode="chat_completions",
            fallback_model=None,
        )

        assert agent._fallback_chain == original_chain
        assert agent._fallback_model == original_chain[0]
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False

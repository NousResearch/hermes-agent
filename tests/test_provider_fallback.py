"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from unittest.mock import MagicMock, patch

import pytest
import run_agent
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _clear_hot_failover_state():
    run_agent._HOT_FAILOVER_COOLDOWNS.clear()
    yield
    run_agent._HOT_FAILOVER_COOLDOWNS.clear()


def _make_agent(
    fallback_model=None,
    *,
    provider=None,
    model="claude-opus-4-6",
    provider_logged_in=False,
    default_model=None,
):
    """Create a minimal AIAgent with optional fallback config."""
    provider_patch = (
        patch("run_agent._provider_is_logged_in", side_effect=provider_logged_in)
        if callable(provider_logged_in)
        else patch("run_agent._provider_is_logged_in", return_value=provider_logged_in)
    )
    explicit_base_url = None
    if provider == "openai-codex":
        explicit_base_url = "https://chatgpt.com/backend-api/codex"
    elif provider == "anthropic":
        explicit_base_url = "https://api.anthropic.com"

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        provider_patch,
        patch("run_agent._default_model_for_provider", return_value=default_model),
    ):
        agent = AIAgent(
            api_key="***",
            base_url=explicit_base_url,
            provider=provider,
            model=model,
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

    def test_auto_appends_codex_for_anthropic_primary(self):
        agent = _make_agent(
            fallback_model=None,
            provider="anthropic",
            model="claude-opus-4-6",
            provider_logged_in=lambda provider: provider == "openai-codex",
            default_model="gpt-5.4",
        )

        assert agent._fallback_chain == [{"provider": "openai-codex", "model": "gpt-5.4"}]

    def test_auto_appends_anthropic_for_codex_primary(self):
        agent = _make_agent(
            fallback_model=None,
            provider="openai-codex",
            model="gpt-5.4",
            provider_logged_in=lambda provider: provider == "anthropic",
            default_model="claude-opus-4-6",
        )

        assert agent._fallback_chain == [{"provider": "anthropic", "model": "claude-opus-4-6"}]

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

    def test_skips_rate_limited_provider_to_next(self):
        fbs = [
            {"provider": "openai-codex", "model": "gpt-5.4"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs, provider="anthropic", model="claude-opus-4-6")
        with (
            patch.object(agent, "_provider_rate_limit_hint", side_effect=[{"source": "cooldown", "retry_after_seconds": 90}, None]),
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(_mock_client(), "glm-4.7")),
        ):
            assert agent._try_activate_fallback() is True
            assert agent.model == "glm-4.7"
            assert agent._fallback_index == 2

    def test_preflight_hot_failover_switches_before_request(self):
        agent = _make_agent(
            fallback_model={"provider": "anthropic", "model": "claude-opus-4-6"},
            provider="openai-codex",
            model="gpt-5.4",
        )
        with (
            patch.object(agent, "_provider_rate_limit_hint", return_value={"provider": "openai-codex", "retry_after_seconds": 120, "source": "codexbar"}),
            patch.object(agent, "_try_activate_fallback", return_value=True) as mock_activate,
        ):
            assert agent._maybe_activate_hot_failover_before_request() is True
            mock_activate.assert_called_once()

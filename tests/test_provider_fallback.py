 """Tests for ordered provider fallback sequence (#1734)."""
import pytest
from unittest.mock import MagicMock, patch


def make_agent(fallback_model=None):
    with patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("run_agent.load_hermes_dotenv"), \
         patch("run_agent.load_config", return_value={}):
        from run_agent import AIAgent
        return AIAgent(
            model="anthropic/claude-opus-4.6",
            quiet_mode=True,
            fallback_model=fallback_model,
        )


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0

    def test_single_dict_backwards_compat(self):
        fb = {"provider": "openai", "model": "gpt-4o"}
        agent = make_agent(fallback_model=fb)
        assert agent._fallback_chain == [fb]

    def test_list_of_providers(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 2

    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
        ]
        agent = make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1


class TestFallbackSequence:
    def test_try_activate_exhausted(self):
        agent = make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_try_activate_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rc:
            mock_client = MagicMock()
            mock_client.base_url = "https://api.openai.com/v1"
            mock_rc.return_value = (mock_client, "gpt-4o")
            result = agent._try_activate_fallback()
            assert result is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"

    def test_all_exhausted_returns_false(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rc:
            mock_client = MagicMock()
            mock_client.base_url = "https://api.openai.com/v1"
            mock_rc.return_value = (mock_client, "gpt-4o")
            agent._try_activate_fallback()
            result = agent._try_activate_fallback()
            assert result is False

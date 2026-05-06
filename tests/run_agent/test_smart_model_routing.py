"""Tests for conservative two-tier smart model routing."""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.smart_model_routing import decide_route


def _make_agent(smart_model_routing=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            provider="openai-codex",
            model="gpt-5.5",
            api_key="primary-key",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            smart_model_routing=smart_model_routing,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://cheap.example/v1", api_key="cheap-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


class TestSmartRoutingDecision:
    def test_short_plain_text_routes_to_cheap_model(self):
        cfg = {
            "enabled": True,
            "max_simple_chars": 160,
            "max_simple_words": 28,
            "cheap_model": {"provider": "custom", "model": "minimax-m2.5"},
        }

        decision = decide_route("你好，简单介绍一下你自己", cfg)

        assert decision.target == "cheap"
        assert decision.reason == "short_plain_text"

    def test_code_and_tool_like_tasks_stay_primary(self):
        cfg = {
            "enabled": True,
            "max_simple_chars": 400,
            "max_simple_words": 80,
            "cheap_model": {"provider": "custom", "model": "minimax-m2.5"},
        }

        assert decide_route("帮我修改 run_agent.py 并运行测试", cfg).target == "primary"
        assert decide_route("```python\nprint('hi')\n```", cfg).target == "primary"


class TestSmartRoutingRuntime:
    def test_simple_turn_temporarily_switches_to_cheap_model(self):
        cfg = {
            "enabled": True,
            "max_simple_chars": 160,
            "max_simple_words": 28,
            "cheap_model": {
                "provider": "custom",
                "model": "minimax-m2.5",
                "base_url": "https://cheap.example/v1",
                "api_key": "cheap-key",
            },
        }
        agent = _make_agent(cfg)

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(), "minimax-m2.5"),
        ) as mock_rpc:
            assert agent._maybe_apply_smart_routing("你好") is True

        assert mock_rpc.call_args.kwargs["explicit_base_url"] == "https://cheap.example/v1"
        assert mock_rpc.call_args.kwargs["explicit_api_key"] == "cheap-key"
        assert agent.model == "minimax-m2.5"
        assert agent.provider == "custom"
        assert agent._fallback_activated is True

    def test_complex_turn_stays_on_primary(self):
        cfg = {
            "enabled": True,
            "cheap_model": {"provider": "custom", "model": "minimax-m2.5"},
        }
        agent = _make_agent(cfg)

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            assert agent._maybe_apply_smart_routing("帮我 debug 这个 traceback 并修改代码") is False

        mock_rpc.assert_not_called()
        assert agent.model == "gpt-5.5"
        assert agent.provider == "openai-codex"

    def test_disabled_config_does_nothing(self):
        agent = _make_agent({"enabled": False})

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            assert agent._maybe_apply_smart_routing("你好") is False

        mock_rpc.assert_not_called()

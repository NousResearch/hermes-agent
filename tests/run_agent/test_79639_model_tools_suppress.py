"""Regression test for #79639: `model.tools: false` suppresses the tools payload.

Hermes unconditionally sends `tools` on every chat-completions request when a
toolset is loaded. For Ollama-backed models whose tool-use template drops
prior non-tool turns (Gemma 4, reproduced on qwen3), the presence of `tools`
alone makes the model "forget" earlier context. This test pins the config
parsing: an explicit `model.tools: false` sets ``agent._suppress_tools`` so
the transport can drop the tools payload; default behavior is unchanged.
"""
from unittest.mock import patch, MagicMock

import pytest

from run_agent import AIAgent
from agent.agent_init import init_agent


def _make_agent():
    agent = object.__new__(AIAgent)
    agent._base_url = "http://127.0.0.1:11434/v1"
    agent._base_url_lower = ""
    agent._base_url_hostname = ""
    return agent


def _run_init(agent, model_cfg: dict):
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None)), \
         patch("run_agent.get_tool_definitions", return_value=[]), \
         patch("hermes_cli.config.load_config_readonly", return_value=model_cfg), \
         patch("hermes_cli.config.load_config", return_value=model_cfg), \
         patch("hermes_cli.config.cfg_get", return_value=None), \
         patch("agent.credential_pool.load_pool", return_value=MagicMock()), \
         patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]), \
         patch("agent.iteration_budget.IterationBudget"), \
         patch("agent.agent_init.query_ollama_num_ctx", return_value=None), \
         patch("agent.agent_init.is_local_endpoint", return_value=False):
        init_agent(
            agent,
            base_url="http://127.0.0.1:11434/v1",
            api_key="test-key",
            provider="custom",
            model="gemma4:12b",
            skip_context_files=True,
            skip_memory=True,
            quiet_mode=True,
        )


class TestModelToolsSuppress:
    """`model.tools: false` must set `_suppress_tools`; defaults must not."""

    def test_tools_false_sets_suppress(self):
        agent = _make_agent()
        _run_init(agent, {"model": {"tools": False}})
        assert agent._suppress_tools is True

    def test_default_does_not_suppress(self):
        agent = _make_agent()
        _run_init(agent, {"model": {}})
        assert getattr(agent, "_suppress_tools", False) is False

    def test_tools_true_does_not_suppress(self):
        agent = _make_agent()
        _run_init(agent, {"model": {"tools": True}})
        assert agent._suppress_tools is False

    def test_no_model_section_does_not_suppress(self):
        agent = _make_agent()
        _run_init(agent, {})
        assert getattr(agent, "_suppress_tools", False) is False

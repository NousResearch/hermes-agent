"""Ollama's ``options.num_ctx`` belongs to the local endpoint it was resolved for.

``_ollama_num_ctx`` is resolved once at construction. An agent that boots on a local Ollama and is
then moved to a hosted endpoint (``/model``, fallback activation, primary restore) kept sending it, and strict
hosts reject unknown body fields: Mistral answers ``HTTP 422 extra_forbidden`` on ``options``.
"""
from contextlib import ExitStack
from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import _build_api_kwargs_for_mode
from agent.error_classifier import FailoverReason

_CFG = {"agent": {}, "model": {}}


@pytest.fixture
def ollama_agent():
    import agent.context_compressor as cc_mod
    with ExitStack() as stack:
        for target, value in (
            ("model_tools.get_tool_definitions", []),
            ("model_tools.check_toolset_requirements", {}),
            ("hermes_cli.config.load_config", _CFG),
            ("hermes_cli.config.load_config_readonly", _CFG),
            ("agent.model_metadata.get_model_context_length", 262144),
            ("agent.agent_init.query_ollama_num_ctx", 65536),
        ):
            stack.enter_context(patch(target, return_value=value))
        stack.enter_context(patch("agent.process_bootstrap.OpenAI"))
        stack.enter_context(patch.object(cc_mod, "get_model_context_length", return_value=262144))
        from run_agent import AIAgent
        agent = AIAgent(
            model="qwen3.6:35b-mlx", provider="custom", api_key="ollama",
            base_url="http://localhost:11434/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
            fallback_model={"provider": "custom", "model": "mistral-medium-latest",
                            "base_url": "https://api.mistral.ai/v1", "api_key": "sk-test"},
        )
        yield agent


def _wire_extra_body(agent):
    return _build_api_kwargs_for_mode(agent, [{"role": "user", "content": "hi"}], []).get("extra_body") or {}


def test_num_ctx_goes_to_the_local_ollama_it_was_resolved_for(ollama_agent):
    assert _wire_extra_body(ollama_agent)["options"] == {"num_ctx": 65536}


@pytest.mark.parametrize("entrypoint", ["switch", "fallback"])
def test_moving_to_a_hosted_endpoint_drops_num_ctx(ollama_agent, entrypoint):
    if entrypoint == "switch":
        ollama_agent.switch_model(
            "mistral-medium-latest", "custom:mistral", api_key="sk-test",
            base_url="https://api.mistral.ai/v1", api_mode="chat_completions",
        )
    else:
        assert ollama_agent._try_activate_fallback(FailoverReason.rate_limit)
    assert ollama_agent.base_url.rstrip("/") == "https://api.mistral.ai/v1"
    assert "options" not in _wire_extra_body(ollama_agent)

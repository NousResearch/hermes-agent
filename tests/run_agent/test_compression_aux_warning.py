"""Regression tests for compression auxiliary provider startup warnings."""

from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


def _bare_agent():
    agent = object.__new__(AIAgent)
    agent.compression_enabled = True
    agent.model = "anthropic/claude-sonnet-4-6"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key"
    agent.api_mode = "chat_completions"
    agent._aux_compression_context_length_config = None
    agent.context_compressor = SimpleNamespace(threshold_tokens=100_000)
    agent._compression_warning = None
    emitted = []
    agent._emit_status = emitted.append
    return agent, emitted


def test_missing_explicit_compression_provider_warning_names_provider_and_env_var():
    agent, emitted = _bare_agent()

    with (
        patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(None, None)),
        patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("minimax", "mini-max-01", None, None, None),
        ),
    ):
        agent._check_compression_model_feasibility()

    assert emitted
    warning = emitted[0]
    assert "auxiliary.compression.provider=minimax" in warning
    assert "MINIMAX_API_KEY" in warning
    assert "OPENROUTER_API_KEY" not in warning
    assert agent._compression_warning == warning

"""A session's explicit reasoning effort survives a provider fallback.

Precedence is session choice > per-model ``agent.reasoning_overrides`` > global
``agent.reasoning_effort``. Fallback activation re-resolves the last two for the fallback model
(#21256); it must not replace a level the session chose (``hermes chat --reasoning``, a session
``/reasoning``), which callers mark with ``agent.reasoning_pinned``.
"""

from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import get_hermes_home
from run_agent import AIAgent

_FALLBACK = {"provider": "openai", "model": "fb-model"}


@pytest.fixture
def profile_config():
    """The profile's levels: global medium, and a per-model override for the fallback model."""
    (get_hermes_home() / "config.yaml").write_text(
        "agent:\n  reasoning_effort: medium\n  reasoning_overrides:\n    fb-model: low\n", encoding="utf-8")


def _agent(reasoning_config):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="primary-model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[_FALLBACK], reasoning_config=reasoning_config,
        )
    agent.client = MagicMock()
    return agent


def _activate_fallback(agent):
    client = MagicMock()
    client.base_url, client.api_key = "https://api.openai.com/v1", "fb-key"
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, _FALLBACK["model"])):
        assert agent._try_activate_fallback() is True
    assert agent.model == _FALLBACK["model"]


def test_fallback_keeps_a_pinned_session_level(profile_config):
    agent = _agent({"enabled": True, "effort": "high"})
    agent.reasoning_pinned = True

    _activate_fallback(agent)

    assert agent.reasoning_config == {"enabled": True, "effort": "high"}


def test_fallback_without_a_session_level_resolves_the_fallback_models_config(profile_config):
    agent = _agent({"enabled": True, "effort": "medium"})

    _activate_fallback(agent)

    assert agent.reasoning_config == {"enabled": True, "effort": "low"}


def test_primary_restore_after_a_pinned_fallback_runs_the_session_level(profile_config):
    agent = _agent({"enabled": True, "effort": "high"})
    agent.reasoning_pinned = True
    _activate_fallback(agent)

    with patch("agent.agent_runtime_helpers._rebuild_primary_client"):
        assert agent._restore_primary_runtime() is True

    assert agent.model == "primary-model"
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}

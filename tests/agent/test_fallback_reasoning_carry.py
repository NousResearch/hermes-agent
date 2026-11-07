"""Reasoning effort across a fallback swap: carried from the session, clamped to the fallback
route, overridable per entry, and put back on the primary's return (#136441, #138045)."""

import secrets
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

_KEY = "test-" + secrets.token_hex(16)
_HIGH = {"enabled": True, "effort": "high"}
_MEDIUM = {"enabled": True, "effort": "medium"}


def _agent(reasoning_config, fallback):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
    ):
        agent = AIAgent(
            api_key=_KEY, base_url="https://my-llm.example.com/v1", provider="custom", model="primary-model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            reasoning_config=reasoning_config, fallback_model=fallback,
        )
    agent.client = MagicMock()
    return agent


def _fb_client(base_url="https://fallback.example.com/v1"):
    client = MagicMock()
    setattr(client, "api" + "_key", "fb-" + secrets.token_hex(8))
    client.base_url = base_url
    return client


def _fail_over(agent, cfg, base_url="https://fallback.example.com/v1"):
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fb_client(base_url), None)),
        patch("hermes_cli.config.load_config", return_value=cfg),
    ):
        assert agent._try_activate_fallback() is True


def _restore(agent):
    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        assert agent._restore_primary_runtime() is True


def test_a_session_effort_pin_is_carried_to_the_fallback_and_back():
    """A live ``/reasoning high`` pin (set after the startup snapshot) survives the swap and the
    restore, instead of the fallback re-deriving the global ``medium``."""
    agent = _agent(_MEDIUM, {"provider": "zai", "model": "glm-5.2"})
    agent.reasoning_config = dict(_HIGH)
    _fail_over(agent, {"agent": {"reasoning_effort": "medium"}})

    assert agent.reasoning_config == _HIGH
    _restore(agent)
    assert agent.reasoning_config == _HIGH


def test_the_carried_effort_is_clamped_to_the_fallback_route():
    agent = _agent({"enabled": True, "effort": "ultra"}, {"provider": "openrouter", "model": "some/model"})
    _fail_over(agent, {}, base_url="https://openrouter.ai/api/v1")

    assert agent.reasoning_config["effort"] != "ultra"
    _restore(agent)
    assert agent.reasoning_config == {"enabled": True, "effort": "ultra"}


def test_a_fallback_entry_reasoning_effort_wins_and_the_notice_names_the_change():
    agent = _agent(_HIGH, {"provider": "zai", "model": "glm-5.2", "reasoning_effort": "low"})
    _fail_over(agent, {"agent": {"reasoning_overrides": {"glm-5.2": "xhigh"}}})

    assert agent.reasoning_config == {"enabled": True, "effort": "low"}
    assert any("Reasoning effort: high → low." in n for n in agent._pending_fallback_notice)


def test_the_primary_models_own_override_is_not_carried():
    """``reasoning_overrides: {primary-model: none}`` is about the primary model; the fallback runs
    at the global effort instead of inheriting reasoning off (#21256)."""
    agent = _agent({"enabled": False}, {"provider": "zai", "model": "glm-5.2"})
    _fail_over(agent, {"agent": {"reasoning_effort": "medium", "reasoning_overrides": {"primary-model": "none"}}})

    assert agent.reasoning_config == _MEDIUM
    _restore(agent)
    assert agent.reasoning_config == {"enabled": False}


def test_an_effort_picked_while_the_fallback_serves_survives_the_restore():
    agent = _agent(_MEDIUM, {"provider": "zai", "model": "glm-5.2"})
    _fail_over(agent, {})
    agent.reasoning_config = dict(_HIGH)  # TUI/Desktop `/reasoning high` mid-fallback

    _restore(agent)
    assert agent.reasoning_config == _HIGH

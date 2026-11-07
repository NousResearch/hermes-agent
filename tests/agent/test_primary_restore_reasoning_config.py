"""The startup snapshot must carry the primary's reasoning effort so a fallback turn's
re-resolved effort (e.g. a ``none`` override for the fallback model) does not stick after
``restore_primary_runtime`` (#138045)."""

import secrets
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

_TEST_KEY = "test-" + secrets.token_hex(16)


def _make_agent(reasoning_config):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key=_TEST_KEY,
            base_url="https://my-llm.example.com/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            reasoning_config=reasoning_config,
            fallback_model={"provider": "zai", "model": "glm-5.2"},
        )
        agent.client = MagicMock()
        return agent


def _fallback_client():
    fb_client = MagicMock()
    setattr(fb_client, "api" + "_key", "fallback-" + secrets.token_hex(16))
    fb_client.base_url = "https://fallback.example.com/v1"
    return fb_client


def test_startup_snapshot_records_the_primary_reasoning_config():
    agent = _make_agent({"enabled": True, "effort": "medium"})

    assert agent._primary_runtime["reasoning_config"] == {"enabled": True, "effort": "medium"}


def test_startup_snapshot_records_none_when_the_primary_has_no_reasoning_config():
    agent = _make_agent(None)

    assert agent._primary_runtime["reasoning_config"] is None


def test_restore_puts_the_primary_effort_back_after_a_fallback_override():
    agent = _make_agent({"enabled": True, "effort": "medium"})
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), None)),
        patch(
            "hermes_cli.config.load_config",
            return_value={"agent": {"reasoning_overrides": {"glm-5.2": "none"}}},
        ),
    ):
        assert agent._try_activate_fallback() is True
    assert agent.reasoning_config == {"enabled": False}

    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        restored = agent._restore_primary_runtime()

    assert restored is True
    assert agent.reasoning_config == {"enabled": True, "effort": "medium"}


def test_restore_clears_the_fallback_override_when_the_primary_had_no_reasoning_config():
    agent = _make_agent(None)
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(_fallback_client(), None)),
        patch(
            "hermes_cli.config.load_config",
            return_value={"agent": {"reasoning_overrides": {"glm-5.2": "none"}}},
        ),
    ):
        assert agent._try_activate_fallback() is True
    assert agent.reasoning_config == {"enabled": False}

    with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
        restored = agent._restore_primary_runtime()

    assert restored is True
    assert agent.reasoning_config is None

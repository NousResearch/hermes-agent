"""Stable parent prompt guidance for the opt-in automatic mode router."""

from unittest.mock import patch

from agent.system_prompt import PARENT_MODE_ROUTING_GUIDANCE, build_system_prompt_parts
from run_agent import AIAgent


_ROUTING_MARKERS = (
    "thinking-expansion",
    "research-analysis",
    "execution-development",
)


def _make_agent(mode_router, *, ephemeral="private per-call instruction"):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch(
            "hermes_cli.config.load_config",
            return_value={"agent": {"mode_router": mode_router}},
        ),
    ):
        return AIAgent(
            model="test-model",
            api_key="test-key-1234567890",
            base_url="https://example.test/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
            ephemeral_system_prompt=ephemeral,
        )


def test_disabled_mode_router_preserves_prompt_byte_for_byte():
    baseline = _make_agent({})._build_system_prompt()
    disabled = _make_agent({"enabled": False})._build_system_prompt()

    assert disabled == baseline
    assert not any(marker in disabled for marker in _ROUTING_MARKERS)


def test_enabled_mode_router_adds_complete_policy_once_to_stable_prompt_only():
    agent = _make_agent({"enabled": True})
    parts = build_system_prompt_parts(agent, system_message="turn context")
    prompt = agent._build_system_prompt(system_message="turn context")

    assert parts["stable"].count(PARENT_MODE_ROUTING_GUIDANCE) == 1
    assert prompt.count(PARENT_MODE_ROUTING_GUIDANCE) == 1
    for marker in _ROUTING_MARKERS:
        assert marker in parts["stable"]
        assert marker not in parts["context"]
        assert marker not in parts["volatile"]
    assert "exploration" in parts["stable"]
    assert "no side effects" in parts["stable"]
    assert "evidence" in parts["stable"]
    assert "explicitly requested or approved" in parts["stable"]
    assert "research-analysis to execution-development" in parts["stable"]
    assert "implicitly" in parts["stable"]
    assert "verification" in parts["stable"]


def test_mode_router_config_is_snapshotted_at_agent_construction():
    agent = _make_agent({"enabled": True})

    with patch(
        "hermes_cli.config.load_config",
        return_value={"agent": {"mode_router": {"enabled": False}}},
    ):
        first = agent._build_system_prompt()
        second = agent._build_system_prompt()

    assert first == second
    assert first.count("thinking-expansion") == 1


def test_mode_router_does_not_mutate_ephemeral_prompt_or_tool_schema_identity():
    ephemeral = "private per-call instruction"
    agent = _make_agent({"enabled": True}, ephemeral=ephemeral)
    tools = agent.tools

    agent._build_system_prompt()

    assert agent.ephemeral_system_prompt == ephemeral
    assert agent.tools is tools

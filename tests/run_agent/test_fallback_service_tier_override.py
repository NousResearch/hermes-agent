"""Route-local tier filtering through the real request/transport builders."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from agent.fast_mode import begin_turn, effective_request_overrides
from run_agent import AIAgent


@pytest.fixture
def make_agent():
    agents = []
    with (
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
    ):
        def make(**kwargs):
            options = dict(
                provider="custom", model="gpt-5.5", api_mode="chat_completions",
                api_key="fixture-key", base_url="https://tier.example.test/v1",
                quiet_mode=True, skip_memory=True, skip_context_files=True,
            )
            options.update(kwargs)
            agent = AIAgent(**options)
            agents.append(agent)
            return agent
        yield make
        for agent in agents:
            agent.shutdown_memory_provider()
            agent.close()


@pytest.mark.parametrize("active_policy", ["normal", None, "unsupported"])
def test_real_transport_strips_only_tier_selectors_without_mutating_inputs(make_agent, active_policy):
    source = {
        "service_tier": "priority", "speed": "fast",
        "extra_body": {
            "service_tier": "flex", "speed": "fast", "store": False,
            "provider_marker": {"items": ["keep"]},
        },
        "metadata": {"trace": ["original"]},
    }
    before = deepcopy(source)
    agent = make_agent(request_overrides=source, service_tier="priority")
    configured = deepcopy(agent.request_overrides)
    agent._active_fallback_service_tier_override = active_policy
    wire = agent._build_api_kwargs([{"role": "user", "content": "fixture"}])
    if active_policy == "normal":
        for section in (wire, wire["extra_body"]):
            assert "service_tier" not in section
            assert "speed" not in section
    else:
        assert wire["service_tier"] == "priority"
        assert wire["speed"] == "fast"
        assert wire["extra_body"]["service_tier"] == "flex"
        assert wire["extra_body"]["speed"] == "fast"
    assert wire["extra_body"]["store"] is False
    assert wire["extra_body"]["provider_marker"] == {"items": ["keep"]}
    wire["extra_body"]["provider_marker"]["items"].append("wire-only")
    wire["metadata"]["trace"].append("wire-only")
    assert source == before
    assert agent.request_overrides == configured
    assert agent.service_tier == "priority"


@pytest.mark.parametrize("mode", ["auto", "cold"])
@pytest.mark.parametrize("api_mode", ["chat_completions", "codex_responses", "anthropic_messages"])
def test_normal_policy_filters_after_real_fast_window_evaluation(make_agent, mode, api_mode):
    anthropic = api_mode == "anthropic_messages"
    agent = make_agent(
        provider="anthropic" if anthropic else "openai",
        model="claude-opus-4-8" if anthropic else "gpt-5.5",
        base_url="https://api.anthropic.com" if anthropic else "https://api.openai.com/v1",
        api_mode=api_mode, service_tier=mode,
    )
    field = "speed" if anthropic else "service_tier"
    premium = "fast" if anthropic else "priority"
    with patch("agent.fast_mode.time.monotonic", return_value=1000):
        begin_turn(agent, [])
        assert effective_request_overrides(agent)[field] == premium
        # Absent-policy control proves this real transport would otherwise send premium.
        wire = agent._build_api_kwargs([{"role": "user", "content": "fixture"}])
        assert (wire.get("extra_body", {}) if anthropic else wire)[field] == premium
        agent._active_fallback_service_tier_override = "normal"
        wire = agent._build_api_kwargs([{"role": "user", "content": "fixture"}])
        assert field not in wire
        assert field not in (wire.get("extra_body") or {})
        assert effective_request_overrides(agent)[field] == premium
        assert agent.service_tier == mode
    with patch("agent.fast_mode.time.monotonic", return_value=2000):
        assert field not in effective_request_overrides(agent)
        assert field not in agent._build_api_kwargs([{"role": "user", "content": "fixture"}])


@pytest.mark.parametrize("value", ["", "priority", "flex", True, 3, [], {}])
def test_invalid_activation_warns_clears_previous_policy_and_preserves_ordinary_overrides(value, caplog):
    from agent.fallback_policy import activate_fallback_service_tier_override
    from agent.fallback_policy import apply_fallback_service_tier_override
    import logging
    from types import SimpleNamespace

    agent = SimpleNamespace(_active_fallback_service_tier_override="normal")
    log = logging.getLogger(__name__)
    assert activate_fallback_service_tier_override(agent, {"service_tier_override": value}, log=log) is None
    assert agent._active_fallback_service_tier_override is None
    assert "Ignoring unsupported fallback service_tier_override" in caplog.text
    overrides = {"speed": "fast", "extra_body": {"service_tier": "priority"}}
    effective = apply_fallback_service_tier_override(overrides, agent._active_fallback_service_tier_override)
    assert effective == overrides
    assert effective["extra_body"] is not overrides["extra_body"]


def test_normalize_and_empty_nested_body_contract():
    from agent.fallback_policy import normalize_fallback_service_tier_override
    from agent.fallback_policy import apply_fallback_service_tier_override

    assert normalize_fallback_service_tier_override(" NORMAL ") == "normal"
    assert normalize_fallback_service_tier_override(None) is None
    assert apply_fallback_service_tier_override(None, "normal") == {}
    assert apply_fallback_service_tier_override({"extra_body": {"speed": "fast"}}, "normal") == {"extra_body": {}}
    assert apply_fallback_service_tier_override({"extra_body": "unchanged"}, "normal") == {"extra_body": "unchanged"}

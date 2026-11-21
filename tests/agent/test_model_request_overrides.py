"""``model.request_overrides`` (config.yaml) → ``agent.request_overrides``.

Mirrors ``delegation.request_overrides`` for the main loop. Values land on every request's
API kwargs, so an out-of-range or non-numeric sampling value must be dropped with a warning
instead of producing an opaque provider 400.
"""
from types import SimpleNamespace

import pytest

from agent.agent_init import _merge_model_request_overrides


def _agent(model="deepseek-flash", request_overrides=None, provider="custom"):
    return SimpleNamespace(
        model=model,
        base_url="https://ai.example.com/v1",
        provider=provider,
        request_overrides=dict(request_overrides or {}),
    )


def test_temperature_is_applied():
    agent = _agent()
    _merge_model_request_overrides(agent, {"request_overrides": {"temperature": 0.4}})
    assert agent.request_overrides["temperature"] == 0.4


def test_top_p_is_applied_and_coerced_to_float():
    agent = _agent()
    _merge_model_request_overrides(agent, {"request_overrides": {"top_p": 1}})
    assert agent.request_overrides["top_p"] == 1.0
    assert isinstance(agent.request_overrides["top_p"], float)


@pytest.mark.parametrize("bad", [2.5, -0.1, "0.4", True, None])
def test_out_of_range_or_non_numeric_temperature_is_ignored(bad, caplog):
    agent = _agent()
    with caplog.at_level("WARNING"):
        _merge_model_request_overrides(agent, {"request_overrides": {"temperature": bad}})
    assert "temperature" not in agent.request_overrides


def test_out_of_range_top_p_is_ignored():
    agent = _agent()
    _merge_model_request_overrides(agent, {"request_overrides": {"top_p": 2.5}})
    assert "top_p" not in agent.request_overrides


def test_extra_body_is_deep_merged_over_provider_extras():
    agent = _agent(request_overrides={"extra_body": {"provider": {"sort": "throughput"}}})
    _merge_model_request_overrides(
        agent, {"request_overrides": {"extra_body": {"reasoning": {"enabled": False}}}}
    )
    assert agent.request_overrides["extra_body"] == {
        "provider": {"sort": "throughput"},
        "reasoning": {"enabled": False},
    }


def test_explicit_top_level_keys_win_over_existing_overrides():
    agent = _agent(request_overrides={"service_tier": "priority", "temperature": 1.0})
    _merge_model_request_overrides(
        agent, {"request_overrides": {"service_tier": "default", "temperature": 0.2}}
    )
    assert agent.request_overrides["service_tier"] == "default"
    assert agent.request_overrides["temperature"] == 0.2


def test_model_that_manages_temperature_server_side_wins(caplog):
    # Kimi keeps temperature server-side: the OMIT_TEMPERATURE contract must not be broken.
    agent = _agent(model="kimi-k2-0905-preview")
    with caplog.at_level("WARNING"):
        _merge_model_request_overrides(
            agent, {"request_overrides": {"temperature": 0.3, "top_p": 0.9}},
        )
    assert "temperature" not in agent.request_overrides
    assert agent.request_overrides["top_p"] == 0.9  # unrelated keys still apply


def test_no_config_is_a_noop():
    agent = _agent(request_overrides={"temperature": 1.0})
    _merge_model_request_overrides(agent, None)
    _merge_model_request_overrides(agent, {})
    _merge_model_request_overrides(agent, {"request_overrides": {}})
    assert agent.request_overrides == {"temperature": 1.0}


def test_empty_overrides_after_validation_leave_agent_untouched():
    agent = _agent()
    _merge_model_request_overrides(agent, {"request_overrides": {"temperature": 9.9}})
    assert agent.request_overrides == {}

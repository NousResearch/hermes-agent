"""``route_change_callback`` fires once a fallback switch or primary restore has installed the new route.

A viewer's status bar reads ``session.info.model``. Without this hook the label only refreshed at
turn end, so a long turn ran for half an hour on the fallback while the label named the dead
primary (#78613). The callback must see the new ``model``/``provider``, must not fire for a
switch or restore that did not happen, and must never break the switch it reports.
"""

import time
from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason
from run_agent import AIAgent

PRIMARY_MODEL = "anthropic/claude-opus-4.6"
FALLBACK = {"provider": "openai", "model": "gpt-4o"}


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", provider="openrouter",
            model=PRIMARY_MODEL, quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        agent._rate_limited_until = 0
        return agent


def _client():
    mock = MagicMock()
    mock.base_url = "https://api.openai.com/v1"
    mock.api_key = "fb-key"
    return mock


def _resolving(**kw):
    return patch("agent.auxiliary_client.resolve_provider_client", return_value=(_client(), "resolved"), **kw)


def _recorder(agent):
    seen = []
    agent.route_change_callback = lambda *args: seen.append((args, agent.model, agent.provider))
    return seen


def test_agent_declares_the_callback_unset():
    assert _make_agent().route_change_callback is None


def test_successful_switch_calls_back_once_on_the_fallback_route():
    agent = _make_agent(fallback_model=[FALLBACK])
    seen = _recorder(agent)
    with _resolving():
        assert agent._try_activate_fallback() is True
    assert seen == [((), "gpt-4o", "openai")]


def test_exhausted_chain_does_not_call_back():
    agent = _make_agent(fallback_model=[])
    seen = _recorder(agent)
    assert agent._try_activate_fallback() is False
    assert seen == []


def test_a_candidate_that_raises_during_activation_does_not_call_back():
    agent = _make_agent(fallback_model=[FALLBACK])
    seen = _recorder(agent)
    with _resolving(side_effect=RuntimeError("client build failed")):
        assert agent._try_activate_fallback() is False
    assert seen == []


def test_a_switch_deferred_by_the_reset_window_does_not_call_back():
    agent = _make_agent(fallback_model=[FALLBACK])
    seen = _recorder(agent)
    with patch("agent.fallback_cooldown.switch_deferred_by_reset", return_value=True), _resolving():
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit) is False
    assert seen == [] and agent.model == PRIMARY_MODEL


def _on_fallback():
    agent = _make_agent(fallback_model=[FALLBACK])
    with _resolving():
        assert agent._try_activate_fallback() is True
    assert agent.model == "gpt-4o"
    agent._rate_limited_until = 0
    return agent


def test_successful_restore_calls_back_once_on_the_primary_route():
    agent = _on_fallback()
    seen = _recorder(agent)
    assert agent._restore_primary_runtime() is True
    assert seen == [((), PRIMARY_MODEL, "openrouter")]


def test_restore_blocked_by_the_rate_limit_cooldown_does_not_call_back():
    agent = _on_fallback()
    seen = _recorder(agent)
    agent._rate_limited_until = time.monotonic() + 600
    assert agent._restore_primary_runtime() is False
    assert seen == [] and agent.model == "gpt-4o"


def _raising(*_args):
    raise RuntimeError("viewer gone")


def test_a_raising_callback_keeps_the_switch_and_its_true_result():
    agent = _make_agent(fallback_model=[FALLBACK])
    agent.route_change_callback = _raising
    with _resolving():
        assert agent._try_activate_fallback() is True
    assert (agent.model, agent.provider) == ("gpt-4o", "openai")


def test_a_raising_callback_keeps_the_restore_and_its_true_result():
    agent = _on_fallback()
    agent.route_change_callback = _raising
    assert agent._restore_primary_runtime() is True
    assert (agent.model, agent.provider) == (PRIMARY_MODEL, "openrouter")


def test_a_restore_without_a_provider_fallback_calls_back_once():
    # ``/model --once`` restoration (tui_gateway/model_switch.py) and a voice turn route
    # (agent/voice_turn_route.py) set ``_fallback_activated`` with no provider fallback active.
    agent = _on_fallback()
    agent._provider_fallback_active = False
    seen = _recorder(agent)
    assert agent._restore_primary_runtime() is True
    assert seen == [((), PRIMARY_MODEL, "openrouter")]
